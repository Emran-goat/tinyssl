"""Baseline training for comparison methods. Ponytail ultra."""

import sys
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import timm
from timm.data import create_transform
from timm.loss import LabelSmoothingCrossEntropy


def parse_flags():
    flags = {
        "dataset_name": "flowers102",
        "model_name": "resnet18",
        "output_dir": "./results/baselines",
        "epochs": 100,
        "batch_size": 64,
        "lr": 3e-4,
    }
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        key = args[i].lstrip("-")
        if key in flags and i + 1 < len(args):
            val = args[i + 1]
            if key in ("epochs", "batch_size"):
                val = int(val)
            elif key == "lr":
                val = float(val)
            flags[key] = val
            i += 2
        else:
            print(f"Unknown flag: {args[i]}")
            sys.exit(1)
    return flags


MODEL_MAP = {
    "vit_tiny": "vit_tiny_patch16_224",
    "vit_small": "vit_small_patch16_224",
    "cct_7_2x2": "cct_7_2x2_224",
    "convnext_tiny": "convnext_tiny",
    "resnet18": "resnet18",
}

IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]


class ContrastiveView(Dataset):
    """Wraps a dataset to produce two augmented views per image for SimCLR/BYOL."""

    def __init__(self, base_dataset, img_size=224):
        self.base = base_dataset
        self.view1 = create_transform(
            input_size=(3, img_size, img_size), is_training=True,
            color_jitter=0.4, auto_augment="rand-m9-mstd0.5-inc1",
            interpolation="bicubic", re_prob=0.25, re_mode="pixel", re_count=1,
        )
        self.view2 = create_transform(
            input_size=(3, img_size, img_size), is_training=True,
            color_jitter=0.4, auto_augment="rand-m9-mstd0.5-inc1",
            interpolation="bicubic", re_prob=0.25, re_mode="pixel", re_count=1,
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return self.view1(img), self.view2(img), label


def load_dataset(name, img_size=224):
    """Return train_loader, val_loader, num_classes."""
    data_root = Path("./data") / name

    if name == "breastmnist":
        import medmnist
        from medmnist import BreastMnist

        train_transform = create_transform(
            input_size=(3, img_size, img_size), is_training=True,
            color_jitter=0.4, auto_augment="rand-m9-mstd0.5-inc1",
            interpolation="bicubic",
        )
        val_transform = create_transform(
            input_size=(3, img_size, img_size), is_training=False, interpolation="bicubic"
        )
        train_ds = BreastMnist(split="train", transform=train_transform, download=True, root=str(data_root))
        val_ds = BreastMnist(split="test", transform=val_transform, download=True, root=str(data_root))
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
        return train_loader, val_loader, 2

    # ImageFolder-style datasets
    train_transform = create_transform(
        input_size=(3, img_size, img_size), is_training=True,
        color_jitter=0.4, auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic", re_prob=0.25, re_mode="pixel", re_count=1,
    )
    val_transform = create_transform(
        input_size=(3, img_size, img_size), is_training=False, interpolation="bicubic"
    )

    from torchvision.datasets import ImageFolder
    train_ds = ImageFolder(data_root / "train", transform=train_transform)
    val_ds = ImageFolder(data_root / "val", transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, len(train_ds.classes)


def make_model(name, num_classes, device):
    if name in MODEL_MAP:
        return timm.create_model(MODEL_MAP[name], pretrained=False, num_classes=num_classes).to(device)
    if name == "simclr":
        return SimCLR(num_classes=num_classes).to(device)
    if name == "byol":
        return BYOL(num_classes=num_classes).to(device)
    raise ValueError(f"Unknown model: {name}")


# ── SimCLR (contrastive pre-training then linear probe) ──────────────────────

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden=256, out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out),
        )

    def forward(self, x):
        return self.net(x)


class SimCLR(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model("resnet18", pretrained=False, num_classes=0)
        feat_dim = self.backbone.num_features
        self.projector = ProjectionHead(feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.out_dim = feat_dim

    def forward(self, x):
        if self.training and x.dim() == 5:
            # x: [B, 2, C, H, W] — two views
            B = x.size(0)
            views = x.view(B * 2, *x.shape[2:])
            feats = self.backbone(views)
            proj = self.projector(feats)
            return proj.view(B, 2, -1), feats.view(B, 2, -1)
        return self.backbone(x)


def nt_xent_loss(z1, z2, temperature=0.5):
    """Normalized temperature-scaled cross-entropy."""
    B = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    sim = z @ z.T / temperature
    # mask out self-similarity
    mask = ~torch.eye(2 * B, dtype=bool, device=sim.device)
    sim = sim.masked_select(mask).view(2 * B, -1)
    # positive pairs: i ↔ i+B
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z1.device)
    return F.cross_entropy(sim, labels)


def train_simclr_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0, 0
    for images, _ in loader:
        images = images.to(device)
        if images.dim() == 4:
            # fallback: apply random augment twice inline
            images = torch.stack([images, images], dim=1)
        (proj1, proj2), _ = model(images)
        loss = nt_xent_loss(proj1, proj2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return total_loss / n


def train_linear_probe(model, loader, criterion, optimizer, device):
    """Freeze backbone, train classifier head only."""
    model.backbone.eval()
    model.projector.eval()
    model.classifier.train()
    total_loss, correct, total = 0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        with torch.no_grad():
            feats = model.backbone(images)
        logits = model.classifier(feats)
        loss = criterion(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


# ── BYOL (bootstrap self-learning) ──────────────────────────────────────────

class BYOL(nn.Module):
    def __init__(self, num_classes, momentum=0.996):
        super().__init__()
        self.backbone = timm.create_model("resnet18", pretrained=False, num_classes=0)
        feat_dim = self.backbone.num_features
        self.projector = ProjectionHead(feat_dim)
        self.predictor = ProjectionHead(128, 256, 128)
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.out_dim = feat_dim

        # EMA target
        self.backbone_target = timm.create_model("resnet18", pretrained=False, num_classes=0)
        self.projector_target = ProjectionHead(feat_dim)
        for p in self.backbone_target.parameters():
            p.requires_grad = False
        for p in self.projector_target.parameters():
            p.requires_grad = False
        self._momentum = momentum
        self._update_target()

    @torch.no_grad()
    def _update_target(self):
        for ps, pt in zip(self.backbone.parameters(), self.backbone_target.parameters()):
            pt.data = self._momentum * pt.data + (1 - self._momentum) * ps.data
        for ps, pt in zip(self.projector.parameters(), self.projector_target.parameters()):
            pt.data = self._momentum * pt.data + (1 - self._momentum) * ps.data

    def forward(self, x):
        if self.training and x.dim() == 5:
            B = x.size(0)
            views = x.view(B * 2, *x.shape[2:])
            feats = self.backbone(views)
            proj = self.projector(feats)
            pred = self.predictor(proj)
            return pred.view(B, 2, -1), feats.view(B, 2, -1)
        return self.backbone(x)


def byol_loss(p, z):
    """Negative cosine similarity (no negatives needed)."""
    return 2 - 2 * F.cosine_similarity(p, z, dim=-1).mean()


def train_byol_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0, 0
    for images, _ in loader:
        images = images.to(device)
        if images.dim() == 4:
            images = torch.stack([images, images], dim=1)

        B = images.size(0)
        views = images.view(B * 2, *images.shape[2:])

        feats_online = model.backbone(views)
        proj_online = model.projector(feats_online)
        pred_online = model.predictor(proj_online)

        with torch.no_grad():
            feats_target = model.backbone_target(views)
            proj_target = model.projector_target(feats_target)

        pred_online = pred_online.view(B, 2, -1)
        proj_target = proj_target.view(B, 2, -1)

        loss = byol_loss(pred_online[:, 0], proj_target[:, 1]) + byol_loss(pred_online[:, 1], proj_target[:, 0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model._update_target()

        total_loss += loss.item() * B
        n += B
    return total_loss / n


# ── Standard supervised training ────────────────────────────────────────────

def train_supervised_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        correct += (model(images).argmax(1) == targets).sum().item()
        total += targets.size(0)
    return correct / total


@torch.no_grad()
def evaluate_linear(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        feats = model.backbone(images)
        logits = model.classifier(feats)
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.size(0)
    return correct / total


def main():
    flags = parse_flags()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(flags["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, num_classes = load_dataset(flags["dataset_name"])
    model = make_model(flags["model_name"], num_classes, device)

    csv_path = output_dir / f"{flags['model_name']}_{flags['dataset_name']}_metrics.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_acc"])

    is_ssl = flags["model_name"] in ("simclr", "byol")

    if is_ssl:
        # Phase 1: self-supervised pre-training
        pretrain_epochs = max(flags["epochs"] // 2, 10)
        ssl_loader = DataLoader(
            ContrastiveView(train_loader.dataset),
            batch_size=flags["batch_size"], shuffle=True, num_workers=4, pin_memory=True,
        )
        optimizer = AdamW(model.parameters(), lr=flags["lr"], weight_decay=0.05)
        total_iters = pretrain_epochs
        sched = CosineAnnealingLR(optimizer, T_max=total_iters, eta_min=1e-6)

        print(f"[SSL pre-train] {flags['model_name']} on {flags['dataset_name']} for {pretrain_epochs} epochs")
        for epoch in range(1, pretrain_epochs + 1):
            if flags["model_name"] == "simclr":
                loss = train_simclr_epoch(model, ssl_loader, optimizer, device)
            else:
                loss = train_byol_epoch(model, ssl_loader, optimizer, device)
            sched.step()
            print(f"  Epoch {epoch:3d}/{pretrain_epochs} | SSL Loss: {loss:.4f}")

        # Phase 2: linear probe
        probe_epochs = flags["epochs"] - pretrain_epochs
        optimizer = AdamW(model.classifier.parameters(), lr=flags["lr"])
        sched = CosineAnnealingLR(optimizer, T_max=probe_epochs, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss()

        print(f"[Linear probe] {probe_epochs} epochs")
        for epoch in range(1, probe_epochs + 1):
            train_loss, train_acc = train_linear_probe(model, train_loader, criterion, optimizer, device)
            val_acc = evaluate_linear(model, val_loader, device)
            sched.step()
            csv_writer.writerow([pretrain_epochs + epoch, train_loss, train_acc, val_acc])
            print(f"  Epoch {epoch:3d}/{probe_epochs} | Loss: {train_loss:.4f} | Train: {train_acc:.4f} | Val: {val_acc:.4f}")

    else:
        # Standard supervised training
        criterion = LabelSmoothingCrossEntropy()
        optimizer = AdamW(model.parameters(), lr=flags["lr"], weight_decay=0.05)
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=min(10, flags["epochs"]))
        cosine = CosineAnnealingLR(optimizer, T_max=flags["epochs"] - 10, eta_min=1e-6)
        sched = SequentialLR(optimizer, [warmup, cosine], milestones=[min(10, flags["epochs"])])

        best_acc = 0.0
        for epoch in range(1, flags["epochs"] + 1):
            train_loss, train_acc = train_supervised_epoch(model, train_loader, criterion, optimizer, device)
            val_acc = evaluate(model, val_loader, device)
            sched.step()
            csv_writer.writerow([epoch, train_loss, train_acc, val_acc])
            print(f"Epoch {epoch:3d}/{flags['epochs']} | Loss: {train_loss:.4f} | Train: {train_acc:.4f} | Val: {val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({"model": model.state_dict(), "epoch": epoch, "best_acc": best_acc},
                           output_dir / f"{flags['model_name']}_best.pt")

    csv_file.close()
    print(f"\nFinal accuracy: {val_acc:.4f} | Best: {best_acc:.4f if not is_ssl else val_acc:.4f}")
    print(f"Metrics saved to {csv_path}")


if __name__ == "__main__":
    main()

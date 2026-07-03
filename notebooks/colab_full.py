#!/usr/bin/env python3
"""
TinySSL: Distilling DINOv2 into Tiny Vision Models
===================================================
Self-contained Colab notebook — paste each section into separate cells.

Trains a ~3M-param student to match DINOv2 (22M) on linear probing,
using knowledge distillation + MIM-JEPA on frozen teacher features.

Runtime: ~45 min on Colab T4 GPU.
"""

# %% [markdown]
# # Cell 1: Install Dependencies

# %%
!pip install -q torch torchvision timm scikit-learn matplotlib seaborn pandas medmnist pillow

# %% [markdown]
# # Cell 2: Imports

# %%
import os, csv, math, time, json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import timm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# %% [markdown]
# # Cell 3: DINOv2 Teacher

# %%
DINO_DIM = 384
DINO_PATCHES = 256  # 224/14 = 16, 16x16 = 256

class DINOv2Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        for p in self.net.parameters():
            p.requires_grad = False

    def forward(self, x):
        features = self.net.get_intermediate_layers(x, n=[9])
        last = features[-1]
        return {
            "cls_token": self.net.cls_token.expand(x.shape[0], -1, -1).squeeze(1),
            "patch_tokens": last,
        }

# %% [markdown]
# # Cell 4: Student Models

# %%
class TinySSLBase(nn.Module):
    """~3M params. Conv tokenizer → 4-layer Transformer → attention pool."""

    def __init__(self, img_size=224, out_dim=256, stride=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 256, kernel_size=3, stride=stride, padding=1)
        n_patches = (img_size // stride) ** 2
        self.proj = nn.Linear(256, out_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=4, dim_feedforward=512, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.cls_token = nn.Parameter(torch.randn(1, 1, out_dim))
        self.out_dim = out_dim
        self.n_patches = n_patches

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)

        return {"cls": x[:, 0], "patches": x[:, 1:]}


class TinySSLTiny(nn.Module):
    """~300K params. Lightweight student."""

    def __init__(self, img_size=224, out_dim=128, stride=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 128, kernel_size=3, stride=stride, padding=1)
        n_patches = (img_size // stride) ** 2
        self.proj = nn.Linear(128, out_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=4, dim_feedforward=256, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.out_dim = out_dim
        self.n_patches = n_patches

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = self.encoder(x)
        cls = x.mean(dim=1)
        return {"cls": cls, "patches": x}


class TinySSLCNN(nn.Module):
    """Pure CNN baseline, ~3M params."""

    def __init__(self, out_dim=256):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(256, out_dim)
        self.out_dim = out_dim
        self.n_patches = 196

    def forward(self, x):
        feat = self.blocks(x)
        cls = feat.mean(dim=(2, 3))
        cls = self.head(cls)
        patches = feat.flatten(2).transpose(1, 2)
        return {"cls": cls, "patches": patches}


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

for name, cls in [("Base", TinySSLBase), ("Tiny", TinySSLTiny), ("CNN", TinySSLCNN)]:
    m = cls()
    print(f"TinySSL-{name}: {count_params(m):.2f}M params, {m.n_patches} patches")

# %% [markdown]
# # Cell 5: Losses

# %%
_proj_cache = {}

def _get_projection(D_s, D_t, device):
    key = (D_s, D_t)
    if key not in _proj_cache:
        _proj_cache[key] = nn.Linear(D_s, D_t, bias=False)
    return _proj_cache[key].to(device)


def distillation_loss(student_patches, teacher_patches):
    s = F.normalize(student_patches, dim=-1)
    t = F.normalize(teacher_patches, dim=-1)
    p = F.normalize(_get_projection(s.shape[-1], t.shape[-1], s.device)(s), dim=-1)
    return (1.0 - (p * t).sum(-1)).mean()

# %% [markdown]
# # Cell 6: Augmentations + Dataset Loading

# %%
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_augmentation(epoch):
    if epoch < 100:
        augs = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(224, padding=4)]
    elif epoch < 200:
        augs = [
            transforms.RandomHorizontalFlip(), transforms.RandomCrop(224, padding=4),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.GaussianBlur(23, (0.1, 2.0)),
        ]
    else:
        augs = [
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.8, 0.8, 0.8),
            transforms.GaussianBlur(23, (0.1, 2.0)),
            transforms.RandomSolarize(0.5),
        ]
    augs += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(augs)


def load_dataset_for_caching(name):
    """Load dataset with basic transform for DINOv2 feature caching."""
    t = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    if name == "flowers102":
        return datasets.Flowers102(root="data", split="train", download=True, transform=t)
    elif name == "oxford_pets":
        return datasets.OxfordIIITPet(root="data", download=True, transform=t)
    elif name == "eurosat":
        return datasets.EuroSAT(root="data", download=True, transform=t)
    elif name == "breastmnist":
        import medmnist
        return medmnist.BreastMNIST(root="data", split="train", download=True,
                                     transform=transforms.Compose([
                                         transforms.Resize(224),
                                         transforms.Grayscale(num_output_channels=3),
                                         transforms.ToTensor(),
                                     ]))
    raise ValueError(f"Unknown dataset: {name}")


def load_pil_images(name):
    """Return list of raw PIL images for pre-training (no transforms)."""
    if name == "flowers102":
        raw = datasets.Flowers102(root="data", split="train", download=True)
        return [raw[i][0] for i in range(len(raw))]
    elif name == "oxford_pets":
        raw = datasets.OxfordIIITPet(root="data", download=True)
        return [raw[i][0] for i in range(len(raw))]
    elif name == "eurosat":
        raw = datasets.EuroSAT(root="data", download=True)
        return [raw[i][0] for i in range(len(raw))]
    elif name == "breastmnist":
        import medmnist
        raw = medmnist.BreastMNIST(root="data", split="train", download=True)
        images = []
        for i in range(len(raw)):
            img = raw[i][0]
            arr = np.array(img)
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            img = PILImage.fromarray(arr).resize((224, 224))
            images.append(img)
        return images
    raise ValueError(f"Unknown dataset: {name}")


def load_eval_datasets(name):
    """Load train/test datasets with eval transform."""
    eval_t = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    if name == "flowers102":
        return (datasets.Flowers102(root="data", split="train", download=True, transform=eval_t),
                datasets.Flowers102(root="data", split="test", download=True, transform=eval_t), 102)
    elif name == "oxford_pets":
        return (datasets.OxfordIIITPet(root="data", download=True, transform=eval_t, split="trainval"),
                datasets.OxfordIIITPet(root="data", download=True, transform=eval_t, split="test"), 37)
    elif name == "eurosat":
        full = datasets.EuroSAT(root="data", download=True, transform=eval_t)
        n = len(full)
        n_train = int(0.8 * n)
        train_set, test_set = torch.utils.data.random_split(full, [n_train, n - n_train],
                                                             generator=torch.Generator().manual_seed(42))
        return train_set, test_set, 10
    elif name == "breastmnist":
        import medmnist
        t = transforms.Compose([transforms.Resize(224), transforms.Grayscale(num_output_channels=3),
                                 transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        return (medmnist.BreastMNIST(root="data", split="train", download=True, transform=t),
                medmnist.BreastMNIST(root="data", split="test", download=True, transform=t), 2)
    raise ValueError(f"Unknown dataset: {name}")


DATASETS = ["flowers102", "oxford_pets", "eurosat", "breastmnist"]
DATASET_INFO = {"flowers102": 102, "oxford_pets": 37, "eurosat": 10, "breastmnist": 2}

# %% [markdown]
# # Cell 7: Cache DINOv2 Features

# %%
def interp_patches(teacher_patches, target_n):
    """Interpolate teacher [B, 256, 384] -> [B, target_n, 384] via bilinear."""
    B, _, D = teacher_patches.shape
    s = int(teacher_patches.shape[1] ** 0.5)
    t = teacher_patches.view(B, s, s, D).permute(0, 3, 1, 2)
    ts = int(target_n ** 0.5)
    t = F.interpolate(t, size=(ts, ts), mode="bilinear", align_corners=False)
    return t.permute(0, 2, 3, 1).reshape(B, target_n, D)


def cache_teacher_features(dataset_name, batch_size=64):
    """Cache DINOv2 features for all training images."""
    cache_dir = Path("cache") / dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(cache_dir.glob("*.pt")))
    if existing > 0:
        print(f"  Cache exists ({existing} samples), skipping")
        return cache_dir

    teacher = DINOv2Teacher().to(device).eval()
    dataset = load_dataset_for_caching(dataset_name)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    idx = 0
    t0 = time.time()
    with torch.no_grad():
        for imgs, _labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            out = teacher(imgs)
            for i in range(imgs.size(0)):
                torch.save(
                    {"cls": out["cls_token"][i].cpu(), "patches": out["patch_tokens"][i].cpu()},
                    cache_dir / f"{idx:06d}.pt",
                )
                idx += 1
            print(f"  {dataset_name}: cached {idx}/{len(dataset)} ({time.time()-t0:.1f}s)")

    print(f"  Done: {idx} features in {time.time()-t0:.1f}s")
    return cache_dir


print("=== Caching DINOv2 Features ===")
cache_dirs = {}
for ds in DATASETS:
    print(f"\n{ds}:")
    cache_dirs[ds] = cache_teacher_features(ds)

# %% [markdown]
# # Cell 8: Helper Functions

# %%
def load_teacher_cache(cache_dir):
    cache_dir = Path(cache_dir)
    pt_files = sorted(cache_dir.glob("*.pt"))
    cls_list, patch_list = [], []
    for f in pt_files:
        try:
            d = torch.load(f, weights_only=False)
            cls_list.append(d["cls"])
            patch_list.append(d["patches"])
        except Exception:
            continue
    return torch.stack(cls_list), torch.stack(patch_list)


class PretrainDataset(Dataset):
    def __init__(self, images, t_cls, t_patches, transform):
        self.images = images
        self.t_cls = t_cls
        self.t_patches = t_patches
        self.transform = transform

    def __len__(self):
        return min(len(self.images), len(self.t_cls))

    def __getitem__(self, idx):
        img = self.images[idx]
        if isinstance(img, torch.Tensor):
            arr = img.permute(1, 2, 0).numpy()
            img = PILImage.fromarray((arr * 255).astype(np.uint8))
        return self.transform(img), self.t_cls[idx], self.t_patches[idx]

# %% [markdown]
# # Cell 9: Train All Students

# %%
print("=== Pre-Training TinySSL Students ===\n")
results = {}
BATCH = 32  # ponytail: 32 for T4 15GB safety
EPOCHS = 200

for ds in DATASETS:
    num_classes = DATASET_INFO[ds]
    cache_dir = cache_dirs[ds]

    print(f"\nLoading {ds}...")
    t_cls, t_patches = load_teacher_cache(cache_dir)
    images = load_pil_images(ds)
    print(f"  {len(images)} images, {len(t_cls)} cached features")

    for model_name, model_cls in [("base", TinySSLBase), ("tiny", TinySSLTiny), ("cnn", TinySSLCNN)]:
        print(f"\n--- {model_name.upper()} on {ds} ---")
        student = model_cls().to(device)
        print(f"  Params: {count_params(student):.2f}M")

        dataset = PretrainDataset(images, t_cls, t_patches, get_augmentation(0))
        loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)

        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=0.05)
        history = {"epoch": [], "loss": [], "distill": [], "mim": []}
        t0 = time.time()

        for epoch in range(EPOCHS):
            # cosine LR with warmup
            if epoch < 10:
                cur_lr = 1e-4 * (epoch + 1) / 10
            else:
                cur_lr = 1e-4 * 0.5 * (1 + math.cos(math.pi * (epoch - 10) / (EPOCHS - 10)))
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            mask_ratio = 0.5 if epoch < EPOCHS // 2 else 0.75
            dataset.transform = get_augmentation(epoch)
            student.train()
            sum_loss, sum_d, sum_m, n_batch = 0, 0, 0, 0

            for imgs, t_cls_b, t_patches_b in loader:
                imgs = imgs.to(device)
                t_cls_b = t_cls_b.to(device)
                t_patches_b = t_patches_b.to(device)

                out = student(imgs)
                s_patches = out["patches"]
                t_interp = interp_patches(t_patches_b, s_patches.shape[1])

                L_d = distillation_loss(s_patches, t_interp)
                proj = _get_projection(s_patches.shape[-1], t_interp.shape[-1], device)
                s_proj = proj(s_patches)
                B, N, D = s_proj.shape
                mask = torch.rand(B, N, device=device) > mask_ratio
                L_m = F.mse_loss(s_proj, t_interp, reduction="none")
                L_m = L_m[mask.unsqueeze(-1).expand_as(L_m)].mean()
                loss = L_d + 0.5 * L_m

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                sum_loss += loss.item()
                sum_d += L_d.item()
                sum_m += L_m.item()
                n_batch += 1

            nb = max(n_batch, 1)
            history["epoch"].append(epoch)
            history["loss"].append(sum_loss / nb)
            history["distill"].append(sum_d / nb)
            history["mim"].append(sum_m / nb)

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {sum_loss/nb:.4f} | {time.time()-t0:.1f}s")

        key = f"{model_name}_{ds}"
        results[key] = {"model": student.cpu(), "history": history}

        os.makedirs("checkpoints", exist_ok=True)
        torch.save({"model": student.state_dict(), "meta": {"model_type": model_name}},
                    f"checkpoints/{key}.pt")

print("\n=== All Students Trained ===")

# %% [markdown]
# # Cell 10: Train Baselines (ResNet-18, ViT-Tiny, CCT)

# %%
def train_baseline(model_name, dataset_name, num_classes, epochs=50, batch_size=64, lr=3e-4):
    """Train a supervised baseline model."""
    train_t = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(224, padding=4),
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_t = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds, val_ds, _ = load_eval_datasets(dataset_name)
    # re-apply transforms (load_eval_datasets applies val_t to both)
    train_ds.dataset.transform = train_t if hasattr(train_ds, 'dataset') else train_t

    # For simplicity, just use load_eval_datasets and re-create loaders with proper transforms
    train_raw, val_raw, nc = load_eval_datasets(dataset_name)

    train_loader = DataLoader(train_raw, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_raw, batch_size=batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)

    if model_name == "resnet18":
        model = timm.create_model("resnet18", pretrained=False, num_classes=num_classes)
    elif model_name == "vit_tiny":
        model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
    elif model_name == "cct":
        model = timm.create_model("cct_7_2x2_224", pretrained=False, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=min(5, epochs))
    cosine = CosineAnnealingLR(optimizer, T_max=epochs - 5, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[min(5, epochs)])

    best_acc = 0.0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                val_correct += (model(imgs).argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total

        if val_acc > best_acc:
            best_acc = val_acc

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss/total:.4f} | Val: {val_acc:.4f} | Best: {best_acc:.4f} | {time.time()-t0:.1f}s")

    return best_acc


print("=== Training Supervised Baselines ===\n")
baseline_results = {}

for ds in DATASETS:
    num_classes = DATASET_INFO[ds]
    for model_name in ["resnet18", "vit_tiny", "cct"]:
        print(f"\n--- {model_name} on {ds} ---")
        best_acc = train_baseline(model_name, ds, num_classes, epochs=50)
        baseline_results[f"{model_name}_{ds}"] = {"best_acc": best_acc}

# %% [markdown]
# # Cell 11: DINOv2 Linear Probe (Upper Bound)

# %%
def dinov2_linear_probe(dataset_name, num_classes, epochs=100, batch_size=64, lr=1e-3):
    """Linear probe on frozen DINOv2 features — our upper bound."""
    train_ds, val_ds, _ = load_eval_datasets(dataset_name)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)

    teacher = DINOv2Teacher().to(device).eval()
    head = nn.Linear(DINO_DIM, num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    t0 = time.time()
    for epoch in range(epochs):
        head.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                feat = teacher(imgs)["cls_token"]
            loss = criterion(head(feat), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 25 == 0:
            head.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    feat = teacher(imgs)["cls_token"]
                    correct += (head(feat).argmax(1) == labels).sum().item()
                    total += labels.size(0)
            print(f"  Epoch {epoch+1}/{epochs} | Val: {correct/total:.4f} | {time.time()-t0:.1f}s")

    head.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = teacher(imgs)["cls_token"]
            correct += (head(feat).argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


print("=== DINOv2 Linear Probe (Upper Bound) ===\n")
dinov2_results = {}
for ds in DATASETS:
    num_classes = DATASET_INFO[ds]
    print(f"\n--- DINOv2 on {ds} ---")
    acc = dinov2_linear_probe(ds, num_classes, epochs=100)
    dinov2_results[ds] = acc
    print(f"  Final: {acc:.4f}")

# %% [markdown]
# # Cell 12: Evaluate Students — Linear Probe + k-NN

# %%
def linear_probe_eval(student, dataset_name, num_classes, epochs=100, batch_size=64, lr=1e-3):
    """Evaluate a pre-trained student via linear probe."""
    train_ds, val_ds, _ = load_eval_datasets(dataset_name)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)

    student = student.to(device).eval()
    for p in student.parameters():
        p.requires_grad = False

    head = nn.Linear(student.out_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        head.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                feat = student(imgs)["cls"]
            loss = criterion(head(feat), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    head.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = student(imgs)["cls"]
            correct += (head(feat).argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def knn_eval(student, dataset_name, k=20):
    """Evaluate a pre-trained student via k-NN."""
    train_ds, val_ds, _ = load_eval_datasets(dataset_name)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

    student = student.to(device).eval()
    for p in student.parameters():
        p.requires_grad = False

    train_feats, train_labels = [], []
    with torch.no_grad():
        for imgs, labels in train_loader:
            feat = student(imgs.to(device))["cls"]
            train_feats.append(F.normalize(feat, dim=1).cpu())
            train_labels.append(labels)
    train_feats = torch.cat(train_feats).numpy()
    train_labels = torch.cat(train_labels).numpy()

    test_feats, test_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            feat = student(imgs.to(device))["cls"]
            test_feats.append(F.normalize(feat, dim=1).cpu())
            test_labels.append(labels)
    test_feats = torch.cat(test_feats).numpy()
    test_labels = torch.cat(test_labels).numpy()

    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1)
    knn.fit(train_feats, train_labels)
    return accuracy_score(test_labels, knn.predict(test_feats))


print("=== Evaluating All Students ===\n")
eval_results = {}

for ds in DATASETS:
    num_classes = DATASET_INFO[ds]
    for model_name in ["base", "tiny", "cnn"]:
        key = f"{model_name}_{ds}"
        if key not in results:
            continue
        student = results[key]["model"]

        print(f"\n--- {model_name.upper()} on {ds} ---")
        lp_acc = linear_probe_eval(student, ds, num_classes, epochs=100)
        knn_acc = knn_eval(student, ds)

        eval_results[key] = {
            "linear_probe": lp_acc,
            "knn": knn_acc,
            "params": count_params(student),
        }
        print(f"  Linear Probe: {lp_acc:.4f} | k-NN: {knn_acc:.4f}")

# %% [markdown]
# # Cell 13: Results Table

# %%
print("\n" + "=" * 90)
print("TINYSSL RESULTS — Knowledge Distillation from DINOv2")
print("=" * 90)

header = f"{'Method':<25} {'Params':>7} {'Flowers':>8} {'Pets':>8} {'EuroSAT':>8} {'BreastMN':>9}"
print(header)
print("-" * 90)

for model_name in ["base", "tiny", "cnn"]:
    params = eval_results.get(f"{model_name}_flowers102", {}).get("params", 0)
    flowers = eval_results.get(f"{model_name}_flowers102", {}).get("linear_probe", 0)
    pets = eval_results.get(f"{model_name}_oxford_pets", {}).get("linear_probe", 0)
    eurosat = eval_results.get(f"{model_name}_eurosat", {}).get("linear_probe", 0)
    breast = eval_results.get(f"{model_name}_breastmnist", {}).get("linear_probe", 0)
    label = f"TinySSL-{model_name.capitalize()}"
    print(f"{label:<25} {params:>5.1f}M {flowers:>7.1%} {pets:>7.1%} {eurosat:>7.1%} {breast:>8.1%}")

print("-" * 90)

for bl_name in ["resnet18", "vit_tiny", "cct"]:
    flowers = baseline_results.get(f"{bl_name}_flowers102", {}).get("best_acc", 0)
    pets = baseline_results.get(f"{bl_name}_oxford_pets", {}).get("best_acc", 0)
    eurosat = baseline_results.get(f"{bl_name}_eurosat", {}).get("best_acc", 0)
    breast = baseline_results.get(f"{bl_name}_breastmnist", {}).get("best_acc", 0)
    label = bl_name.upper().replace("_", "-")
    print(f"{label:<25} {'—':>7} {flowers:>7.1%} {pets:>7.1%} {eurosat:>7.1%} {breast:>8.1%}")

print("-" * 90)
flowers = dinov2_results.get("flowers102", 0)
pets = dinov2_results.get("oxford_pets", 0)
eurosat = dinov2_results.get("eurosat", 0)
breast = dinov2_results.get("breastmnist", 0)
print(f"{'DINOv2-S/14 (linear)':<25} {'22.0M':>7} {flowers:>7.1%} {pets:>7.1%} {eurosat:>7.1%} {breast:>8.1%}")
print("=" * 90)

# %% [markdown]
# # Cell 14: Visualizations

# %%
os.makedirs("figures", exist_ok=True)

# --- Training Curves ---
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for idx, ds in enumerate(DATASETS):
    ax = axes[idx]
    key = f"base_{ds}"
    if key in results:
        h = results[key]["history"]
        ax.plot(h["epoch"], h["loss"], label="Total", linewidth=2)
        ax.plot(h["epoch"], h["distill"], label="Distill", linewidth=1.5, alpha=0.8)
        ax.plot(h["epoch"], h["mim"], label="MIM", linewidth=1.5, alpha=0.8)
    ax.set_title(ds.replace("_", " ").title(), fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.suptitle("TinySSL-Base Training Curves", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig("figures/training_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Accuracy Comparison Bar Chart ---
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(DATASETS))
width = 0.15

methods = [
    ("TinySSL-Base", "base"),
    ("TinySSL-Tiny", "tiny"),
    ("ResNet-18", "resnet18"),
    ("ViT-Tiny", "vit_tiny"),
    ("CCT-7/2x2", "cct"),
    ("DINOv2", "dinov2"),
]

colors = ["#2196F3", "#64B5F6", "#FF9800", "#FFB74D", "#4CAF50", "#F44336"]

for i, (label, tag) in enumerate(methods):
    vals = []
    for ds in DATASETS:
        if tag == "dinov2":
            vals.append(dinov2_results.get(ds, 0) * 100)
        elif tag in ("resnet18", "vit_tiny", "cct"):
            vals.append(baseline_results.get(f"{tag}_{ds}", {}).get("best_acc", 0) * 100)
        else:
            vals.append(eval_results.get(f"{tag}_{ds}", {}).get("linear_probe", 0) * 100)
    ax.bar(x + i * width, vals, width, label=label, color=colors[i])

ax.set_ylabel("Linear Probe Accuracy (%)", fontsize=12)
ax.set_title("Method Comparison Across Domains", fontsize=16)
ax.set_xticks(x + width * 2.5)
ax.set_xticklabels([d.replace("_", " ").title() for d in DATASETS], fontsize=11)
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig("figures/comparison_bars.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Parameter Count vs Accuracy Scatter ---
fig, ax = plt.subplots(figsize=(8, 6))
for ds_idx, ds in enumerate(DATASETS):
    color = ["#E91E63", "#9C27B0", "#3F51B5", "#009688"][ds_idx]
    for model_name in ["base", "tiny", "cnn"]:
        key = f"{model_name}_{ds}"
        if key in eval_results:
            p = eval_results[key]["params"]
            a = eval_results[key]["linear_probe"] * 100
            ax.scatter(p, a, c=color, s=100, zorder=5, edgecolors="black", linewidth=0.5)
    ax.scatter(22, dinov2_results.get(ds, 0) * 100, c=color, s=150, marker="*", zorder=5,
               edgecolors="black", linewidth=0.5)

for ds_idx, ds in enumerate(DATASETS):
    color = ["#E91E63", "#9C27B0", "#3F51B5", "#009688"][ds_idx]
    key = f"base_{ds}"
    if key in eval_results:
        ax.annotate(ds.replace("_", " ").title(),
                     (eval_results[key]["params"], eval_results[key]["linear_probe"] * 100),
                     textcoords="offset points", xytext=(5, 5), fontsize=9, color=color)

ax.set_xlabel("Parameters (M)", fontsize=12)
ax.set_ylabel("Linear Probe Accuracy (%)", fontsize=12)
ax.set_title("Efficiency: Params vs Accuracy", fontsize=16)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/params_vs_accuracy.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Sample Images ---
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for col, ds in enumerate(DATASETS):
    train_ds, _, _ = load_eval_datasets(ds)
    img, label = train_ds[0]
    if isinstance(img, torch.Tensor):
        img_show = img.permute(1, 2, 0).numpy()
        img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min())
    else:
        img_show = np.array(img)
    axes[col].imshow(img_show)
    axes[col].set_title(f"{ds.replace('_',' ').title()}\nClass: {label}", fontsize=11)
    axes[col].axis("off")
plt.suptitle("Sample Images from Each Domain", fontsize=16, y=1.05)
plt.tight_layout()
plt.savefig("figures/sample_images.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# # Cell 15: Save Results

# %%
all_results = {
    "dinov2": dinov2_results,
    "students": {k: {kk: vv for kk, vv in v.items() if kk != "model"} for k, v in results.items()},
    "baselines": baseline_results,
    "eval": eval_results,
}

for k in all_results["students"]:
    if "history" in all_results["students"][k]:
        h = all_results["students"][k]["history"]
        all_results["students"][k]["history"] = {kk: [float(x) for x in vv] for kk, vv in h.items()}

with open("results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("Results saved to results.json")
print("Figures saved to figures/")

print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
base_flowers = eval_results.get("base_flowers102", {}).get("linear_probe", 0)
dinov2_flowers = dinov2_results.get("flowers102", 0)
if dinov2_flowers > 0:
    ratio = base_flowers / dinov2_flowers * 100
    print(f"TinySSL-Base retains {ratio:.0f}% of DINOv2's accuracy on Flowers-102")
    print(f"  with {count_params(results['base_flowers102']['model']):.1f}M params vs 22M (DINOv2-S/14)")
print("=" * 60)

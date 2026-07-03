"""Evaluation: linear probe, kNN, finetune. One file, three protocols."""
import sys
import os
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from tinyssl.models.students import TinySSLBase, TinySSLTiny, TinySSLCNN

# ponytail: manual argv, no argparse
model_path = cache_dir = dataset_name = output_dir = None
argv = sys.argv[1:]
i = 0
while i < len(argv):
    if argv[i] == "--model_path":   model_path = argv[i + 1]; i += 2
    elif argv[i] == "--cache_dir":  cache_dir = argv[i + 1];  i += 2
    elif argv[i] == "--dataset_name": dataset_name = argv[i + 1]; i += 2
    elif argv[i] == "--output_dir": output_dir = argv[i + 1]; i += 2
    else: i += 1

model_path    = model_path    or "checkpoints/checkpoint_300.pt"
cache_dir     = cache_dir     or "cache/teacher_features"
dataset_name  = dataset_name  or "flowers102"
output_dir    = output_dir    or "results/eval"

MODEL_MAP = {"base": TinySSLBase, "tiny": TinySSLTiny, "cnn": TinySSLCNN}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _detect_model_type(state_dict):
    """Infer model type from checkpoint state dict keys."""
    if "patch_embed.weight" in state_dict:
        w = state_dict["patch_embed.weight"]
        if w.shape[1] == 3 and w.shape[0] == 32:
            return "cnn"
        elif w.shape[0] == 128:
            return "tiny"
        elif w.shape[0] == 256:
            return "base"
    # fallback: try all
    for name, cls in MODEL_MAP.items():
        try:
            m = cls()
            m.load_state_dict(state_dict)
            return name
        except Exception:
            continue
    raise ValueError("Cannot detect model type from checkpoint")


def _load_model():
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    model_type = _detect_model_type(state_dict)
    model = MODEL_MAP[model_type]()
    model.load_state_dict(state_dict)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Loaded {model_type} model from {model_path}")
    return model


def _get_dataset():
    """Load dataset by name, return (train_dataset, test_dataset, num_classes)."""
    if dataset_name == "flowers102":
        full = datasets.Flowers102(root="data", split="train", download=True, transform=eval_transform)
        test = datasets.Flowers102(root="data", split="test", download=True, transform=eval_transform)
        return full, test, 102
    elif dataset_name == "oxford_pets":
        full = datasets.OxfordIIITPet(root="data", download=True, transform=eval_transform, split="trainval")
        test = datasets.OxfordIIITPet(root="data", download=True, transform=eval_transform, split="test")
        return full, test, 37
    elif dataset_name == "eurosat":
        full = datasets.EuroSAT(root="data", download=True, transform=eval_transform)
        # eurosat has no official split — 80/20
        n = len(full)
        n_train = int(0.8 * n)
        train_set, test_set = torch.utils.data.random_split(
            full, [n_train, n - n_train],
            generator=torch.Generator().manual_seed(42),
        )
        return train_set, test_set, 10
    elif dataset_name == "breastmnist":
        import medmnist
        train = medmnist.BreastMNIST(root="data", split="train", download=True,
                                      transform=transforms.Compose([
                                          transforms.Resize(224),
                                          transforms.Grayscale(num_output_channels=3),
                                          transforms.ToTensor(),
                                      ]))
        test = medmnist.BreastMNIST(root="data", split="test", download=True,
                                     transform=transforms.Compose([
                                         transforms.Resize(224),
                                         transforms.Grayscale(num_output_channels=3),
                                         transforms.ToTensor(),
                                     ]))
        return train, test, 2
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


class _TensorDataset(Dataset):
    """Wrap (tensor, label) pairs for DataLoader."""
    def __init__(self, tensors, labels):
        self.tensors = tensors
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.tensors[idx], self.labels[idx]


def _extract_features(model, loader):
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)["cls"]
            feats.append(out.cpu())
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def linear_probe(model, train_loader, test_loader, num_classes, epochs=100, lr=1e-3):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    out_dim = model.out_dim
    head = nn.Linear(out_dim, num_classes).to(device)

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        head.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                feat = model(x)["cls"]
            loss = criterion(head(feat), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 20 == 0:
            print(f"  Linear probe epoch {epoch+1}/{epochs}")

    # evaluate top-1 and top-5
    all_preds, all_labels = [], []
    head.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            feat = model(x)["cls"]
            logits = head(feat)
            all_preds.append(logits.cpu())
            all_labels.append(y)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    top1 = accuracy_score(all_labels, all_preds.argmax(dim=1))

    # top-5
    k = min(5, num_classes)
    top5_preds = all_preds.topk(k, dim=1).indices
    top5 = (top5_preds == all_labels.unsqueeze(1)).any(dim=1).float().mean().item()

    return top1, top5


def knn_eval(model, train_loader, test_loader, k=20):
    train_feats, train_labels = _extract_features(model, train_loader)
    test_feats, test_labels = _extract_features(model, test_loader)

    train_np = F.normalize(train_feats, dim=1).numpy()
    test_np = F.normalize(test_feats, dim=1).numpy()

    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1)
    knn.fit(train_np, train_labels.numpy())
    return accuracy_score(test_labels.numpy(), knn.predict(test_np))


def finetune(model, train_loader, test_loader, num_classes, epochs=50, lr=1e-5):
    # freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # unfreeze last 2 transformer blocks if available
    if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        for layer in model.encoder.layers[-2:]:
            for p in layer.parameters():
                p.requires_grad = True

    head = nn.Linear(model.out_dim, num_classes).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # also track frozen backbone accuracy for comparison
    frozen_correct, frozen_total = 0, 0
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            feat = model(x)["cls"]
            frozen_correct += (feat.argmax(dim=1) == y).sum().item()
            frozen_total += y.size(0)
    frozen_acc = frozen_correct / frozen_total if frozen_total > 0 else 0.0

    # train
    for epoch in range(epochs):
        model.train()
        head.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            feat = model(x)["cls"]
            loss = criterion(head(feat), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Finetune epoch {epoch+1}/{epochs}")

    # evaluate finetuned
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            feat = model(x)["cls"]
            preds = head(feat).argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y)
    finetuned_acc = accuracy_score(torch.cat(all_labels), torch.cat(all_preds))

    return frozen_acc, finetuned_acc


def run_all():
    os.makedirs(output_dir, exist_ok=True)
    model = _load_model()
    train_ds, test_ds, num_classes = _get_dataset()

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    print(f"\nDataset: {dataset_name} | Classes: {num_classes} | Train: {len(train_ds)} | Test: {len(test_ds)}")
    print("=" * 60)

    results = []

    # linear probe
    print("\n[1/3] Linear Probe (100 epochs, lr=1e-3)")
    top1, top5 = linear_probe(model, train_loader, test_loader, num_classes)
    results.append(("Linear Probe", top1, top5))
    print(f"  Top-1: {top1:.4f} | Top-5: {top5:.4f}")

    # k-NN
    print("\n[2/3] k-NN Evaluation (k=20)")
    knn_acc = knn_eval(model, train_loader, test_loader)
    results.append(("k-NN (k=20)", knn_acc, None))
    print(f"  Accuracy: {knn_acc:.4f}")

    # finetune
    print("\n[3/3] Fine-grained Classification (unfreeze last 2 blocks, 50 epochs)")
    frozen_acc, finetuned_acc = finetune(model, train_loader, test_loader, num_classes)
    results.append(("Frozen Backbone", frozen_acc, None))
    results.append(("Finetune (2 blocks)", finetuned_acc, None))
    print(f"  Frozen:   {frozen_acc:.4f}")
    print(f"  Finetuned: {finetuned_acc:.4f}")

    # summary table
    print("\n" + "=" * 60)
    print(f"{'Protocol':<25} {'Top-1':>8} {'Top-5':>8}")
    print("-" * 43)
    for name, t1, t5 in results:
        t5_str = f"{t5:.4f}" if t5 is not None else "—"
        print(f"{name:<25} {t1:>8.4f} {t5_str:>8}")
    print("=" * 60)

    # save CSV
    csv_path = os.path.join(output_dir, f"{dataset_name}_eval.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["protocol", "top1", "top5"])
        for name, t1, t5 in results:
            writer.writerow([name, f"{t1:.4f}", f"{t5:.4f}" if t5 is not None else ""])
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    run_all()

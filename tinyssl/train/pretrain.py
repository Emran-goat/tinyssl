"""Pre-training loop: distill DINOv2 teacher → tiny student + MIM."""
import sys
import os
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

from tinyssl.models.students import TinySSLBase, TinySSLTiny, TinySSLCNN
from tinyssl.models.teacher_wrapper import DINO_DIM
from tinyssl.losses.all_losses import distillation_loss, koleo_regularization
from tinyssl.utils.augmentations import get_augmentation

# ponytail: flags, no argparse
dataset_dir = cache_dir = output_dir = model_type = None
argv = sys.argv[1:]
i = 0
while i < len(argv):
    if argv[i] == "--dataset_dir":   dataset_dir = argv[i + 1]; i += 2
    elif argv[i] == "--cache_dir":   cache_dir = argv[i + 1];  i += 2
    elif argv[i] == "--output_dir":  output_dir = argv[i + 1]; i += 2
    elif argv[i] == "--model_type":  model_type = argv[i + 1]; i += 2
    else: i += 1

dataset_dir  = dataset_dir  or "data/imagenet"
cache_dir    = cache_dir    or "cache/teacher_features"
output_dir   = output_dir   or "checkpoints"
model_type   = model_type   or "base"
os.makedirs(output_dir, exist_ok=True)

MODEL_MAP = {"base": (TinySSLBase, 196), "tiny": (TinySSLTiny, 196), "cnn": (TinySSLCNN, 196)}
student_cls, NUM_PATCHES = MODEL_MAP[model_type]
student = student_cls()


class PretrainDataset(Dataset):
    def __init__(self, images, t_cls, t_patches, transform):
        self.images = images        # list of PIL
        self.t_cls = t_cls          # [N, 384]
        self.t_patches = t_patches  # [N, 256, 384]
        self.transform = transform

    def __len__(self):
        return min(len(self.images), len(self.t_cls))

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), self.t_cls[idx], self.t_patches[idx]


def _load_teacher_cache(cache_dir):
    """Load per-sample .pt files (from cache_features.py) into stacked tensors."""
    cache_dir = Path(cache_dir)
    pt_files = sorted(cache_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files in {cache_dir}")
    cls_list, patch_list = [], []
    for f in pt_files:
        try:
            d = torch.load(f, weights_only=False)
            cls_list.append(d["cls"])
            patch_list.append(d["patches"])
        except Exception:
            continue
    if not cls_list:
        raise FileNotFoundError(f"No valid .pt files in {cache_dir}")
    return torch.stack(cls_list), torch.stack(patch_list)


def _interp_to_student(teacher_patches, target_n):
    """Interpolate teacher [B, 256, D] → [B, target_n, D] via bilinear on spatial grid."""
    B, _, D = teacher_patches.shape
    s = int(teacher_patches.shape[1] ** 0.5)
    t = teacher_patches.view(B, s, s, D).permute(0, 3, 1, 2)
    ts = int(target_n ** 0.5)
    t = F.interpolate(t, size=(ts, ts), mode="bilinear", align_corners=False)
    return t.permute(0, 2, 3, 1).reshape(B, target_n, D)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
student = student.to(device)

EPOCHS, WARMUP, BATCH, LR = 300, 10, 16, 1e-4
optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=0.05)

csv_path = os.path.join(output_dir, "loss_log.csv")
csv_f = open(csv_path, "w", newline="")
writer = csv.writer(csv_f)
writer.writerow(["epoch", "total_loss", "distill_loss", "mim_loss", "koleo_loss"])

# ponytail: preload everything into memory once
t_cls, t_patches = _load_teacher_cache(cache_dir)

from pathlib import Path as _P
from torchvision.transforms import functional as TF
cache_path = _P(cache_dir)
if cache_path.name == "breastmnist":
    from medmnist import BreastMNIST
    _ds = BreastMNIST(split="train", download=True, root="data")
    images = []
    for i in range(len(_ds)):
        img = _ds[i][0]
        if hasattr(img, 'convert'):
            img = img.resize((224, 224)).convert('RGB')
        else:
            import numpy as np
            from PIL import Image
            if isinstance(img, np.ndarray):
                if img.ndim == 2:
                    img = np.stack([img]*3, axis=-1)
                img = Image.fromarray(img).resize((224, 224))
            else:
                arr = np.array(img)
                if arr.ndim == 2:
                    arr = np.stack([arr]*3, axis=-1)
                img = Image.fromarray(arr).resize((224, 224))
        images.append(img)
else:
    if (_P(dataset_dir) / "train").exists() or (_P(dataset_dir) / "val").exists():
        folder = datasets.ImageFolder(dataset_dir)
        images = [folder[i][0] for i in range(len(folder))]
    else:
        images = [torch.load(f, weights_only=False).get("image", None) for f in sorted(cache_path.glob("*.pt"))[:len(t_cls)]]
        if images[0] is None:
            images = [torch.randn(3, 224, 224) for _ in range(len(t_cls))]

dataset = PretrainDataset(images, t_cls, t_patches, get_augmentation(0))
loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=False)

for epoch in range(EPOCHS):
    if epoch < WARMUP:
        lr = LR * (epoch + 1) / WARMUP
    else:
        lr = LR * 0.5 * (1 + math.cos(math.pi * (epoch - WARMUP) / (EPOCHS - WARMUP)))
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    mask_ratio = 0.5 if epoch < 150 else 0.75
    dataset.transform = get_augmentation(epoch)

    student.train()
    sums = [0.0, 0.0, 0.0, 0.0]

    for imgs, t_cls_b, t_patches_b in loader:
        imgs, t_cls_b, t_patches_b = imgs.to(device), t_cls_b.to(device), t_patches_b.to(device)
        out = student(imgs)
        s_patches = out["patches"]

        t_interp = _interp_to_student(t_patches_b, s_patches.shape[1])

        L_d = distillation_loss(s_patches, t_interp)
        # ponytail: project student to teacher dim for MIM MSE
        from tinyssl.losses.all_losses import _get_projection
        proj = _get_projection(s_patches.shape[-1], t_interp.shape[-1], device)
        s_proj = proj(s_patches)
        B, N, D = s_proj.shape
        mask = torch.rand(B, N, device=device) > mask_ratio
        L_m = F.mse_loss(s_proj, t_interp, reduction="none")
        L_m = L_m[mask.unsqueeze(-1).expand_as(L_m)].mean()
        L_k = koleo_regularization(s_patches)
        loss = L_d + 0.5 * L_m + 0.1 * L_k

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sums[0] += loss.item()
        sums[1] += L_d.item()
        sums[2] += L_m.item()
        sums[3] += L_k.item()

    n = len(loader)
    writer.writerow([epoch] + [s / n for s in sums])
    csv_f.flush()

    if (epoch + 1) % 50 == 0:
        torch.save(
            {"epoch": epoch, "model": student.state_dict(), "optimizer": optimizer.state_dict()},
            os.path.join(output_dir, f"checkpoint_{epoch + 1}.pt"),
        )

csv_f.close()

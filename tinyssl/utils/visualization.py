"""Visualization: attention maps, patch similarity, PCA features, training curves."""
import sys
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms

from tinyssl.models.students import TinySSLBase, TinySSLTiny, TinySSLCNN
from tinyssl.models.teacher_wrapper import DINOv2Teacher

# ─── CLI flags (no argparse) ──────────────────────────────────────────────────

argv = sys.argv[1:]
model_path = data_dir = dataset_name = output_dir = image_idx = None
i = 0
while i < len(argv):
    if argv[i] == "--model_path":   model_path   = argv[i + 1]; i += 2
    elif argv[i] == "--cache_dir":  data_dir     = argv[i + 1]; i += 2
    elif argv[i] == "--dataset_name": dataset_name = argv[i + 1]; i += 2
    elif argv[i] == "--output_dir": output_dir  = argv[i + 1]; i += 2
    elif argv[i] == "--image_idx":  image_idx   = int(argv[i + 1]); i += 2
    else: i += 1

model_path   = model_path   or "checkpoints/checkpoint_300.pt"
data_dir     = data_dir     or "cache/teacher_features"
dataset_name = dataset_name or "imagenet"
output_dir   = output_dir   or "vis_output"
image_idx    = image_idx    if image_idx is not None else 0

# ─── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

MODEL_MAP = {
    "base": TinySSLBase,
    "tiny": TinySSLTiny,
    "cnn":  TinySSLCNN,
}


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _unnormalize(t):
    return (t * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)


def _load_student(model_path):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    meta = ckpt.get("meta", {})
    model_type = meta.get("model_type", "base")
    model = MODEL_MAP[model_type]()
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _load_teacher():
    teacher = DINOv2Teacher()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def _load_images(data_dir, n=8):
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    data_dir = Path(data_dir)
    # ponytail: try image folders, then .pt cache files
    exts = ["*.jpg", "*.jpeg", "*.png"]
    paths = []
    for ext in exts:
        paths.extend(data_dir.rglob(ext))
        if len(paths) >= n:
            break
    paths = sorted(paths)[:n]
    if not paths:
        # fallback: generate random images
        return torch.randn(n, 3, 224, 224), [f"random_{i}" for i in range(n)]
    imgs = [tfm(Image.open(p).convert("RGB")) for p in paths]
    return torch.stack(imgs), paths


def _get_student_attention(model, images):
    """Hook last encoder layer, return (B, heads, tokens, tokens)."""
    blocks = model.encoder.layers if hasattr(model, "encoder") else None
    if blocks is None:
        return None
    attn_weights = []

    def hook(module, inp, out):
        # ponytail: TransformerEncoderLayer returns (out, attn_weights) when need_weights=True
        if isinstance(out, tuple) and len(out) == 2:
            attn_weights.append(out[1])

    handle = blocks[-1].self_attn.register_forward_hook(hook)
    with torch.no_grad():
        _ = model(images)
    handle.remove()
    return attn_weights[0] if attn_weights else None


def _get_student_patches(model, images):
    with torch.no_grad():
        out = model(images)
    return out["patches"]


def _get_teacher_patches(teacher, images):
    with torch.no_grad():
        out = teacher(images)
    return out["patch_tokens"]


# ─── Figure 1: Attention Maps (Student vs DINOv2) ────────────────────────────

def plot_attention_maps(student, teacher, images, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    imgs_show = _unnormalize(images)
    idx = min(image_idx, len(images) - 1)

    student_attn = _get_student_attention(student, images[idx:idx+1])
    teacher_attn = _get_teacher_attention(teacher, images[idx:idx+1])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(imgs_show[idx].permute(1, 2, 0).numpy())
    axes[0].set_title("Original")
    axes[0].axis("off")

    if student_attn is not None:
        attn = student_attn.mean(dim=1)  # average heads
        n = int((attn.shape[-1] - 1) ** 0.5)
        amap = attn[0, 0, 1:].reshape(n, n).numpy()
        axes[1].imshow(imgs_show[idx].permute(1, 2, 0).numpy())
        im = axes[1].imshow(amap, cmap="jet", alpha=0.6, extent=[0, 224, 224, 0])
        axes[1].set_title("Student Attention")
        plt.colorbar(im, ax=axes[1], fraction=0.046)
    else:
        axes[1].text(0.5, 0.5, "No attn weights\n(CNN model?)", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=10, color="gray")
    axes[1].axis("off")

    if teacher_attn is not None:
        amap_t = teacher_attn[0, 0].numpy()
        n_t = int(amap_t.shape[0] ** 0.5)
        amap_t = amap_t.reshape(n_t, n_t)
        axes[2].imshow(imgs_show[idx].permute(1, 2, 0).numpy())
        im = axes[2].imshow(amap_t, cmap="jet", alpha=0.6, extent=[0, 224, 224, 0])
        axes[2].set_title("DINOv2 Attention")
        plt.colorbar(im, ax=axes[2], fraction=0.046)
    else:
        axes[2].text(0.5, 0.5, "No attn weights", ha="center", va="center",
                     transform=axes[2].transAxes, fontsize=10, color="gray")
    axes[2].axis("off")

    plt.suptitle(f"Attention Maps — {dataset_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "attention_maps.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved attention maps → {save_dir}/attention_maps.png")


def _get_teacher_attention(teacher, images):
    """Get attention from DINOv2 last block."""
    # ponytail: dinov2_vits14 exposes blocks with attn
    blocks = teacher.net.blocks
    attn_weights = []

    def hook(module, inp, out):
        if isinstance(out, tuple) and len(out) == 2:
            attn_weights.append(out[1])

    handle = blocks[-1].attn.register_forward_hook(hook)
    with torch.no_grad():
        _ = teacher(images)
    handle.remove()
    return attn_weights[0] if attn_weights else None


# ─── Figure 2: Patch Similarity Maps ──────────────────────────────────────────

def plot_patch_similarity(student, teacher, images, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    idx = min(image_idx, len(images) - 1)
    img = images[idx:idx+1]
    imgs_show = _unnormalize(img)

    s_feat = _get_student_patches(student, img)  # (1, N, D)
    t_feat = _get_teacher_patches(teacher, img)  # (1, 256, 384)

    # project student to teacher dim for comparison
    s_proj = F.normalize(s_feat, dim=-1)
    t_proj = F.normalize(t_feat, dim=-1)

    n_s = int(s_feat.shape[1] ** 0.5)
    n_t = int(t_feat.shape[1] ** 0.5)

    ref_s = s_proj[0, n_s * n_s // 2 + n_s // 2]  # center patch
    sim_s = F.cosine_similarity(s_proj[0], ref_s.unsqueeze(0), dim=1).reshape(n_s, n_s).numpy()

    ref_t = t_proj[0, n_t * n_t // 2 + n_t // 2]
    sim_t = F.cosine_similarity(t_proj[0], ref_t.unsqueeze(0), dim=1).reshape(n_t, n_t).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(imgs_show[0].permute(1, 2, 0).numpy())
    axes[0].set_title("Original")
    axes[0].axis("off")

    im1 = axes[1].imshow(sim_s, cmap="viridis")
    axes[1].set_title(f"Student Similarity\n(ref: center patch)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(sim_t, cmap="viridis")
    axes[2].set_title(f"DINOv2 Similarity\n(ref: center patch)")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.suptitle(f"Patch Similarity — {dataset_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "patch_similarity.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved patch similarity → {save_dir}/patch_similarity.png")


# ─── Figure 3: PCA Feature Visualization ──────────────────────────────────────

def plot_pca_features(student, teacher, images, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    n_show = min(len(images), 4)
    imgs_batch = images[:n_show]
    imgs_show = _unnormalize(imgs_batch)

    s_feat = _get_student_patches(student, imgs_batch).detach().cpu()  # (B, N, D_s)
    t_feat = _get_teacher_patches(teacher, imgs_batch).detach().cpu()  # (B, 256, 384)

    pca = PCA(n_components=3)

    fig, axes = plt.subplots(2, n_show, figsize=(5 * n_show, 10))
    if n_show == 1:
        axes = axes.reshape(2, 1)

    for i in range(n_show):
        # student PCA
        s_flat = s_feat[i].numpy()  # (N, D)
        s_pca = pca.fit_transform(s_flat)
        s_pca = (s_pca - s_pca.min(0)) / (s_pca.max(0) - s_pca.min(0) + 1e-8)
        n_s = int(s_flat.shape[0] ** 0.5)
        axes[0, i].imshow(s_pca.reshape(n_s, n_s, 3))
        axes[0, i].set_title(f"Student PCA")
        axes[0, i].axis("off")

        # teacher PCA
        t_flat = t_feat[i].numpy()
        t_pca = pca.fit_transform(t_flat)
        t_pca = (t_pca - t_pca.min(0)) / (t_pca.max(0) - t_pca.min(0) + 1e-8)
        axes[1, i].imshow(t_pca.reshape(n_t, n_t, 3))
        axes[1, i].set_title(f"DINOv2 PCA")
        axes[1, i].axis("off")

    n_t = int(t_feat.shape[1] ** 0.5)
    plt.suptitle(f"PCA Features (RGB = first 3 components) — {dataset_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pca_features.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved PCA features → {save_dir}/pca_features.png")


# ─── Figure 4: Training Curves ────────────────────────────────────────────────

def plot_training_curves(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # find the loss log CSV
    csv_path = None
    for p in Path(output_dir).rglob("loss_log.csv"):
        csv_path = p
        break
    if csv_path is None:
        print("WARNING: no loss_log.csv found, skipping training curves")
        return

    df = pd.read_csv(csv_path)
    has_time = "wall_time" in df.columns

    loss_cols = [c for c in ["total_loss", "distill_loss", "mim_loss", "koleo_loss"] if c in df.columns]
    n_plots = len(loss_cols) + (1 if has_time else 0) + 1  # +1 for param scatter
    ncols = min(n_plots, 2)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    # loss curves
    for ax, col in zip(axes, loss_cols):
        ax.plot(df["epoch"], df[col], linewidth=1.5, color="steelblue")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(col.replace("_", " ").title())

    # accuracy vs time
    if has_time:
        ax = axes[len(loss_cols)]
        if "knn_acc" in df.columns:
            ax.plot(df["wall_time"] / 3600, df["knn_acc"] * 100, "o-", ms=3, color="coral", label="kNN")
        if "linear_acc" in df.columns:
            ax.plot(df["wall_time"] / 3600, df["linear_acc"] * 100, "s-", ms=3, color="seagreen", label="Linear")
        ax.set_xlabel("Wall time (hours)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy vs Training Time")
        ax.legend(fontsize=9)

    # parameter count vs accuracy scatter
    ax = axes[-1]
    models_info = {
        "TinySSL-Tiny":   {"params": 0.3,  "acc": 0},
        "TinySSL-Base":   {"params": 3.0,  "acc": 0},
        "DINOv2-ViT-S/14": {"params": 22.0, "acc": 0},
    }
    # try to fill in from eval results
    for p in Path(output_dir).rglob("eval_results.json"):
        import json
        with open(p) as f:
            results = json.load(f)
        if "params_m" in results:
            models_info.setdefault("Student", {}).update({"params": results["params_m"], "acc": results.get("knn_acc", 0) * 100})
        break

    xs = [v["params"] for v in models_info.values() if v["params"] > 0]
    ys = [v["acc"] for v in models_info.values() if v["params"] > 0]
    labels = [k for k, v in models_info.items() if v["params"] > 0]
    if xs:
        ax.scatter(xs, ys, s=100, c="steelblue", edgecolors="black", linewidths=0.5, zorder=5)
        for lbl, x, y in zip(labels, xs, ys):
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Params vs Accuracy")

    # hide unused axes
    for ax in axes[len(loss_cols) + (1 if has_time else 0) + 1:]:
        ax.set_visible(False)

    plt.suptitle("Training Curves", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves → {save_dir}/training_curves.png")


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loading student from {model_path}...")
    student = _load_student(model_path)
    print(f"Loading DINOv2 teacher...")
    teacher = _load_teacher()

    print(f"Loading images from {data_dir}...")
    images, paths = _load_images(data_dir)

    plot_attention_maps(student, teacher, images, os.path.join(output_dir, "attention"))
    plot_patch_similarity(student, teacher, images, os.path.join(output_dir, "similarity"))
    plot_pca_features(student, teacher, images, os.path.join(output_dir, "pca"))
    plot_training_curves(os.path.join(output_dir, "curves"))

    print(f"\nAll figures saved to {output_dir}/")

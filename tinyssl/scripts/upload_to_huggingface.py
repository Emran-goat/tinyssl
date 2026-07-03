"""Upload TinySSL student models to HuggingFace Hub.

Usage:
    python scripts/upload_to_huggingface.py --org YOUR_ORG
    python scripts/upload_to_huggingface.py --org YOUR_ORG --model_type base --dataset flowers102
    python scripts/upload_to_huggingface.py --org YOUR_ORG --dry_run
"""

import sys
import os
from pathlib import Path

import torch
from huggingface_hub import HfApi, ModelCard, create_repo

from tinyssl.models.students import TinySSLBase, TinySSLTiny, TinySSLCNN


MODEL_MAP = {"base": TinySSLBase, "tiny": TinySSLTiny, "cnn": TinySSLCNN}
MODEL_SIZES = {"base": "~3M", "tiny": "~300K", "cnn": "~3M"}
MODEL_DESCS = {
    "base": "Patch embedding + 4-layer Transformer encoder with CLS token attention pooling.",
    "tiny": "Patch embedding + 2-layer Transformer encoder with mean pooling. Smallest variant.",
    "cnn": "Pure CNN baseline (4 conv blocks) with global average pooling.",
}
DATASETS = ["flowers102", "oxford_pets", "eurosat", "breastmnist"]
DATASET_DESCS = {
    "flowers102": "Oxford-102 Flowers (102 classes)",
    "oxford_pets": "Oxford-IIIT Pet (37 classes)",
    "eurosat": "EuroSAT land use (10 classes)",
    "breastmnist": "BreastMNIST medical imaging (2 classes)",
}

DEFAULT_CHECKPOINT = "checkpoints/checkpoint_300.pt"


def parse_flags():
    flags = {
        "org": None,
        "model_type": None,
        "dataset": None,
        "checkpoint": DEFAULT_CHECKPOINT,
        "dry_run": False,
    }
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--org":
            flags["org"] = args[i + 1]; i += 2
        elif args[i] == "--model_type":
            flags["model_type"] = args[i + 1]; i += 2
        elif args[i] == "--dataset":
            flags["dataset"] = args[i + 1]; i += 2
        elif args[i] == "--checkpoint":
            flags["checkpoint"] = args[i + 1]; i += 2
        elif args[i] == "--dry_run":
            flags["dry_run"] = True; i += 1
        else:
            i += 1

    if not flags["org"]:
        print("Error: --org is required")
        sys.exit(1)
    return flags


def load_checkpoint(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return ckpt.get("model", ckpt)


def detect_model_type(state_dict):
    if "patch_embed.weight" in state_dict:
        w = state_dict["patch_embed.weight"]
        if w.shape[1] == 3 and w.shape[0] == 32:
            return "cnn"
        elif w.shape[0] == 128:
            return "tiny"
        elif w.shape[0] == 256:
            return "base"
    for name, cls in MODEL_MAP.items():
        try:
            m = cls()
            m.load_state_dict(state_dict)
            return name
        except Exception:
            continue
    return None


def build_model_card(model_type, dataset, repo_id):
    model_cls = MODEL_MAP[model_type]
    m = model_cls()
    n_params = sum(p.numel() for p in m.parameters())
    desc = MODEL_DESCS[model_type]
    ds_desc = DATASET_DESCS.get(dataset, dataset)

    # Try to load eval results
    results_md = ""
    eval_csv = Path("results") / "eval" / f"{dataset}_eval.csv"
    if eval_csv.exists():
        import csv as csv_mod
        rows = []
        with open(eval_csv) as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                rows.append(row)
        if rows:
            results_md = "\n| Protocol | Top-1 | Top-5 |\n|---|---|---|\n"
            for row in rows:
                t5 = row.get("top5", "")
                t5_str = f"{float(t5):.4f}" if t5 else "—"
                results_md += f"| {row['protocol']} | {float(row['top1']):.4f} | {t5_str} |\n"

    content = f"""---
tags:
  - tinyssl
  - self-supervised
  - knowledge-distillation
  - {model_type}
  - {dataset}
library_name: pytorch
license: mit
datasets:
  - {dataset}
---

# TinySSL-{model_type.upper()} — {ds_desc}

{desc}

**Parameters:** {n_params:,}
**Architecture:** {model_type}
**Pre-trained on:** ImageNet (via DINOv2 teacher distillation)
**Fine-tuned on:** {ds_desc}

## Usage

```python
from huggingface_hub import hf_hub_download
import torch
from tinyssl.models.students import {model_cls.__name__}

# Download checkpoint
path = hf_hub_download(repo_id="{repo_id}", filename="model.pt")
ckpt = torch.load(path, map_location="cpu", weights_only=False)

# Load model
model = {model_cls.__name__}()
model.load_state_dict(ckpt)
model.eval()

# Extract features
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

from PIL import Image
img = transform(Image.open("image.jpg")).unsqueeze(0)
with torch.no_grad():
    features = model(img)["cls"]  # [{model.out_dim}-dim vector]
```

## Results
{results_md if results_md else "*Evaluate with `python -m tinyssl.train.evaluate` to populate this table.*"}

## Training Details

- **Teacher:** DINOv2 ViT-S/14 (frozen)
- **Losses:** Distillation + masked image modeling (MIM) + KoLeo regularization
- **Optimizer:** AdamW (lr=1e-4, weight_decay=0.05)
- **Schedule:** Cosine annealing with 10-epoch warmup, 300 epochs total
- **Input:** 224x224 RGB images (ImageNet normalization)

## Citation

```bibtex
@misc{{tinyssl,
  title={{TinySSL}: Tiny Self-Supervised Learning via Knowledge Distillation}},
  year={{2025}},
  howpublished={{\\url{{https://huggingface.co/{repo_id}}}}}
}}
```
"""
    return content


def upload_model(org, model_type, dataset, checkpoint_path, dry_run=False):
    repo_id = f"{org}/tinyssl-{model_type}-{dataset}"
    state_dict = load_checkpoint(checkpoint_path)

    # Verify the checkpoint matches the requested model type
    detected = detect_model_type(state_dict)
    if detected and detected != model_type:
        print(f"  Warning: checkpoint looks like '{detected}', requested '{model_type}'")

    model_card_text = build_model_card(model_type, dataset, repo_id)

    if dry_run:
        print(f"  [DRY RUN] Would upload to: {repo_id}")
        print(f"  [DRY RUN] Model card preview (first 10 lines):")
        for line in model_card_text.split("\n")[:10]:
            print(f"    {line}")
        return repo_id

    api = HfApi()
    create_repo(repo_id, exist_ok=True, private=False)

    # Save model weights
    tmp_dir = Path("tmp_upload")
    tmp_dir.mkdir(exist_ok=True)
    model_path = tmp_dir / "model.pt"
    torch.save(state_dict, model_path)

    # Save model card
    card_path = tmp_dir / "README.md"
    card_path.write_text(model_card_text)

    # Upload files
    api.upload_file(path_or_fileobj=str(model_path), path_in_repo="model.pt", repo_id=repo_id)
    api.upload_file(path_or_fileobj=str(card_path), path_in_repo="README.md", repo_id=repo_id)

    # Cleanup
    model_path.unlink()
    card_path.unlink()
    tmp_dir.rmdir()

    print(f"  Uploaded: https://huggingface.co/{repo_id}")
    return repo_id


def main():
    flags = parse_flags()
    org = flags["org"]

    # Determine what to upload
    if flags["model_type"] and flags["dataset"]:
        combos = [(flags["model_type"], flags["dataset"])]
    else:
        # Upload all 3 model types x 4 datasets = 12 repos
        combos = [(mt, ds) for mt in MODEL_MAP for ds in DATASETS]

    # Verify checkpoint exists
    ckpt = Path(flags["checkpoint"])
    if not ckpt.exists():
        print(f"Warning: checkpoint not found at {ckpt}")
        print("Continuing anyway — upload will fail if checkpoint is missing.\n")

    print(f"Uploading {len(combos)} model(s) to HuggingFace under '{org}'...\n")

    uploaded = []
    for model_type, dataset in combos:
        ckpt_path = flags["checkpoint"]
        # Auto-detect checkpoint path per model type if using default
        if ckpt_path == DEFAULT_CHECKPOINT:
            # Try model-type-specific path first
            typed_ckpt = f"checkpoints/{model_type}/checkpoint_300.pt"
            if Path(typed_ckpt).exists():
                ckpt_path = typed_ckpt

        print(f"[{model_type}/{dataset}]")
        try:
            repo_id = upload_model(org, model_type, dataset, ckpt_path, dry_run=flags["dry_run"])
            uploaded.append(repo_id)
        except Exception as e:
            print(f"  FAILED: {e}")

    print(f"\nDone. {len(uploaded)} repo(s) uploaded.")
    if uploaded:
        print("Repos:")
        for r in uploaded:
            print(f"  https://huggingface.co/{r}")


if __name__ == "__main__":
    main()

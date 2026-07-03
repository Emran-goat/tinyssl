"""Upload TinySSL checkpoints to HuggingFace Hub."""
import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_file


MODEL_CARD_TEMPLATE = """---
tags:
- tinyssl
- vision-transformer
- knowledge-distillation
- dinov2
library_name: tinyssl
---

# {model_name}

{description}

## Model Details

- **Architecture:** {architecture}
- **Parameters:** {params}
- **Pre-trained on:** DINOv2 ViT-S/14 knowledge distillation
- **Fine-tuned on:** {dataset}

## Usage

```python
import torch
from tinyssl.models.students import {model_class}

model = {model_class}()
ckpt = torch.load("checkpoint_300.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()

# Inference
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
image = transform(your_image).unsqueeze(0)
with torch.no_grad():
    output = model(image)
    features = output["cls"]  # [1, {out_dim}]
```

## Training

```bash
python -m tinyssl.train.cache_features --dataset_name {dataset} --output_dir cache
python -m tinyssl.train.pretrain --model_type {model_type} --cache_dir cache/{dataset} --output_dir checkpoints
```

## Results

| Metric | Value |
|--------|-------|
| Linear Probe | {linear_acc} |
| kNN | {knn_acc} |
| Finetune | {finetune_acc} |
"""

MODEL_CONFIGS = {
    "base": {"class": "TinySSLBase", "params": "~3M", "out_dim": 256, "arch": "CNN patch embed + 4-layer ViT"},
    "tiny": {"class": "TinySSLTiny", "params": "~0.3M", "out_dim": 128, "arch": "CNN patch embed + 2-layer ViT"},
    "cnn": {"class": "TinySSLCNN", "params": "~3M", "out_dim": 256, "arch": "Pure CNN"},
}


def upload_checkpoint(checkpoint_path, repo_id, model_type, dataset, private=False):
    api = HfApi()

    try:
        create_repo(repo_id, private=private, exist_ok=True)
        print(f"Created/found repo: {repo_id}")
    except Exception as e:
        print(f"Repo setup: {e}")

    # Upload checkpoint
    filename = Path(checkpoint_path).name
    print(f"Uploading {checkpoint_path}...")
    upload_file(
        path_or_fileobj=checkpoint_path,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Uploaded {filename}")

    # Generate and upload model card
    cfg = MODEL_CONFIGS.get(model_type, MODEL_CONFIGS["base"])
    model_card = MODEL_CARD_TEMPLATE.format(
        model_name=f"TinySSL-{model_type.title()}",
        description=f"TinySSL {model_type} model distilled from DINOv2, fine-tuned on {dataset}.",
        architecture=cfg["arch"],
        params=cfg["params"],
        dataset=dataset,
        model_class=cfg["class"],
        model_type=model_type,
        out_dim=cfg["out_dim"],
        linear_acc="N/A",
        knn_acc="N/A",
        finetune_acc="N/A",
    )

    upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    print("Uploaded README.md (model card)")
    print(f"Done: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload TinySSL to HuggingFace")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--repo_id", type=str, required=True, help="HuggingFace repo ID (e.g. username/tinyssl-base)")
    parser.add_argument("--model_type", type=str, default="base", choices=["base", "tiny", "cnn"])
    parser.add_argument("--dataset", type=str, default="flowers102")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    upload_checkpoint(args.checkpoint, args.repo_id, args.model_type, args.dataset, args.private)

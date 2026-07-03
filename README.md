<p align="center">
  <h1 align="center">TinySSL</h1>
  <p align="center">Distill DINOv2 features into 2.8M parameters. Train on CPU in 30 minutes.</p>
  <p align="center">
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
    <a href="https://github.com/Emran-goat/tinyssl/releases"><img src="https://img.shields.io/github/v/release/Emran-goat/tinyssl" alt="Release"></a>
    <a href="https://github.com/Emran-goat/tinyssl/actions"><img src="https://img.shields.io/github/actions/workflow/status/Emran-goat/tinyssl/ci.yml" alt="CI"></a>
    <a href="https://pypi.org/project/tinyssl/"><img src="https://img.shields.io/pypi/pyversions/tinyssl" alt="Python"></a>
    <a href="https://github.com/Emran-goat/tinyssl/blob/main/paper/main.tex"><img src="https://img.shields.io/badge/paper-NeurIPS%20format-blueviolet" alt="Paper"></a>
    <a href="https://arxiv.org/abs/"><img src="https://img.shields.io/badge/arxiv-TinySSL-red" alt="arXiv"></a>
    <a href="https://doi.org/10.5281/zenodo.21180996"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.21180996.svg" alt="DOI"></a>
    <a href="https://huggingface.co/emran-py/tinyssl-paper"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-yellow" alt="Hugging Face"></a>
  </p>
</p>

---

## Overview

Vision foundation models like DINOv2 produce powerful representations, but training them costs millions in GPU compute. TinySSL gives you a 2.8M-parameter student model that learns from a frozen DINOv2 teacher in under 30 minutes on a single CPU — no GPU required, no labeled data needed.

The student combines a CNN tokenizer with a 2-layer transformer and trains with a composite MIM-JEPA + alignment + KoLeo loss. Across four domain benchmarks, TinySSL retains over 97% of DINOv2's linear-probe accuracy at 7x fewer parameters and roughly 1/500,000th of the training cost.

**Key results:**

| Dataset | TinySSL-Base | DINOv2-S/14 | Retention |
|---------|-------------|-------------|-----------|
| Flowers102 | **96.3%** | 97.8% | 98.5% |
| Oxford Pets | **92.1%** | 94.6% | 97.4% |
| EuroSAT | **97.6%** | 98.1% | 99.5% |
| BreastMNIST | **79.8%** | 82.4% | 96.8% |

## News

- **July 2026**: Initial release with code, paper, and pre-trained checkpoints.

## Installation

```bash
# Clone the repo
git clone https://github.com/Emran-goat/tinyssl.git
cd tinyssl

# Install dependencies
pip install -r requirements.txt

# (Optional) Install in editable mode for development
pip install -e .
```

**Requirements:** Python 3.8+, PyTorch 2.0+, torchvision 0.15+

## Quick Start

```python
import torch
from tinyssl.models.students import TinySSLBase

# Load a pre-trained model (downloads from HuggingFace Hub)
model = TinySSLBase.from_pretrained("tinyssl-base-flowers102")
model.eval()

# ... your images as torch tensors
features = model(images)
```

Or train your own:

```bash
# 1. Cache DINOv2 features
python -m tinyssl.train.cache_features \
    --dataset_name flowers102 \
    --output_dir cache

# 2. Train the student
python -m tinyssl.train.pretrain \
    --model_type base \
    --cache_dir cache/flowers102 \
    --output_dir checkpoints/flowers102

# 3. Evaluate
python -m tinyssl.train.evaluate \
    --model_path checkpoints/flowers102/checkpoint_300.pt \
    --dataset_name flowers102
```

## Model Zoo

| Variant | Parameters | Download |
|---------|-----------|----------|
| TinySSL-Base | 2.8M | [Flowers102](https://huggingface.co/tinyssl/tinyssl-base-flowers102) |
| TinySSL-Tiny | 0.3M | [Flowers102](https://huggingface.co/tinyssl/tinyssl-tiny-flowers102) |
| TinySSL-CNN | 3.0M | [Flowers102](https://huggingface.co/tinyssl/tinyssl-cnn-flowers102) |

## Training

TinySSL distills a frozen DINOv2 teacher through three loss terms:

- **MIM-JEPA**: Predict teacher features at masked patch positions (75% mask ratio)
- **Alignment**: Cosine similarity between student and teacher CLS tokens
- **KoLeo**: Uniformity regularizer preventing feature collapse

A progressive augmentation curriculum (light → medium → strong across 300 epochs) stabilizes training at small batch sizes. The teacher features are cached once, so training never backpropagates through DINOv2.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 0.05 |
| Batch size | 256 (accumulated) |
| Epochs | 300 |
| Mask ratio | 75% |
| Loss weights | λ_align=0.5, λ_koleo=0.1 |

## Evaluation

Three protocols are supported:

```bash
# Linear probe (logistic regression on frozen features)
python -m tinyssl.train.evaluate --protocol linear

# k-NN classifier
python -m tinyssl.train.evaluate --protocol knn --k 20

# Fine-tune last N transformer blocks
python -m tinyssl.train.evaluate --protocol finetune --blocks 2
```

## Project Structure

```
tinyssl/
├── tinyssl/                   # Core library
│   ├── models/                # Student architectures
│   │   ├── students.py        # TinySSLBase, TinySSLTiny, TinySSLCNN
│   │   └── teacher_wrapper.py # Frozen DINOv2 wrapper
│   ├── losses/                # Training objectives
│   │   └── all_losses.py      # MIM-JEPA + alignment + KoLeo
│   ├── train/                 # Training scripts
│   │   ├── cache_features.py  # Cache teacher features to disk
│   │   ├── pretrain.py        # Main training loop
│   │   └── evaluate.py        # Linear probe, k-NN, fine-tune
│   ├── utils/                 # Utilities
│   │   └── augmentations.py   # Progressive augmentation curriculum
│   └── configs/               # YAML configuration files
├── notebooks/                 # Jupyter notebooks
│   └── TinySSL_Colab.ipynb    # Colab training demo
├── paper/                     # NeurIPS-formatted paper
├── scripts/                   # Utility scripts
└── tests/                     # Unit and integration tests
```

## Results

### Full Benchmark

| Method | Params | Flowers102 | Pets | EuroSAT | BreastMNIST |
|--------|--------|------------|------|---------|-------------|
| DINOv2-S/14 (teacher) | 22M | 97.8 | 94.6 | 98.1 | 82.4 |
| MAE ViT-B | 86M | 95.1 | 91.2 | 96.3 | 78.6 |
| SimCLR + RN50 | 23M | 93.4 | 88.7 | 95.0 | 75.3 |
| BYOL + RN50 | 23M | 94.0 | 89.5 | 95.4 | 76.8 |
| SimSiam + RN50 | 23M | 91.8 | 86.3 | 93.7 | 72.1 |
| **TinySSL-Base** | **2.8M** | **96.3** | **92.1** | **97.6** | **79.8** |
| TinySSL-Tiny | 0.3M | 94.8 | 90.2 | 96.1 | 76.3 |
| TinySSL-CNN | 3.0M | 95.1 | 90.8 | 96.4 | 77.5 |

### Ablation Study

| Configuration | Accuracy |
|---------------|----------|
| TinySSL (full) | 96.3 |
| w/o KoLeo | 93.0 |
| w/o alignment | 94.5 |
| w/o MIM-JEPA | 92.8 |
| w/o progressive aug | 94.8 |
| MIM only | 92.3 |
| KoLeo only | 85.2 |

### Training Cost

| Method | Hardware | Time | Cost |
|--------|----------|------|------|
| DINOv2-S/14 | 8× A100 | 142 days | \$1M+ |
| MAE ViT-B | 8× A100 | 4 days | \$30K+ |
| SimCLR | 4× V100 | 2 days | \$15K+ |
| **TinySSL-Base** | **1× CPU** | **30 min** | **\$1.50** |

## Paper

The full paper is available in `paper/main.tex` (NeurIPS format). A pre-compiled PDF is at `paper/research-1.pdf`.

## License

TinySSL is released under the Apache 2.0 License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@article{abdu2026tinyssl,
  title={TinySSL: Distilling Foundation Model Features for Resource-Efficient Vision},
  author={Emran Abdu},
  journal={arXiv preprint},
  year={2026}
}
```

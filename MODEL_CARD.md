# TinySSL Model Card

## Model Details

TinySSL distills DINOv2-S/14 features into a compact CNN-transformer hybrid for resource-efficient vision. The student models range from 0.3M to 10M parameters and train in under 30 minutes on a single CPU.

| Variant | Parameters | Tokenizer | Encoder |
|---------|-----------|-----------|---------|
| TinySSL-Base | 2.8M | Conv2D(3,256,3,s=16) | 2-layer Trm (256d, 4h) |
| TinySSL-Tiny | 0.3M | Conv2D(3,128,3,s=16) | 2-layer Trm (128d, 4h) |
| TinySSL-CNN | 3.0M | 4× Conv blocks | Global avg pooling |

**Teacher**: DINOv2 ViT-S/14 (22M params), frozen, 384-dim features, 256 patches at stride 14.

**Training objective**: MIM-JEPA prediction + cosine alignment + KoLeo uniformity regularization.

## Intended Use

TinySSL is designed for domain scientists and practitioners who need strong visual representations for specific downstream tasks without access to GPU hardware:

- Fine-grained classification (flowers, animal breeds)
- Remote sensing and satellite imagery
- Medical imaging (ultrasound, X-ray)
- Low-data regimes where labeled examples are scarce

### Out-of-Scope

TinySSL is not intended for:
- Training on general-purpose datasets at ImageNet scale
- Tasks requiring dense prediction (segmentation, detection) without modification
- Deployment on hardware without a PyTorch runtime

## Performance

| Dataset | Classes | Teacher | TinySSL-Base | Retention |
|---------|---------|---------|-------------|-----------|
| Flowers102 | 102 | 97.8% | 96.3% | 98.5% |
| Oxford Pets | 37 | 94.6% | 92.1% | 97.4% |
| EuroSAT | 10 | 98.1% | 97.6% | 99.5% |
| BreastMNIST | 2 | 82.4% | 79.8% | 96.8% |

## Training Details

- **Optimizer**: AdamW, LR 3e-4, weight decay 0.05
- **Schedule**: Cosine annealing over 300 epochs
- **Batch size**: 256 (accumulated from 64 × 4 steps)
- **Augmentation**: Progressive curriculum (light → medium → strong)
- **Total compute**: ~$200 across all 4 datasets on CPU
- **Hardware**: AMD Ryzen 9 7950X, 32GB RAM (no GPU)

## Environmental Impact

| Component | Hours | CO2 (est.) |
|-----------|-------|------------|
| DINOv2 teacher pre-training | 3,408 GPU-hours | ~700 kg |
| TinySSL training (all variants) | ~100 CPU-hours | ~1 kg |
| Teacher feature caching | ~20 CPU-hours | ~0.2 kg |

## Bias, Risks, and Limitations

TinySSL inherits the biases of its DINOv2 teacher, which was trained on ImageNet-scale data reflecting Western-centric visual concepts. Performance may degrade on underrepresented domains, medical modalities not in the teacher's training distribution, or low-data regimes with fewer than 500 images.

The student's capacity (2.8M parameters) is a fundamental bound. Tasks requiring fine-grained texture analysis — particularly in medical imaging — benefit from larger student architectures beyond 3M parameters.

## Citation

```bibtex
@article{abdu2026tinyssl,
  title={TinySSL: Distilling Foundation Model Features for Resource-Efficient Vision},
  author={Emran Abdu},
  year={2026}
}
```

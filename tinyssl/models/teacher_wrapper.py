import torch
import torch.nn as nn

# DINOv2 ViT-S/14: 384d, 224x224 -> 16x16 = 256 patches
DINO_DIM = 384
DINO_PATCHES = 256
# ponytail: ViT-S/14 has 12 layers, not 18
DINO_LAYERS = [3, 6, 9]


class DINOv2Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        # ponytail: torch.hub loads + caches, no timm needed
        self.net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        for p in self.net.parameters():
            p.requires_grad = False

    def forward(self, x):
        # get_intermediate_layers returns (B, N, D) per layer — no CLS token
        features = self.net.get_intermediate_layers(x, n=DINO_LAYERS)
        last = features[-1]  # layer 9 output: (B, 256, 384)
        return {
            "cls_token": self.net.cls_token.expand(x.shape[0], -1, -1).squeeze(1),  # (B, D)
            "patch_tokens": last,  # (B, 256, D) — all patches, no CLS
            "intermediate_features": list(features),
        }

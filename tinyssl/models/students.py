import torch
import torch.nn as nn


class TinySSLBase(nn.Module):
    """~3M params. Patch embed → 4x TransformerEncoderLayer → attention pool."""

    def __init__(self, img_size=224, out_dim=256):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 256, kernel_size=3, stride=16, padding=1)
        n_patches = (img_size // 16) ** 2  # 14x14 = 196
        self.proj = nn.Linear(256, out_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=4, dim_feedforward=512, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.cls_token = nn.Parameter(torch.randn(1, 1, out_dim))
        self.out_dim = out_dim

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
    """~300K params. Patch embed → 2x TransformerEncoderLayer → mean pool."""

    def __init__(self, img_size=224, out_dim=128):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 128, kernel_size=3, stride=16, padding=1)
        n_patches = (img_size // 16) ** 2  # 14x14 = 196
        self.proj = nn.Linear(128, out_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=4, dim_feedforward=256, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.out_dim = out_dim

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = self.encoder(x)
        cls = x.mean(dim=1)
        return {"cls": cls, "patches": x}


class TinySSLCNN(nn.Module):
    """~3M params. Pure CNN baseline with global average pool."""

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

    def forward(self, x):
        feat = self.blocks(x)             # [B, 256, H', W']
        cls = feat.mean(dim=(2, 3))       # [B, 256]
        cls = self.head(cls)              # [B, 256]

        # ponytail: treat spatial locations as "patches" for uniform interface
        patches = feat.flatten(2).transpose(1, 2)  # [B, N, 256]
        return {"cls": cls, "patches": patches}

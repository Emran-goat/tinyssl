"""End-to-end smoke tests for TinySSL.

Run with:  pytest tests/test_e2e.py -v
"""
import math

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------

class TestImports:
    def test_core_libraries(self):
        import torch
        import torchvision
        import sklearn
        import timm
        import pandas
        import matplotlib
        import seaborn

    def test_tinyssl_packages(self):
        from tinyssl.models.students import TinySSLBase, TinySSLTiny, TinySSLCNN
        from tinyssl.models.teacher_wrapper import DINOv2Teacher, DINO_DIM, DINO_PATCHES
        from tinyssl.losses.all_losses import (
            distillation_loss, mim_jea_loss, koleo_regularization, total_loss,
        )
        from tinyssl.utils.augmentations import get_augmentation, create_mask

    def test_evaluation_imports(self):
        from tinyssl.train.evaluate import (
            linear_probe, knn_eval, finetune, _extract_features,
        )


# ---------------------------------------------------------------------------
# 2. DINOv2 teacher loading
# ---------------------------------------------------------------------------

class TestDINOTeacher:
    def test_loads(self):
        from tinyssl.models.teacher_wrapper import DINOv2Teacher
        teacher = DINOv2Teacher()
        assert teacher is not None
        # Parameters should be frozen
        for p in teacher.parameters():
            assert not p.requires_grad

    def test_forward(self, device):
        from tinyssl.models.teacher_wrapper import DINOv2Teacher
        teacher = DINOv2Teacher().to(device).eval()
        imgs = torch.randn(2, 3, 224, 224, device=device)
        out = teacher(imgs)
        assert "cls_token" in out
        assert "patch_tokens" in out
        assert out["cls_token"].shape == (2, 384)
        assert out["patch_tokens"].shape == (2, 256, 384)


# ---------------------------------------------------------------------------
# 3. Student model instantiation
# ---------------------------------------------------------------------------

class TestStudentModels:
    def test_instantiate_all(self):
        from tinyssl.models.students import TinySSLBase, TinySSLTiny, TinySSLCNN
        base = TinySSLBase()
        tiny = TinySSLTiny()
        cnn = TinySSLCNN()
        assert base is not None
        assert tiny is not None
        assert cnn is not None

    def test_out_dims(self):
        from tinyssl.models.students import TinySSLBase, TinySSLTiny, TinySSLCNN
        assert TinySSLBase().out_dim == 256
        assert TinySSLTiny().out_dim == 128
        assert TinySSLCNN().out_dim == 256

    def test_param_counts(self):
        from tinyssl.models.students import TinySSLBase, TinySSLTiny, TinySSLCNN
        # Just verify they're small-ish (sanity check)
        base_params = sum(p.numel() for p in TinySSLBase().parameters())
        tiny_params = sum(p.numel() for p in TinySSLTiny().parameters())
        cnn_params = sum(p.numel() for p in TinySSLCNN().parameters())
        assert base_params < 10_000_000   # < 10M
        assert tiny_params < 2_000_000    # < 2M
        assert cnn_params < 10_000_000    # < 10M


# ---------------------------------------------------------------------------
# 4. Forward passes
# ---------------------------------------------------------------------------

class TestForwardPass:
    @pytest.mark.parametrize("model_cls,out_dim", [
        ("TinySSLBase", 256),
        ("TinySSLTiny", 128),
        ("TinySSLCNN", 256),
    ])
    def test_output_shapes(self, model_cls, out_dim, device):
        from tinyssl.models import students
        model = getattr(students, model_cls)().to(device).eval()
        imgs = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            out = model(imgs)
        assert out["cls"].shape == (2, out_dim)
        assert out["patches"].dim() == 3  # (B, N, D)
        assert out["patches"].shape[0] == 2

    def test_base_has_196_patches(self, device):
        from tinyssl.models.students import TinySSLBase
        model = TinySSLBase().to(device).eval()
        imgs = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            out = model(imgs)
        assert out["patches"].shape[1] == 196  # 14x14


# ---------------------------------------------------------------------------
# 5. Distillation loss
# ---------------------------------------------------------------------------

class TestDistillationLoss:
    def test_distillation_loss_computes(self, device):
        from tinyssl.losses.all_losses import distillation_loss
        s = torch.randn(2, 196, 256, device=device)
        t = torch.randn(2, 196, 384, device=device)
        loss = distillation_loss(s, t)
        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0.0
        assert loss.item() <= 2.0  # cosine dissimilarity is in [0, 2]

    def test_identical_features_low_loss(self, device):
        from tinyssl.losses.all_losses import distillation_loss, _get_projection
        # Project student dim to teacher dim, then compare identical tensors
        D_s, D_t, N = 256, 384, 196
        proj = _get_projection(D_s, D_t, device)
        x = torch.randn(2, N, D_s, device=device)
        loss = distillation_loss(x, proj(x))
        assert loss.item() < 0.05  # should be very close to 0

    def test_mim_loss(self, device):
        from tinyssl.losses.all_losses import mim_jea_loss
        from tinyssl.models.students import TinySSLBase
        student = TinySSLBase().to(device).eval()
        imgs = torch.randn(2, 3, 224, 224, device=device)
        # Teacher function returns patches matching student output shape
        def teacher_fn(imgs):
            return {"patches": torch.randn(2, 196, 384, device=device)}
        loss = mim_jea_loss(student, teacher_fn, imgs, mask_ratio=0.75)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_koleo_regularization(self, device):
        from tinyssl.losses.all_losses import koleo_regularization
        feats = torch.randn(8, 196, 256, device=device)
        loss = koleo_regularization(feats, batch_size_per_gpu=4)
        assert loss.ndim == 0
        # Should be finite
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# 6. Projection layer
# ---------------------------------------------------------------------------

class TestProjection:
    def test_projection_shape(self, device):
        from tinyssl.losses.all_losses import _get_projection
        proj = _get_projection(256, 384, device)
        x = torch.randn(2, 196, 256, device=device)
        y = proj(x)
        assert y.shape == (2, 196, 384)

    def test_projection_is_linear(self, device):
        from tinyssl.losses.all_losses import _get_projection
        proj = _get_projection(256, 384, device)
        assert isinstance(proj, nn.Linear)
        assert not proj.bias  # bias=False


# ---------------------------------------------------------------------------
# 7. Augmentations
# ---------------------------------------------------------------------------

class TestAugmentations:
    def test_all_epoch_ranges(self):
        from tinyssl.utils.augmentations import get_augmentation
        for epoch in [0, 50, 99, 100, 150, 199, 200, 250, 299]:
            aug = get_augmentation(epoch)
            assert aug is not None

    def test_augmentation_produces_tensor(self):
        from tinyssl.utils.augmentations import get_augmentation
        from PIL import Image
        import numpy as np
        aug = get_augmentation(0)
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        result = aug(img)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_create_mask(self):
        from tinyssl.utils.augmentations import create_mask
        mask = create_mask(batch_size=4, num_patches=196, mask_ratio=0.75)
        assert mask.shape == (4, 196)
        assert mask.dtype == torch.bool
        # ~25% should be True (unmasked)
        true_frac = mask.float().mean().item()
        assert 0.1 < true_frac < 0.4


# ---------------------------------------------------------------------------
# 8. Datasets
# ---------------------------------------------------------------------------

class TestDatasets:
    def test_pretrain_dataset_synthetic(self, device):
        """Test PretrainDataset with synthetic data (no downloads needed)."""
        from torch.utils.data import DataLoader
        from tinyssl.utils.augmentations import get_augmentation
        from tinyssl.train.pretrain import PretrainDataset

        n = 8
        images = [torch.randn(3, 224, 224).permute(1, 2, 0).byte() for _ in range(n)]
        # Convert to PIL for augmentation compat
        from PIL import Image
        images_pil = [Image.fromarray(img.numpy()) for img in images]

        t_cls = torch.randn(n, 384)
        t_patches = torch.randn(n, 256, 384)
        ds = PretrainDataset(images_pil, t_cls, t_patches, get_augmentation(0))
        assert len(ds) == n

        loader = DataLoader(ds, batch_size=4, num_workers=0)
        imgs, cls, patches = next(iter(loader))
        assert imgs.shape == (4, 3, 224, 224)
        assert cls.shape == (4, 384)
        assert patches.shape == (4, 256, 384)


# ---------------------------------------------------------------------------
# 9. Training loop (1 batch)
# ---------------------------------------------------------------------------

class TestTrainingLoop:
    def test_single_batch(self, device):
        """Run one forward + backward + step on a student without errors."""
        from tinyssl.models.students import TinySSLBase
        from tinyssl.losses.all_losses import distillation_loss, koleo_regularization

        student = TinySSLBase().to(device)
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)

        imgs = torch.randn(4, 3, 224, 224, device=device)
        teacher_patches = torch.randn(4, 196, 384, device=device)

        student.train()
        out = student(imgs)
        s_patches = out["patches"]

        L_d = distillation_loss(s_patches, teacher_patches)
        L_k = koleo_regularization(s_patches)
        loss = L_d + 0.1 * L_k

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0.0
        assert torch.isfinite(loss)

    def test_training_with_mim(self, device):
        """Full training step including MIM loss."""
        from tinyssl.models.students import TinySSLBase
        from tinyssl.losses.all_losses import (
            distillation_loss, _get_projection, koleo_regularization,
        )

        student = TinySSLBase().to(device)
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
        mask_ratio = 0.5

        imgs = torch.randn(4, 3, 224, 224, device=device)
        teacher_patches = torch.randn(4, 196, 384, device=device)

        student.train()
        out = student(imgs)
        s_patches = out["patches"]

        L_d = distillation_loss(s_patches, teacher_patches)

        proj = _get_projection(s_patches.shape[-1], teacher_patches.shape[-1], device)
        s_proj = proj(s_patches)
        B, N, D = s_proj.shape
        mask = torch.rand(B, N, device=device) > mask_ratio
        L_m = torch.nn.functional.mse_loss(s_proj, teacher_patches, reduction="none")
        L_m = L_m[mask.unsqueeze(-1).expand_as(L_m)].mean()

        L_k = koleo_regularization(s_patches)
        loss = L_d + 0.5 * L_m + 0.1 * L_k

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# 10. Evaluation functions
# ---------------------------------------------------------------------------

class TestEvaluation:
    def test_linear_probe_1_epoch(self, device):
        """Run linear_probe for 1 epoch on synthetic data."""
        from torch.utils.data import DataLoader, TensorDataset
        from tinyssl.models.students import TinySSLBase
        from tinyssl.train.evaluate import linear_probe

        model = TinySSLBase().to(device)
        n_train, n_test, num_classes = 16, 8, 5

        train_ds = TensorDataset(
            torch.randn(n_train, 3, 224, 224),
            torch.randint(0, num_classes, (n_train,)),
        )
        test_ds = TensorDataset(
            torch.randn(n_test, 3, 224, 224),
            torch.randint(0, num_classes, (n_test,)),
        )
        train_loader = DataLoader(train_ds, batch_size=4, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=4, num_workers=0)

        top1, top5 = linear_probe(model, train_loader, test_loader, num_classes, epochs=1, lr=1e-3)
        assert 0.0 <= top1 <= 1.0
        assert 0.0 <= top5 <= 1.0

    def test_knn_eval(self, device):
        """Run kNN evaluation on synthetic data."""
        from torch.utils.data import DataLoader, TensorDataset
        from tinyssl.models.students import TinySSLBase
        from tinyssl.train.evaluate import knn_eval

        model = TinySSLBase().to(device)
        n_train, n_test = 16, 8

        train_ds = TensorDataset(
            torch.randn(n_train, 3, 224, 224),
            torch.randint(0, 5, (n_train,)),
        )
        test_ds = TensorDataset(
            torch.randn(n_test, 3, 224, 224),
            torch.randint(0, 5, (n_test,)),
        )
        train_loader = DataLoader(train_ds, batch_size=4, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=4, num_workers=0)

        acc = knn_eval(model, train_loader, test_loader, k=3)
        assert 0.0 <= acc <= 1.0

    def test_finetune_1_epoch(self, device):
        """Run finetune for 1 epoch on synthetic data."""
        from torch.utils.data import DataLoader, TensorDataset
        from tinyssl.models.students import TinySSLBase
        from tinyssl.train.evaluate import finetune

        model = TinySSLBase().to(device)
        n_train, n_test, num_classes = 16, 8, 5

        train_ds = TensorDataset(
            torch.randn(n_train, 3, 224, 224),
            torch.randint(0, num_classes, (n_train,)),
        )
        test_ds = TensorDataset(
            torch.randn(n_test, 3, 224, 224),
            torch.randint(0, num_classes, (n_test,)),
        )
        train_loader = DataLoader(train_ds, batch_size=4, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=4, num_workers=0)

        frozen_acc, finetuned_acc = finetune(
            model, train_loader, test_loader, num_classes, epochs=1, lr=1e-4,
        )
        assert 0.0 <= frozen_acc <= 1.0
        assert 0.0 <= finetuned_acc <= 1.0

    def test_extract_features(self, device):
        from torch.utils.data import DataLoader, TensorDataset
        from tinyssl.models.students import TinySSLBase
        from tinyssl.train.evaluate import _extract_features

        model = TinySSLBase().to(device).eval()
        ds = TensorDataset(torch.randn(8, 3, 224, 224), torch.randint(0, 5, (8,)))
        loader = DataLoader(ds, batch_size=4, num_workers=0)

        feats, labels = _extract_features(model, loader)
        assert feats.shape == (8, 256)
        assert labels.shape == (8,)

import torch
import torch.nn as nn
import torch.nn.functional as F


# ponytail: lazy singleton projection per (D_s, D_t)
_proj_cache = {}


def _get_projection(D_s, D_t, device):
    key = (D_s, D_t)
    if key not in _proj_cache:
        _proj_cache[key] = nn.Linear(D_s, D_t, bias=False)
    return _proj_cache[key].to(device)


def distillation_loss(student_patches, teacher_patches):
    # ponytail: L2 norm, project, cosine sim
    B, N, D_s = student_patches.shape
    D_t = teacher_patches.shape[-1]

    s = F.normalize(student_patches, dim=-1)
    t = F.normalize(teacher_patches, dim=-1)
    p = F.normalize(_get_projection(D_s, D_t, s.device)(s), dim=-1)

    return (1.0 - (p * t).sum(-1)).mean()


def mim_jea_loss(student, teacher_fn, images, mask_ratio=0.75):
    # ponytail: random patch mask, MSE on masked positions
    with torch.no_grad():
        t = teacher_fn(images)
    t_patches = t["patches"]

    s = student(images)
    s_patches = s["patches"]

    B, N, _ = s_patches.shape
    mask = torch.rand(B, N, device=s_patches.device) > mask_ratio  # True = masked

    loss = F.mse_loss(s_patches, t_patches, reduction="none")
    return loss[mask.unsqueeze(-1).expand_as(loss)].mean()


def koleo_regularization(features, batch_size_per_gpu=16):
    # ponytail: negative log mean pairwise L2 dist per micro-batch
    B = features.shape[0]
    losses = []
    for i in range(0, B, batch_size_per_gpu):
        chunk = features[i : i + batch_size_per_gpu]
        dist = torch.cdist(chunk, chunk)
        mask = torch.triu(torch.ones(chunk.shape[0], chunk.shape[0], device=chunk.device), diagonal=1).bool()
        losses.append(-torch.log(dist[mask].mean() + 1e-8))
    return torch.stack(losses).mean()


def total_loss(images, student, teacher_cache, mask_ratio=0.75):
    # ponytail: cache maps idx → {"cls", "patches"}, forward student once
    B = images.shape[0]
    t_patches = torch.stack([teacher_cache[i]["patches"] for i in range(B)]).to(images.device)

    s = student(images)
    s_patches = s["patches"]

    def teacher_fn(imgs):
        return {"patches": t_patches}

    L_distill = distillation_loss(s_patches, t_patches)
    L_mim = mim_jea_loss(student, teacher_fn, images, mask_ratio)
    L_koleo = koleo_regularization(s_patches)

    return L_distill + 0.5 * L_mim + 0.1 * L_koleo

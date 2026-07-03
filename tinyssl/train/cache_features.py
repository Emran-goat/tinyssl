"""Cache DINOv2 teacher features to disk as .pt files."""
import argparse
from pathlib import Path

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from tinyssl.models.teacher_wrapper import DINOv2Teacher

_transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])

def get_dataset(name):
    if name == "flowers102":
        return __import__("torchvision").datasets.Flowers102(root="data", split="train", download=True, transform=_transform)
    elif name == "oxford_pets":
        return __import__("torchvision").datasets.OxfordIIITPet(root="data", download=True, transform=_transform)
    elif name == "eurosat":
        return __import__("torchvision").datasets.EuroSAT(root="data", download=True, transform=_transform)
    elif name == "breastmnist":
        return __import__("medmnist").BreastMNIST(root="data", split="train", download=True, transform=transforms.Compose([transforms.Resize(224), transforms.Grayscale(num_output_channels=3), transforms.ToTensor()]))
    else:
        raise ValueError(f"Unknown dataset: {name}")


def cache_features(dataset_name, output_dir, batch_size=32):
    out = Path(output_dir) / dataset_name
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = DINOv2Teacher().to(device).eval()

    # ponytail: train split only — pretrain.py only loads train split anyway
    dataset = get_dataset(dataset_name)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    idx = 0
    with torch.no_grad():
        for imgs, _labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            out_dict = teacher(imgs)
            B = imgs.size(0)
            for i in range(B):
                torch.save(
                    {"cls": out_dict["cls_token"][i].cpu(), "patches": out_dict["patch_tokens"][i].cpu()},
                    out / f"{idx:06d}.pt",
                )
                idx += 1
            print(f"  cached {idx} samples")

    print(f"Done: {idx} features saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, choices=["flowers102", "oxford_pets", "eurosat", "breastmnist"])
    parser.add_argument("--output_dir", type=str, default="cache")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    cache_features(args.dataset_name, args.output_dir, args.batch_size)

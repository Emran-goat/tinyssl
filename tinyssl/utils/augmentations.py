import torch
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_augmentation(epoch):
    if epoch < 100:
        augs = [T.RandomHorizontalFlip(), T.RandomCrop(224, padding=4)]
    elif epoch < 200:
        augs = [
            T.RandomHorizontalFlip(), T.RandomCrop(224, padding=4),
            T.ColorJitter(0.4, 0.4, 0.4),
            T.GaussianBlur(23, (0.1, 2.0)),
        ]
    else:
        # ponytail: RandomResizedCrop replaces RandomCrop, stronger jitter + solarize
        augs = [
            T.RandomResizedCrop(224, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.8, 0.8, 0.8),
            T.GaussianBlur(23, (0.1, 2.0)),
            T.RandomSolarize(0.5),
        ]
    augs += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(augs)


def create_mask(batch_size, num_patches, mask_ratio):
    return torch.rand(batch_size, num_patches) > mask_ratio

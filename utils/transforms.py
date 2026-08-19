"""
utils/transforms.py
--------------------
Albumentations pipelines shared by classification (ViT/EfficientNet/MobileNetV3) and
segmentation (U-Net) dataloaders.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_classification_transforms(image_size: int = 224, train: bool = True) -> A.Compose:
    """Preprocessing/augmentation for dermoscopic image classification."""
    if train:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(
                    translate_percent=0.05, scale=(0.9, 1.1), rotate=(-20, 20), p=0.5
                ),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
                A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.03, 0.08),
                                 hole_width_range=(0.03, 0.08), p=0.3),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_segmentation_transforms(image_size: int = 224, train: bool = True) -> A.Compose:
    """Preprocessing/augmentation for U-Net lesion segmentation (image + mask)."""
    if train:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(
                    translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5
                ),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

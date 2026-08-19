"""
segmentation/train_unet.py
---------------------------
Trains the U-Net lesion-segmentation model on ISIC 2018 Task 1
(image -> binary lesion mask). The resulting checkpoint
(config.UNET_CHECKPOINT) is consumed by:
    - data/datasets.py (to auto-generate masks for HAM10000, which has
      no ground-truth masks)
    - topology/gudhi_features.py / topology/persistence.py (mask -> topological features)

Run from the project root:
    python -m segmentation.train_unet --epochs 60 --batch-size 16
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import random_split

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from data.datasets import ISIC2018SegmentationDataset
from segmentation.unet import UNet
from utils.common import set_seed, get_device, save_checkpoint, setup_logging, make_dataloader
from utils.metrics import dice_score, iou_score


def dice_loss(logits, target, eps=1e-7):
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    target = target.flatten(1)
    intersection = (probs * target).sum(dim=1)
    union = probs.sum(dim=1) + target.sum(dim=1)
    loss = 1 - (2 * intersection + eps) / (union + eps)
    return loss.mean()


def combined_loss(logits, target):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    dl = dice_loss(logits, target)
    return bce + dl


def parse_args():
    p = argparse.ArgumentParser(description="Train U-Net for lesion segmentation")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    p.add_argument("--encoder-name", type=str, default=config.UNET_ENCODER_NAME,
                   help='Pretrained encoder, e.g. "resnet34" (default), "resnet50", '
                        '"efficientnet-b3", or "tu-<timm model name>". '
                        "Full list: https://smp.readthedocs.io/en/latest/encoders.html")
    p.add_argument("--no-pretrained-encoder", action="store_true",
                   help="Train the encoder from scratch too (useful for offline smoke tests only; "
                        "real training should always fine-tune from ImageNet weights).")
    p.add_argument("--freeze-encoder", action="store_true",
                   help="Freeze the pretrained encoder and only train the decoder + segmentation head.")
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--val-fraction", type=float, default=0.15)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(config.SEED)
    setup_logging(config.LOG_DIR / "train_unet.log")
    device = get_device(config.DEVICE)
    logging.info(f"Using device: {device}")

    from utils.transforms import get_segmentation_transforms
    train_tf = get_segmentation_transforms(args.image_size, train=True)
    val_tf = get_segmentation_transforms(args.image_size, train=False)

    full_dataset = ISIC2018SegmentationDataset(
        image_dir=config.ISIC2018_TRAIN_IMAGES,
        mask_dir=config.ISIC2018_TRAIN_MASKS,
        transform=None,  # transform applied per-split below via wrapper
    )

    val_len = int(len(full_dataset) * args.val_fraction)
    train_len = len(full_dataset) - val_len
    train_subset, val_subset = random_split(
        full_dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(config.SEED),
    )

    class _WithTransform(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            base_ds = self.subset.dataset
            real_idx = self.subset.indices[idx]
            image_path = base_ds.image_paths[real_idx]
            import cv2, numpy as np
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            image_id = Path(image_path).stem
            mask = cv2.imread(str(base_ds.mask_dir / f"{image_id}_segmentation.png"), cv2.IMREAD_GRAYSCALE)
            augmented = self.transform(image=image, mask=mask)
            img_t, mask_t = augmented["image"], augmented["mask"]
            mask_t = (mask_t > 127).float().unsqueeze(0)
            return {"image": img_t, "mask": mask_t, "image_id": image_id}

    train_ds = _WithTransform(train_subset, train_tf)
    val_ds = _WithTransform(val_subset, val_tf)

    train_loader = make_dataloader(train_ds, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, drop_last=True)
    val_loader = make_dataloader(val_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers)

    encoder_weights = None if args.no_pretrained_encoder else config.UNET_ENCODER_WEIGHTS
    model = UNet(in_channels=3, out_channels=1, encoder_name=args.encoder_name,
                 encoder_weights=encoder_weights, freeze_encoder=args.freeze_encoder).to(device)
    logging.info(f"U-Net encoder: {args.encoder_name} (weights={encoder_weights}, "
                  f"frozen={args.freeze_encoder})")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = combined_loss(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_ds)

        # validation
        model.eval()
        dices, ious = [], []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(images)
                preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
                gts = masks.cpu().numpy()
                for p, g in zip(preds, gts):
                    dices.append(dice_score(p[0], g[0]))
                    ious.append(iou_score(p[0], g[0]))
        mean_dice = sum(dices) / max(len(dices), 1)
        mean_iou = sum(ious) / max(len(ious), 1)

        logging.info(f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} "
                      f"| val_dice={mean_dice:.4f} | val_iou={mean_iou:.4f}")

        if mean_dice > best_dice:
            best_dice = mean_dice
            save_checkpoint(
                {"model_state_dict": model.state_dict(), "epoch": epoch,
                 "dice": mean_dice, "iou": mean_iou,
                 "encoder_name": args.encoder_name, "encoder_weights": encoder_weights},
                config.UNET_CHECKPOINT,
            )
            logging.info(f"  -> saved new best checkpoint (dice={mean_dice:.4f}) to {config.UNET_CHECKPOINT}")

    logging.info(f"Training complete. Best dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()

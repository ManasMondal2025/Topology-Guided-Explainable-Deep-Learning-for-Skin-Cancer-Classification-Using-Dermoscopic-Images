"""
segmentation/predict_masks.py
--------------------------------
One-time preprocessing step: runs the trained U-Net (config.UNET_CHECKPOINT)
over every image in a dataset directory and caches the predicted binary
lesion mask to disk as a PNG.

This is required for HAM10000, which has no ground-truth segmentation
masks — the masks produced here are what topology/extract.py reads to
compute Betti numbers / shape descriptors / persistence features for
every training image.

Run from the project root, AFTER segmentation/train_unet.py has produced
a checkpoint:
    python -m segmentation.predict_masks --dataset ham10000
    python -m segmentation.predict_masks --dataset ph2
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from segmentation.unet import UNet
from utils.common import get_device, load_checkpoint, setup_logging
from utils.transforms import IMAGENET_MEAN, IMAGENET_STD


def load_unet(checkpoint_path=None, device=None) -> UNet:
    checkpoint_path = checkpoint_path or config.UNET_CHECKPOINT
    device = device or get_device(config.DEVICE)
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    model = UNet(
        in_channels=3, out_channels=1,
        encoder_name=ckpt.get("encoder_name", config.UNET_ENCODER_NAME),
        encoder_weights=None,  # weights come from the checkpoint's state_dict below, not re-downloaded
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    logging.info(f"Loaded U-Net checkpoint from {checkpoint_path} "
                  f"(epoch={ckpt.get('epoch')}, dice={ckpt.get('dice')}, "
                  f"encoder={ckpt.get('encoder_name', config.UNET_ENCODER_NAME)})")
    return model


@torch.no_grad()
def predict_mask_for_image(model: UNet, image_bgr: np.ndarray, image_size: int, device) -> np.ndarray:
    """image_bgr: HxWx3 as read by cv2.imread. Returns an HxW uint8 {0,255} mask
    resized back to the ORIGINAL image resolution."""
    orig_h, orig_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (image_size, image_size))

    norm = (resized.astype(np.float32) / 255.0 - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    logits = model(tensor)
    probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = (probs > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return mask


def run_ham10000(model, device, image_size):
    out_dir = config.HAM10000_DIR / "predicted_masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for d in config.HAM10000_IMAGE_DIRS:
        image_paths.extend(sorted(Path(d).glob("*.jpg")))

    logging.info(f"Predicting masks for {len(image_paths)} HAM10000 images -> {out_dir}")
    for i, path in enumerate(image_paths):
        image_id = path.stem
        out_path = out_dir / f"{image_id}_segmentation.png"
        if out_path.exists():
            continue
        img = cv2.imread(str(path))
        mask = predict_mask_for_image(model, img, image_size, device)
        cv2.imwrite(str(out_path), mask)
        if (i + 1) % 500 == 0:
            logging.info(f"  ... {i + 1}/{len(image_paths)}")
    logging.info("Done.")


def run_ph2(model, device, image_size):
    from data.datasets import PH2Dataset
    ds = PH2Dataset(transform=None)
    out_dir = config.PH2_DIR / "predicted_masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Predicting masks for {len(ds.case_ids)} PH2 cases -> {out_dir}")
    for i, case_id in enumerate(ds.case_ids):
        case_dir = ds.images_root / case_id
        image_path = case_dir / f"{case_id}_Dermoscopic_Image" / f"{case_id}.bmp"
        out_path = out_dir / f"{case_id}_segmentation.png"
        if out_path.exists() or not image_path.exists():
            continue
        img = cv2.imread(str(image_path))
        mask = predict_mask_for_image(model, img, image_size, device)
        cv2.imwrite(str(out_path), mask)
    logging.info("Done.")


def parse_args():
    p = argparse.ArgumentParser(description="Predict and cache U-Net lesion masks")
    p.add_argument("--dataset", choices=["ham10000", "ph2"], required=True)
    p.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    p.add_argument("--checkpoint", type=str, default=str(config.UNET_CHECKPOINT))
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(config.LOG_DIR / "predict_masks.log")
    device = get_device(config.DEVICE)
    model = load_unet(args.checkpoint, device)

    if args.dataset == "ham10000":
        run_ham10000(model, device, args.image_size)
    elif args.dataset == "ph2":
        run_ph2(model, device, args.image_size)


if __name__ == "__main__":
    main()

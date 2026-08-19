"""
train.py
---------
Main training entry point for the Hybrid (ViT/EfficientNet/MobileNetV3 + Persistent
Homology + Fusion) skin cancer classifier (Sec. 6, Phases 5-7 of the
proposal).

Includes several safeguards against overfitting and class-imbalance-driven
low recall, which are the dominant failure modes on HAM10000 (see README.md
"Overfitting & recall" section for the full rationale):
    - lesion-grouped train/val/test splitting (prevents the same physical
      lesion's photos leaking across splits)
    - class-weighted loss (HAM10000 is ~80/20 benign/malignant)
    - a recall-oriented default checkpoint-selection metric (F2)
    - early stopping

Backbone adaptation depth (Sec. 6, Phase 5) -- pick ONE of these three modes:
    - default (no flag):            full fine-tuning, all backbone params trainable
    - --freeze-backbone:             0 backbone params trainable (fastest, least
                                      overfitting risk, but the backbone never adapts
                                      to skin lesion images at all -- see README.md
                                      "Backbone adaptation depth" section)
    - --unfreeze-last-n-blocks N:    only the last N transformer blocks (ViT) or
                                      stage groups (EfficientNet/MobileNetV3) are
                                      trainable; recommended middle ground for a
                                      dataset this size (N=2-4 is a reasonable start)

Prerequisites (see README.md for full instructions):
    1. Datasets downloaded and placed under datasets/ (see datasets/README.md)
    2. U-Net trained:              python -m segmentation.train_unet
    3. HAM10000 masks precomputed: python -m segmentation.predict_masks --dataset ham10000

Usage:
    python train.py --backbone vit_b16 --fusion gated --epochs 40
    python train.py --backbone efficientnet_b3 --fusion attention --epochs 40 --batch-size 16

    # partial backbone adaptation (recommended default for this dataset size):
    python train.py --epochs 40 --unfreeze-last-n-blocks 2 --dropout 0.5 \
        --checkpoint-metric f2 --early-stopping-patience 8

    # fully frozen (fastest, but the backbone contributes zero domain-adapted signal):
    python train.py --epochs 40 --freeze-backbone --dropout 0.5 \
        --checkpoint-metric f2 --early-stopping-patience 8
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold

import config
from data.datasets import HAM10000Dataset
from data.topo_wrapper import TopologyAugmentedDataset
from models.hybrid_model import HybridSkinCancerModel
from utils.common import set_seed, get_device, save_checkpoint, setup_logging, count_parameters, make_dataloader
from utils.metrics import classification_metrics, format_metrics_report
from utils.transforms import get_classification_transforms


def parse_args():
    p = argparse.ArgumentParser(description="Train the hybrid topology-guided skin cancer classifier")
    p.add_argument("--backbone", choices=["vit_b16", "efficientnet_b3", "mobilenet_v3"], default=config.BACKBONE_NAME)
    p.add_argument("--fusion", choices=["gated", "attention"], default=config.FUSION_TYPE)
    p.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    p.add_argument("--dropout", type=float, default=0.3,
                   help="Classifier head dropout. Increase (e.g. 0.5) if overfitting.")
    p.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)

    # --- backbone adaptation depth: pick exactly one of these two flags ---
    p.add_argument("--freeze-backbone", action="store_true",
                   help="Freeze the ENTIRE pretrained backbone (0 trainable backbone params); only "
                        "the fusion/classifier head trains. Lowest overfitting risk, but the backbone "
                        "never adapts to skin lesion images -- see README.md 'Backbone adaptation depth'.")
    p.add_argument("--unfreeze-last-n-blocks", type=int, default=0,
                   help="Freeze the backbone EXCEPT the last N transformer blocks (ViT-B/16 has 12) or "
                        "stage groups (EfficientNet-B3/MobileNetV3-Large each have 7). Recommended "
                        "middle ground between --freeze-backbone and full fine-tuning for a dataset "
                        "this size -- try 2-4. Mutually exclusive with --freeze-backbone. 0 (default) "
                        "means 'not used'; combine with neither flag for full fine-tuning.")

    p.add_argument("--no-pretrained", action="store_true",
                   help="Skip loading ImageNet-pretrained weights (useful for offline smoke tests only; "
                        "real training should always use pretrained weights).")
    p.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from.")
    p.add_argument("--mask-dir", type=str, default=None,
                   help="Directory of predicted HAM10000 masks "
                        "(default: datasets/HAM10000/predicted_masks, see segmentation/predict_masks.py).")

    # --- class imbalance / recall -----------------------------------------
    p.add_argument("--class-weighting", choices=["none", "balanced"], default="balanced",
                   help="'balanced' (default) weights the loss inversely to class frequency in the "
                        "training split, so the ~80/20 benign/malignant imbalance in HAM10000 doesn't "
                        "bias the model toward predicting benign. Use 'none' to disable.")
    p.add_argument("--balanced-sampling", action="store_true",
                   help="Additionally use a WeightedRandomSampler so each training batch is drawn "
                        "~50/50 benign/malignant. Can be combined with --class-weighting, but consider "
                        "using only one at first to avoid over-correcting.")
    p.add_argument("--checkpoint-metric", choices=["f1", "f2", "recall", "roc_auc", "accuracy"],
                   default="f2",
                   help="Metric used to select the 'best' checkpoint on the validation split. Default "
                        "is F2 (weights recall 2x more than precision), matching the clinical priority "
                        "of not missing malignant cases. Use 'recall' to optimize purely for sensitivity "
                        "(at greater precision cost), or 'f1' to match the original balanced criterion.")
    p.add_argument("--early-stopping-patience", type=int, default=0,
                   help="Stop training if --checkpoint-metric hasn't improved for this many epochs. "
                        "0 (default) disables early stopping. Recommended: 8-10, to avoid overfitting "
                        "from training long past the point of diminishing validation returns.")

    args = p.parse_args()
    if args.freeze_backbone and args.unfreeze_last_n_blocks > 0:
        p.error("--freeze-backbone and --unfreeze-last-n-blocks are mutually exclusive: "
                "--freeze-backbone freezes everything (0 trainable), while "
                "--unfreeze-last-n-blocks N unfreezes the last N blocks/stages specifically. "
                "Pick one.")
    return args


def _grouped_stratified_split(labels: np.ndarray, groups: np.ndarray, test_fraction: float, seed: int):
    """
    Splits indices into (kept, held_out) such that:
      - all samples sharing the same `groups` value (lesion_id) stay together
        on the same side of the split (prevents train/test leakage from
        HAM10000's multiple photos-per-lesion), and
      - class balance is approximately preserved on both sides.

    StratifiedGroupKFold only supports fold-based fractions (~1/n_splits), so
    `test_fraction` is honored approximately, not exactly -- this is an
    inherent trade-off of grouped splitting on a dataset with an uneven
    number of images per lesion.
    """
    n_splits = max(2, round(1.0 / test_fraction))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    kept_idx, held_out_idx = next(sgkf.split(np.zeros(len(labels)), labels, groups))
    return kept_idx, held_out_idx


def compute_class_weights(labels: np.ndarray, num_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency class weights: weight_c = N / (num_classes * count_c)."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1, None)  # avoid div-by-zero if a class is absent
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(args):
    mask_dir = args.mask_dir or (config.HAM10000_DIR / "predicted_masks")

    train_tf = get_classification_transforms(args.image_size, train=True)
    eval_tf = get_classification_transforms(args.image_size, train=False)

    full_ds_for_split = HAM10000Dataset(transform=None, mask_dir=mask_dir)
    labels = full_ds_for_split.df["label"].values
    groups = full_ds_for_split.df["lesion_id"].values

    train_idx, temp_idx = _grouped_stratified_split(
        labels, groups, test_fraction=(1 - config.TRAIN_SPLIT), seed=config.SEED
    )
    val_size_within_temp = config.VAL_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
    val_idx_in_temp, test_idx_in_temp = _grouped_stratified_split(
        labels[temp_idx], groups[temp_idx], test_fraction=(1 - val_size_within_temp), seed=config.SEED
    )
    val_idx = temp_idx[val_idx_in_temp]
    test_idx = temp_idx[test_idx_in_temp]

    n = len(full_ds_for_split)
    logging.info(
        f"Lesion-grouped split -> train: {len(train_idx)} ({len(train_idx)/n:.1%}), "
        f"val: {len(val_idx)} ({len(val_idx)/n:.1%}), test: {len(test_idx)} ({len(test_idx)/n:.1%}) "
        f"[grouped by lesion_id to prevent leakage; ratios are approximate]"
    )

    train_base = HAM10000Dataset(transform=train_tf, mask_dir=mask_dir, indices=train_idx)
    val_base = HAM10000Dataset(transform=eval_tf, mask_dir=mask_dir, indices=val_idx)
    test_base = HAM10000Dataset(transform=eval_tf, mask_dir=mask_dir, indices=test_idx)

    train_ds = TopologyAugmentedDataset(train_base)
    val_ds = TopologyAugmentedDataset(val_base)
    test_ds = TopologyAugmentedDataset(test_base)

    train_labels = train_base.df["label"].values
    logging.info(f"Train split class balance -> benign: {(train_labels==0).sum()}, "
                  f"malignant: {(train_labels==1).sum()} "
                  f"({(train_labels==1).mean():.1%} malignant)")

    if args.balanced_sampling:
        class_counts = np.bincount(train_labels, minlength=2)
        sample_weights = 1.0 / class_counts[train_labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = make_dataloader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                        num_workers=args.num_workers, drop_last=True)
        logging.info("Using WeightedRandomSampler for ~balanced training batches.")
    else:
        train_loader = make_dataloader(train_ds, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, drop_last=True)

    val_loader = make_dataloader(val_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers)
    test_loader = make_dataloader(test_ds, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers)

    class_weights = compute_class_weights(train_labels) if args.class_weighting == "balanced" else None
    if class_weights is not None:
        logging.info(f"Class-weighted loss enabled -> weights (benign, malignant): "
                      f"{class_weights[0]:.3f}, {class_weights[1]:.3f}")

    return train_loader, val_loader, test_loader, class_weights


def apply_backbone_freeze_policy(model: HybridSkinCancerModel, args) -> None:
    """
    Applies exactly one of three backbone adaptation modes and logs how many
    backbone parameters ended up trainable, so it's never ambiguous what a
    given run actually did:
        1. --freeze-backbone            -> 0% trainable
        2. --unfreeze-last-n-blocks N   -> last N blocks/stages trainable
        3. neither flag                 -> 100% trainable (full fine-tuning, the default)
    """
    if args.freeze_backbone:
        model.backbone.unfreeze_last_n_blocks(0)
        logging.info(f"Backbone mode: FULLY FROZEN. {model.backbone.trainable_backbone_summary()}")
    elif args.unfreeze_last_n_blocks > 0:
        model.backbone.unfreeze_last_n_blocks(args.unfreeze_last_n_blocks)
        logging.info(f"Backbone mode: PARTIAL ({args.unfreeze_last_n_blocks} blocks/stages unfrozen). "
                      f"{model.backbone.trainable_backbone_summary()}")
    else:
        logging.info(f"Backbone mode: FULL FINE-TUNING (default). "
                      f"{model.backbone.trainable_backbone_summary()}")


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            images = batch["image"].to(device)
            topo = batch["topo_features"].to(device)
            labels = batch["label"].to(device)

            if train:
                optimizer.zero_grad()

            logits = model(images, topo)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = classification_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = avg_loss
    return metrics


def main():
    args = parse_args()
    set_seed(config.SEED)
    setup_logging(config.LOG_DIR / "train.log")
    device = get_device(config.DEVICE)
    logging.info(f"Using device: {device} | backbone={args.backbone} | fusion={args.fusion} | "
                  f"checkpoint_metric={args.checkpoint_metric}")

    mask_dir = Path(args.mask_dir or (config.HAM10000_DIR / "predicted_masks"))
    if not mask_dir.exists():
        logging.warning(
            f"Predicted mask directory {mask_dir} does not exist yet. "
            "Run `python -m segmentation.predict_masks --dataset ham10000` first, "
            "otherwise topology features will be computed on empty masks."
        )

    train_loader, val_loader, test_loader, class_weights = build_dataloaders(args)

    model = HybridSkinCancerModel(
        backbone_name=args.backbone, fusion_type=args.fusion,
        pretrained=not args.no_pretrained, dropout=args.dropout,
    ).to(device)
    logging.info(f"Model parameters: {count_parameters(model):,} | dropout={args.dropout}")

    apply_backbone_freeze_policy(model, args)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=config.LABEL_SMOOTHING,
        weight=class_weights.to(device) if class_weights is not None else None,
    )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 1
    best_val_metric = -1.0
    epochs_without_improvement = 0
    if args.resume:
        from utils.common import load_checkpoint
        ckpt = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_metric = ckpt.get("best_val_metric", ckpt.get("best_val_f1", -1.0))
        logging.info(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        logging.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_precision={val_metrics['precision']:.4f} val_recall={val_metrics['recall']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_f2={val_metrics['f2']:.4f} "
            f"val_auc={val_metrics.get('roc_auc')}"
        )
        gap = train_metrics["accuracy"] - val_metrics["accuracy"]
        if gap > 0.15:
            logging.warning(f"  train/val accuracy gap = {gap:.3f} -- likely overfitting. "
                              "Consider --freeze-backbone, fewer --unfreeze-last-n-blocks, "
                              "a higher --dropout, or --early-stopping-patience.")

        checkpoint_state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "backbone_name": args.backbone,
            "fusion_type": args.fusion,
            "dropout": args.dropout,
            "val_metrics": val_metrics,
            "best_val_metric": best_val_metric,
            "checkpoint_metric": args.checkpoint_metric,
        }
        save_checkpoint(checkpoint_state, config.HYBRID_CHECKPOINT_LAST)

        current_metric = val_metrics[args.checkpoint_metric]
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            epochs_without_improvement = 0
            checkpoint_state["best_val_metric"] = best_val_metric
            save_checkpoint(checkpoint_state, config.HYBRID_CHECKPOINT_BEST)
            logging.info(f"  -> saved new best checkpoint "
                          f"(val_{args.checkpoint_metric}={best_val_metric:.4f}) "
                          f"to {config.HYBRID_CHECKPOINT_BEST}")
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                logging.info(f"Early stopping: no improvement in val_{args.checkpoint_metric} "
                              f"for {epochs_without_improvement} epochs.")
                break

    if not Path(config.HYBRID_CHECKPOINT_BEST).exists():
        logging.warning("No best checkpoint was saved during training; skipping final test evaluation.")
        return

    logging.info("Training complete. Evaluating best checkpoint on the held-out test split...")
    from utils.common import load_checkpoint
    best_ckpt = load_checkpoint(config.HYBRID_CHECKPOINT_BEST, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    logging.info("\n" + format_metrics_report(test_metrics))


if __name__ == "__main__":
    main()

"""
evaluate.py
------------
Standalone evaluation of a trained hybrid model checkpoint on a held-out
split (Sec. 12 of the proposal: Accuracy, Precision, Recall, F1, ROC-AUC,
Confusion Matrix), with an option to evaluate on PH2 as external
validation (Sec. 6, Phase 1: "PH2 (External Validation)").

Usage:
    # Evaluate on the HAM10000 test split (same split logic as train.py)
    python evaluate.py --checkpoint checkpoints/hybrid_model_best.pth --dataset ham10000

    # External validation on PH2
    python evaluate.py --checkpoint checkpoints/hybrid_model_best.pth --dataset ph2

    # Threshold sweep to pick an operating point
    python evaluate.py --checkpoint checkpoints/hybrid_model_best.pth --dataset ham10000 --sweep-thresholds
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn

import config
from data.datasets import HAM10000Dataset, PH2Dataset
from data.topo_wrapper import TopologyAugmentedDataset
from models.hybrid_model import HybridSkinCancerModel
from utils.common import get_device, load_checkpoint, setup_logging, set_seed, make_dataloader
from utils.metrics import classification_metrics, format_metrics_report, threshold_sweep, format_threshold_sweep
from utils.transforms import get_classification_transforms


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained hybrid model checkpoint")
    p.add_argument("--checkpoint", type=str, default=str(config.HYBRID_CHECKPOINT_BEST))
    p.add_argument("--dataset", choices=["ham10000", "ph2"], default="ham10000")
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--mask-dir", type=str, default=None,
                   help="Predicted mask directory for HAM10000 "
                        "(default: datasets/HAM10000/predicted_masks). Ignored for PH2, which "
                        "uses datasets/PH2/predicted_masks (see segmentation/predict_masks.py).")
    p.add_argument("--output-json", type=str, default=None,
                   help="Optional path to dump the metrics dict as JSON.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Decision threshold on P(malignant). Lower than 0.5 trades precision for "
                        "recall -- use --sweep-thresholds first to pick a value deliberately.")
    p.add_argument("--sweep-thresholds", action="store_true",
                   help="Print precision/recall/F1/F2 across a range of thresholds instead of a single "
                        "report, to help choose an operating point for a screening use case.")
    return p.parse_args()


def build_eval_loader(args):
    eval_tf = get_classification_transforms(args.image_size, train=False)

    if args.dataset == "ham10000":
        mask_dir = args.mask_dir or (config.HAM10000_DIR / "predicted_masks")
        # Re-derive the same held-out test split used in train.py (lesion-
        # grouped, so no lesion's photos leak between splits) so this script
        # reports on genuinely unseen samples, not the training set.
        from train import _grouped_stratified_split

        full_ds_for_split = HAM10000Dataset(transform=None, mask_dir=mask_dir)
        labels = full_ds_for_split.df["label"].values
        groups = full_ds_for_split.df["lesion_id"].values

        train_idx, temp_idx = _grouped_stratified_split(
            labels, groups, test_fraction=(1 - config.TRAIN_SPLIT), seed=config.SEED
        )
        val_size_within_temp = config.VAL_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
        _, test_idx_in_temp = _grouped_stratified_split(
            labels[temp_idx], groups[temp_idx], test_fraction=(1 - val_size_within_temp), seed=config.SEED
        )
        test_idx = temp_idx[test_idx_in_temp]
        base_ds = HAM10000Dataset(transform=eval_tf, mask_dir=mask_dir, indices=test_idx)

    elif args.dataset == "ph2":
        mask_dir = config.PH2_DIR / "predicted_masks"
        base_ds = PH2Dataset(transform=eval_tf)
        # NOTE: PH2Dataset reads its own ground-truth lesion masks
        # (IMDxxx_lesion.bmp) directly, so `mask_dir` (predicted masks) is
        # only used as a fallback reference and is not wired in here by
        # default; ground-truth masks give a cleaner external-validation
        # signal for the topology features.
    else:
        raise ValueError(args.dataset)

    ds = TopologyAugmentedDataset(base_ds)
    loader = make_dataloader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return loader


@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5):
    model.eval()
    all_labels, all_preds, all_probs, all_ids = [], [], [], []

    for batch in loader:
        images = batch["image"].to(device)
        topo = batch["topo_features"].to(device)
        labels = batch["label"]

        logits = model(images, topo)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs >= threshold).long()

        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_ids.extend(batch["image_id"])

    # PH2 samples with unknown diagnosis are labeled -1 by PH2Dataset and
    # must be excluded from metric computation (they have no ground truth).
    valid = [i for i, y in enumerate(all_labels) if y in (0, 1)]
    if len(valid) < len(all_labels):
        logging.warning(f"Excluding {len(all_labels) - len(valid)} samples with unknown label from metrics.")
    y_true = [all_labels[i] for i in valid]
    y_pred = [all_preds[i] for i in valid]
    y_prob = [all_probs[i] for i in valid]

    metrics = classification_metrics(y_true, y_pred, y_prob)
    return metrics, {"image_ids": all_ids, "labels": all_labels, "preds": all_preds, "probs": all_probs,
                      "y_true_valid": y_true, "y_prob_valid": y_prob}


def main():
    args = parse_args()
    set_seed(config.SEED)
    setup_logging()
    device = get_device(config.DEVICE)

    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    backbone_name = ckpt.get("backbone_name", config.BACKBONE_NAME)
    fusion_type = ckpt.get("fusion_type", config.FUSION_TYPE)
    logging.info(f"Loaded checkpoint {args.checkpoint} "
                  f"(epoch={ckpt.get('epoch')}, backbone={backbone_name}, fusion={fusion_type})")

    model = HybridSkinCancerModel(backbone_name=backbone_name, fusion_type=fusion_type,
                                   pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    loader = build_eval_loader(args)
    logging.info(f"Evaluating on {args.dataset} ({len(loader.dataset)} samples)...")

    if args.sweep_thresholds:
        _, raw = evaluate(model, loader, device, threshold=0.5)
        rows = threshold_sweep(raw["y_true_valid"], raw["y_prob_valid"])
        logging.info("\n" + format_threshold_sweep(rows))
        logging.info("Pick a threshold from the table above (higher recall = lower threshold), "
                      "then rerun with --threshold <value> for a full report at that operating point.")
        if args.output_json:
            out_path = Path(args.output_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump({"threshold_sweep": rows}, f, indent=2)
            logging.info(f"Saved threshold sweep to {out_path}")
        return

    metrics, raw = evaluate(model, loader, device, threshold=args.threshold)
    logging.info(f"(decision threshold = {args.threshold})")
    logging.info("\n" + format_metrics_report(metrics))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"metrics": metrics, "threshold": args.threshold, "predictions": raw}, f, indent=2)
        logging.info(f"Saved detailed results to {out_path}")


if __name__ == "__main__":
    main()

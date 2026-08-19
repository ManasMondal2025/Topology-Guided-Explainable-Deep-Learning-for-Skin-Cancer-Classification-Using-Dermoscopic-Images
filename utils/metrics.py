"""
utils/metrics.py
-----------------
Classification metrics (Sec. 12 of the proposal: Accuracy, Precision,
Recall, F1, ROC-AUC, Confusion Matrix) and segmentation metrics
(Dice, IoU).
"""

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    confusion_matrix,
)


def classification_metrics(y_true, y_pred, y_prob=None) -> Dict[str, object]:
    """
    y_true, y_pred: 1D arrays of {0,1} (0=benign, 1=malignant)
    y_prob: 1D array of predicted probability of the positive (malignant) class,
            required for ROC-AUC.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        # F2 weights recall 2x more than precision -- the right metric to
        # optimize for in a cancer screening context, where a missed
        # malignant case (false negative) is far costlier than a false alarm.
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = None
    return metrics


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-7) -> float:
    """Dice coefficient between two binary masks (0/1)."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    return float((2.0 * intersection + eps) / (pred.sum() + gt.sum() + eps))


def iou_score(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-7) -> float:
    """Intersection-over-Union between two binary masks (0/1)."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((intersection + eps) / (union + eps))


def format_metrics_report(metrics: Dict[str, object]) -> str:
    lines = ["=" * 50, "Classification Report", "=" * 50]
    for key in ["accuracy", "precision", "recall", "f1", "f2", "roc_auc"]:
        val = metrics.get(key)
        if val is not None:
            lines.append(f"{key:>12s}: {val:.4f}")
        else:
            lines.append(f"{key:>12s}: N/A")
    lines.append(f"confusion_matrix: {metrics.get('confusion_matrix')}")
    lines.append("=" * 50)
    return "\n".join(lines)


def threshold_sweep(y_true, y_prob, thresholds: List[float] = None) -> List[Dict[str, float]]:
    """
    Sweeps the malignant-class decision threshold (default: argmax, i.e. 0.5)
    across a range of values and reports precision/recall/F1/F2 at each, so a
    threshold can be chosen deliberately rather than defaulting to 0.5.

    In a screening context you typically want the LOWEST threshold that keeps
    precision above some acceptable floor -- i.e. flag more borderline cases
    as "malignant, needs review" rather than clear the model at 0.5 and risk
    missing cases sitting at 0.4-0.5 confidence.
    """
    thresholds = thresholds or [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8]
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        })
    return rows


def format_threshold_sweep(rows: List[Dict[str, float]]) -> str:
    lines = ["=" * 66, f"{'threshold':>10s} {'precision':>10s} {'recall':>10s} {'f1':>10s} {'f2':>10s}", "=" * 66]
    for row in rows:
        lines.append(f"{row['threshold']:>10.2f} {row['precision']:>10.4f} "
                      f"{row['recall']:>10.4f} {row['f1']:>10.4f} {row['f2']:>10.4f}")
    lines.append("=" * 66)
    return "\n".join(lines)

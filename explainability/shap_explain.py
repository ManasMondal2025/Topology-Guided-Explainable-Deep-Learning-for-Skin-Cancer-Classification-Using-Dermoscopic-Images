"""
explainability/shap_explain.py
---------------------------------
SHAP feature-importance explanations (Sec. 6, Phase 8) using the `shap`
library (https://github.com/shap/shap).

Shows *which features contributed most* to a prediction. Because the deep
ViT/EfficientNet/MobileNetV3 embedding dimensions are not individually interpretable,
this module explains the model at the level that IS interpretable:
    - each of the named topological/shape descriptors (Betti numbers,
      circularity, compactness, convexity, fractal dimension, boundary
      complexity), and
    - the overall deep-feature contribution as a single summarized group.

This is done with `shap.KernelExplainer`, a model-agnostic explainer that
only needs a callable `f(X) -> predictions`, so it works regardless of
which fusion module (gated or attention) is used.
"""

from typing import Dict, List
import logging

import numpy as np
import torch
import shap

import config
from topology.gudhi_features import compute_gudhi_features

# shap's KernelExplainer logs verbose internal diagnostics (subset weights,
# phi vectors, etc.) at INFO level, which otherwise clutters this project's
# logging output since utils.common.setup_logging configures the root logger.
logging.getLogger("shap").setLevel(logging.WARNING)


# named, human-interpretable topological features (first 7 dims of the
# topology vector produced by topology/extract.py)
NAMED_TOPO_FEATURES = [
    "betti_0", "betti_1", "circularity", "compactness",
    "convexity", "fractal_dimension", "boundary_complexity",
]


def _make_predict_fn(hybrid_model, image_tensor, full_topo_features, device, max_batch: int = 8):
    """
    Builds a prediction function f(X) -> P(malignant) where X only varies
    the 7 named topological features; the persistence-image/landscape tail
    of the topology vector and the image are held fixed at their original
    values. This isolates the effect of the interpretable descriptors.

    Internally chunks large calls into batches of at most `max_batch` rows
    -- SHAP's KernelExplainer can request more rows at once than the
    `nsamples` argument suggests (it depends on the number of features and
    subset-sampling strategy), so this keeps memory use bounded regardless.
    """
    fixed_tail = full_topo_features[7:].clone()

    def predict_fn(X: np.ndarray) -> np.ndarray:
        all_probs = []
        for start in range(0, len(X), max_batch):
            chunk = X[start:start + max_batch]
            X_t = torch.as_tensor(chunk, dtype=torch.float32, device=device)
            batch_size = X_t.shape[0]
            tail = fixed_tail.unsqueeze(0).expand(batch_size, -1).to(device)
            topo_batch = torch.cat([X_t, tail], dim=1)
            image_batch = image_tensor.expand(batch_size, -1, -1, -1).to(device)

            with torch.no_grad():
                logits = hybrid_model(image_batch, topo_batch)
                probs = torch.softmax(logits, dim=1)[:, 1]  # P(malignant)
            all_probs.append(probs.cpu().numpy())
        return np.concatenate(all_probs)

    return predict_fn


def explain_topological_contributions(hybrid_model, image_tensor: torch.Tensor,
                                       topo_features: torch.Tensor,
                                       background_samples: int = 10,
                                       nsamples: int = 50,
                                       device=None) -> Dict[str, float]:
    """
    Args:
        hybrid_model: trained HybridSkinCancerModel (eval mode)
        image_tensor: Tensor[1, 3, H, W] for the single image being explained
        topo_features: Tensor[1, topo_dim] for this image's lesion mask
        background_samples: size of the synthetic background dataset used
                             by KernelExplainer (perturbations around the
                             observed values)
        nsamples: number of coalition samples KernelExplainer uses to
                  estimate Shapley values (higher = more accurate, slower)

    Returns:
        dict mapping each named topological feature -> its SHAP value
        for the "malignant" class prediction on this single image.
    """
    device = device or next(hybrid_model.parameters()).device
    hybrid_model.eval()

    named_values = topo_features[0, :7].detach().cpu().numpy().reshape(1, -1)

    # background: small random perturbations around the observed feature
    # values, standard practice for KernelExplainer with a single instance
    rng = np.random.default_rng(config.SEED)
    background = named_values + rng.normal(scale=0.05, size=(background_samples, named_values.shape[1]))

    predict_fn = _make_predict_fn(hybrid_model, image_tensor, topo_features[0], device)

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(named_values, nsamples=nsamples, silent=True)

    shap_values = np.asarray(shap_values).reshape(-1)
    return {name: float(val) for name, val in zip(NAMED_TOPO_FEATURES, shap_values)}


def format_shap_report(shap_dict: Dict[str, float]) -> str:
    lines = ["SHAP feature contributions (towards 'malignant'):"]
    for name, val in sorted(shap_dict.items(), key=lambda kv: -abs(kv[1])):
        direction = "-> increases malignant risk" if val > 0 else "-> decreases malignant risk"
        lines.append(f"  {name:>20s}: {val:+.4f}  {direction}")
    return "\n".join(lines)


if __name__ == "__main__":
    from models.hybrid_model import HybridSkinCancerModel

    model = HybridSkinCancerModel(backbone_name="efficientnet_b3", pretrained=False).eval()
    image = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    topo = torch.randn(1, config.TOPO_FEATURE_DIM)

    result = explain_topological_contributions(model, image, topo, background_samples=10, nsamples=50)
    print(format_shap_report(result))
    print("OK")

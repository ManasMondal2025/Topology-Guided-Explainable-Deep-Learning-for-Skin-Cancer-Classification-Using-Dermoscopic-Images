"""
topology/extract.py
---------------------
Single entry point that combines topology/gudhi_features.py (Betti numbers +
shape descriptors) with topology/persistence.py (persistence image +
landscape statistics) into the final topological feature vector consumed
by the fusion module (Sec. 6, Phase 4 + Phase 6 of the proposal).
"""

import numpy as np

from topology.gudhi_features import compute_gudhi_features
from topology.persistence import compute_persistence_features


def extract_topological_features(mask: np.ndarray) -> np.ndarray:
    """
    Args:
        mask: HxW binary (or {0,255}) numpy array — the lesion segmentation
              mask produced by segmentation/unet.py.

    Returns:
        1D float32 numpy array of length `get_topology_feature_dim()`
        (currently 34): [betti_0, betti_1, circularity, compactness,
        convexity, fractal_dimension, boundary_complexity,
        persistence_image(flattened), landscape_mean, landscape_max]
    """
    gudhi_vec, _ = compute_gudhi_features(mask)
    persistence_vec = compute_persistence_features(mask)
    return np.concatenate([gudhi_vec, persistence_vec]).astype(np.float32)


def get_topology_feature_dim() -> int:
    """Probes the pipeline on a dummy mask to determine the exact output
    dimensionality — used by fusion/*.py so the model architecture always
    matches whatever the topology pipeline actually produces, even if the
    persistence-image resolution is changed in config.py."""
    dummy = np.zeros((64, 64), dtype=np.uint8)
    import cv2
    cv2.circle(dummy, (32, 32), 20, 255, -1)
    return extract_topological_features(dummy).shape[0]


if __name__ == "__main__":
    import cv2
    m = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(m, (64, 64), 40, 255, -1)
    cv2.circle(m, (64, 64), 10, 0, -1)
    feats = extract_topological_features(m)
    print("Full topological feature vector shape:", feats.shape)
    print("get_topology_feature_dim():", get_topology_feature_dim())

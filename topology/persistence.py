"""
topology/persistence.py
-------------------------
Point-cloud persistent homology features (Sec. 6, Phase 4 of the proposal):
Persistence Diagram, Persistence Image, Persistence Landscape.

The lesion boundary contour is sampled as a 2D point cloud and a
Vietoris-Rips filtration is computed with `ripser`. The resulting H1
persistence diagram is vectorized with `persim` into:
    - a fixed-size Persistence Image
    - summary statistics of the Persistence Landscape

This complements the cubical-complex Betti numbers + shape descriptors in
topology/gudhi_features.py.
"""

from typing import Tuple

import numpy as np
import cv2
from ripser import ripser
from persim import PersistenceImager
from persim.landscapes import PersLandscapeApprox

import config


def _boundary_point_cloud(mask: np.ndarray, num_points: int = 200) -> np.ndarray:
    """
    Extracts the largest contour of the mask and subsamples it to
    `num_points` (approximately) evenly spaced 2D points, normalized to
    the unit square so scale doesn't bias the filtration.
    """
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2), dtype=np.float64)

    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if len(contour) > num_points:
        idx = np.linspace(0, len(contour) - 1, num_points).astype(int)
        contour = contour[idx]

    # normalize to unit square (translation + scale invariant)
    mins = contour.min(axis=0)
    maxs = contour.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    contour = (contour - mins) / span
    return contour


def compute_persistence_diagram(mask: np.ndarray, num_points: int = 200, maxdim: int = 1):
    """
    Runs ripser on the (normalized) boundary point cloud of the lesion.
    Returns the list of persistence diagrams (one array per homology
    dimension, each row = [birth, death]).
    """
    points = _boundary_point_cloud(mask, num_points=num_points)
    if len(points) < 4:
        return [np.zeros((0, 2)) for _ in range(maxdim + 1)]
    result = ripser(points, maxdim=maxdim)
    return result["dgms"]


def compute_persistence_image(dgms, resolution: Tuple[int, int] = None, homology_dim: int = 1) -> np.ndarray:
    """
    Vectorizes the H1 (or H0) persistence diagram into a fixed-size
    Persistence Image using `persim.PersistenceImager`.
    Returns a flattened float32 array of length resolution[0]*resolution[1].
    """
    resolution = resolution or config.PERSISTENCE_IMAGE_RESOLUTION
    dgm = dgms[homology_dim] if homology_dim < len(dgms) else np.zeros((0, 2))

    # remove infinite death times (points that never die) for the imager
    finite_dgm = dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else dgm

    if len(finite_dgm) == 0:
        return np.zeros(resolution[0] * resolution[1], dtype=np.float32)

    # NOTE: birth_range/pers_range are fixed explicitly (rather than via
    # `.fit()`, which recomputes them from the data) so every lesion produces
    # a persistence image on the SAME grid and therefore a fixed-length,
    # directly-comparable feature vector. This relies on the boundary point
    # cloud already being normalized to the unit square in
    # `_boundary_point_cloud`, so birth/persistence values lie in [0, 1].
    pixel_size = 1.0 / max(resolution)
    pimgr = PersistenceImager(pixel_size=pixel_size, birth_range=(0, 1), pers_range=(0, 1))
    try:
        img = np.array(pimgr.transform([finite_dgm], skew=True)[0], dtype=np.float32)
    except Exception:
        return np.zeros(resolution[0] * resolution[1], dtype=np.float32)

    if img.shape != tuple(resolution):
        img = cv2.resize(img, resolution, interpolation=cv2.INTER_LINEAR)
    return img.flatten().astype(np.float32)


def compute_persistence_landscape_stats(dgms, homology_dim: int = 1,
                                         num_landscapes: int = None,
                                         resolution: int = None) -> np.ndarray:
    """
    Computes the Persistence Landscape for a given homology dimension and
    returns compact summary statistics (mean and max of the first landscape
    level) rather than the full curve, to keep the fused feature vector
    small (Sec. 6, Phase 6: topological features are 30-50 dims).
    """
    num_landscapes = num_landscapes or config.PERSISTENCE_LANDSCAPE_NUM_LANDSCAPES
    resolution = resolution or config.PERSISTENCE_LANDSCAPE_RESOLUTION

    dgm = dgms[homology_dim] if homology_dim < len(dgms) else np.zeros((0, 2))
    finite_dgm = dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else dgm

    if len(finite_dgm) == 0:
        return np.zeros(2, dtype=np.float32)

    try:
        landscape = PersLandscapeApprox(
            dgms=[finite_dgm], hom_deg=0, num_steps=resolution,
        )
        values = np.array(landscape.values, dtype=np.float64)
        if values.size == 0:
            return np.zeros(2, dtype=np.float32)
        return np.array([values.mean(), values.max()], dtype=np.float32)
    except Exception:
        # Fall back to birth/death-based summary if persim's landscape API
        # differs across versions.
        life = finite_dgm[:, 1] - finite_dgm[:, 0]
        return np.array([life.mean(), life.max()], dtype=np.float32)


def compute_persistence_features(mask: np.ndarray) -> np.ndarray:
    """
    Full point-cloud persistence feature vector:
        persistence_image (flattened, H1) + persistence_landscape_stats (H1)
    """
    dgms = compute_persistence_diagram(mask)
    pers_image = compute_persistence_image(dgms, homology_dim=1)
    landscape_stats = compute_persistence_landscape_stats(dgms, homology_dim=1)
    return np.concatenate([pers_image, landscape_stats]).astype(np.float32)


if __name__ == "__main__":
    m = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(m, (64, 64), 40, 255, -1)
    cv2.circle(m, (64, 64), 10, 0, -1)
    feats = compute_persistence_features(m)
    print("Persistence feature vector shape:", feats.shape)
    print(feats)

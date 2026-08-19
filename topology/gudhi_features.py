"""
topology/gudhi_features.py
----------------------------
Topological + geometric descriptors computed directly from a binary lesion
mask (Sec. 6, Phase 4 of the proposal):

    - Betti-0 (connected components) and Betti-1 (holes) via a cubical
      complex persistent-homology computation (GUDHI).
    - Classical shape descriptors used by dermatologists as ABCD-rule
      proxies: circularity, compactness, convexity, fractal dimension,
      boundary complexity.

These are combined with the point-cloud persistence features in
topology/persistence.py to form the full topological feature vector
(see topology/__init__.py: extract_topological_features).
"""

from typing import Dict, Tuple

import numpy as np
import cv2
import gudhi as gd


def _largest_contour(mask: np.ndarray):
    """Returns the largest external contour of a binary mask (uint8, {0,255} or {0,1})."""
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def compute_betti_numbers(mask: np.ndarray, max_dim: int = 1) -> Dict[str, int]:
    """
    Builds a cubical complex from the binary mask and computes the Betti
    numbers via persistent homology (GUDHI CubicalComplex).

    Foreground pixels (lesion) get filtration value 0, background pixels get
    filtration value 1, so that persistence pairs born at 0 and never dying
    (or dying at 1) correspond to topological features of the lesion shape.

    Returns:
        {"betti_0": int, "betti_1": int}
    """
    binary = (mask > 0).astype(np.float64)
    filtration = 1.0 - binary  # foreground=0 (appears first), background=1

    cc = gd.CubicalComplex(top_dimensional_cells=filtration)
    cc.compute_persistence()
    betti_numbers = cc.persistent_betti_numbers(from_value=0.0, to_value=0.0)

    betti_0 = int(betti_numbers[0]) if len(betti_numbers) > 0 else 0
    betti_1 = int(betti_numbers[1]) if len(betti_numbers) > 1 else 0
    return {"betti_0": betti_0, "betti_1": betti_1}


def compute_circularity(area: float, perimeter: float) -> float:
    """4*pi*Area / Perimeter^2. 1.0 for a perfect circle, lower = more irregular."""
    if perimeter <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter ** 2))


def compute_compactness(area: float, perimeter: float) -> float:
    """Area / Perimeter^2 (unnormalized compactness; higher = more compact)."""
    if perimeter <= 0:
        return 0.0
    return float(area / (perimeter ** 2))


def compute_convexity(contour: np.ndarray) -> float:
    """Contour area / convex-hull area. 1.0 = perfectly convex boundary."""
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    contour_area = cv2.contourArea(contour)
    if hull_area <= 0:
        return 0.0
    return float(contour_area / hull_area)


def compute_boundary_complexity(contour: np.ndarray) -> float:
    """Perimeter / convex-hull perimeter. >=1; higher = more irregular boundary."""
    hull = cv2.convexHull(contour)
    hull_perimeter = cv2.arcLength(hull, True)
    perimeter = cv2.arcLength(contour, True)
    if hull_perimeter <= 0:
        return 1.0
    return float(perimeter / hull_perimeter)


def compute_fractal_dimension(mask: np.ndarray, min_box_size: int = 2, max_box_size: int = 64) -> float:
    """
    Box-counting fractal dimension of the lesion boundary.
    Fits log(N(box_size)) vs log(1/box_size) with a linear regression;
    the slope approximates the fractal dimension.
    """
    binary = (mask > 0).astype(np.uint8)
    boundary = cv2.Canny(binary * 255, 50, 150) if binary.max() > 0 else binary
    ys, xs = np.nonzero(boundary)
    if len(xs) == 0:
        return 1.0

    box_sizes = [s for s in [2, 4, 8, 16, 32, 64] if min_box_size <= s <= max_box_size]
    counts = []
    for size in box_sizes:
        grid_x = xs // size
        grid_y = ys // size
        occupied = set(zip(grid_x.tolist(), grid_y.tolist()))
        counts.append(len(occupied))

    counts = np.array(counts, dtype=np.float64)
    sizes = np.array(box_sizes, dtype=np.float64)
    valid = counts > 0
    if valid.sum() < 2:
        return 1.0

    log_inv_size = np.log(1.0 / sizes[valid])
    log_counts = np.log(counts[valid])
    slope, _ = np.polyfit(log_inv_size, log_counts, 1)
    return float(np.clip(slope, 0.5, 3.0))


def compute_shape_descriptors(mask: np.ndarray) -> Dict[str, float]:
    """
    Returns the five classical shape descriptors listed in the proposal
    (Sec. 6, Phase 4): circularity, compactness, convexity, fractal
    dimension, boundary complexity. Falls back to neutral defaults if no
    contour can be found (e.g. an empty mask).
    """
    contour = _largest_contour(mask)
    if contour is None or len(contour) < 5:
        return {
            "circularity": 0.0,
            "compactness": 0.0,
            "convexity": 0.0,
            "fractal_dimension": 1.0,
            "boundary_complexity": 1.0,
        }

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    return {
        "circularity": compute_circularity(area, perimeter),
        "compactness": compute_compactness(area, perimeter),
        "convexity": compute_convexity(contour),
        "fractal_dimension": compute_fractal_dimension(mask),
        "boundary_complexity": compute_boundary_complexity(contour),
    }


def compute_gudhi_features(mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Convenience entry point: returns (feature_vector, named_dict) combining
    Betti numbers + shape descriptors (7 dims total).

    Order: [betti_0, betti_1, circularity, compactness, convexity,
            fractal_dimension, boundary_complexity]
    """
    betti = compute_betti_numbers(mask)
    shape = compute_shape_descriptors(mask)

    named = {**betti, **shape}
    vector = np.array([
        named["betti_0"],
        named["betti_1"],
        named["circularity"],
        named["compactness"],
        named["convexity"],
        named["fractal_dimension"],
        named["boundary_complexity"],
    ], dtype=np.float32)
    return vector, named


if __name__ == "__main__":
    # smoke test on a synthetic lesion-like mask (irregular blob with a hole)
    m = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(m, (64, 64), 40, 255, -1)
    cv2.ellipse(m, (80, 50), (15, 25), 30, 0, 360, 255, -1)
    cv2.circle(m, (64, 64), 10, 0, -1)  # punch a hole -> Betti-1 should be >= 1
    vec, named = compute_gudhi_features(m)
    print("Feature vector:", vec)
    print("Named:", named)

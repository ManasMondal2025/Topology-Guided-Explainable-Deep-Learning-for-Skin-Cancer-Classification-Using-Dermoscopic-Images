"""
explainability/gradcam.py
----------------------------
Grad-CAM visual explanations (Sec. 6, Phase 8) using the `pytorch-grad-cam`
library (https://github.com/jacobgil/pytorch-grad-cam).

Shows *where* the model looked when making its prediction, overlaid on the
original dermoscopic image.

Works with both backbones:
    - ViT-B/16: uses GradCAM with a `reshape_transform` that reshapes the
      [B, num_tokens, dim] transformer output back into a spatial
      [B, dim, H, W] feature map (standard approach for attention-based
      Grad-CAM, following jacobgil/pytorch-grad-cam's ViT example).
    - EfficientNet-B3: uses GradCAM directly on the last convolutional layer.
"""

from typing import Optional

import numpy as np
import cv2
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import config


def _vit_reshape_transform(tensor: torch.Tensor, patch_grid: int = 14):
    """
    Reshapes a ViT layer's [B, num_tokens, dim] activations (including the
    CLS token as the first entry) into a [B, dim, grid, grid] spatial map
    for Grad-CAM. For ViT-B/16 at 224x224 input, patch_grid = 224/16 = 14.
    """
    result = tensor[:, 1:, :]  # drop CLS token
    B, num_patches, dim = result.shape
    grid = int(num_patches ** 0.5) if patch_grid is None else patch_grid
    result = result.reshape(B, grid, grid, dim)
    result = result.permute(0, 3, 1, 2)  # [B, dim, grid, grid]
    return result


class HybridModelGradCAMWrapper(nn.Module):
    """
    Grad-CAM needs a model that maps an image directly to class logits.
    Since HybridSkinCancerModel also requires topological features, this
    wrapper freezes a fixed topo-feature tensor (computed once for the
    image being explained) and exposes a single-argument forward(image).
    """

    def __init__(self, hybrid_model: nn.Module, topo_features: torch.Tensor):
        super().__init__()
        self.hybrid_model = hybrid_model
        self.register_buffer("topo_features", topo_features)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        topo = self.topo_features.expand(image.size(0), -1)
        return self.hybrid_model(image, topo)


def get_gradcam_heatmap(hybrid_model: nn.Module, image_tensor: torch.Tensor,
                         topo_features: torch.Tensor, backbone_name: str,
                         target_class: Optional[int] = None, device=None) -> np.ndarray:
    """
    Args:
        hybrid_model: a trained HybridSkinCancerModel (eval mode recommended)
        image_tensor: Tensor[1, 3, H, W], normalized the same way as training
        topo_features: Tensor[1, topo_dim] for this specific image's lesion mask
        backbone_name: "vit_b16", "efficientnet_b3", or "mobilenet_v3" (selects target layer)
        target_class: which class to explain (default: the predicted class)

    Returns:
        grayscale_cam: np.ndarray[H, W] in [0, 1] — the raw CAM, before overlay.
    """
    device = device or next(hybrid_model.parameters()).device
    wrapped = HybridModelGradCAMWrapper(hybrid_model, topo_features.to(device)).to(device)
    wrapped.eval()

    if backbone_name == "vit_b16":
        target_layers = [wrapped.hybrid_model.backbone.get_last_attention_layer()]
        reshape_transform = _vit_reshape_transform
    else:
        # Any CNN backbone (EfficientNet, MobileNetV3, ...) exposes
        # get_last_conv_layer() via backbone/cnn_backbone.py::CNNBackbone,
        # so no per-architecture branching is needed here.
        target_layers = [wrapped.hybrid_model.backbone.get_last_conv_layer()]
        reshape_transform = None

    cam = GradCAM(model=wrapped, target_layers=target_layers, reshape_transform=reshape_transform)

    targets = None
    if target_class is not None:
        targets = [ClassifierOutputTarget(target_class)]

    grayscale_cam = cam(input_tensor=image_tensor.to(device), targets=targets)
    return grayscale_cam[0]  # first (and only) image in the batch


def overlay_heatmap_on_image(rgb_image_float01: np.ndarray, grayscale_cam: np.ndarray) -> np.ndarray:
    """
    rgb_image_float01: HxWx3 float array in [0, 1] (the ORIGINAL, un-normalized
                        image resized to the same H, W as grayscale_cam)
    grayscale_cam: HxW float array in [0, 1] from get_gradcam_heatmap
    Returns: HxWx3 uint8 BGR image ready to cv2.imwrite
    """
    visualization = show_cam_on_image(rgb_image_float01, grayscale_cam, use_rgb=False)
    return visualization


if __name__ == "__main__":
    # smoke test with a randomly initialized model (no pretrained download needed)
    from models.hybrid_model import HybridSkinCancerModel

    model = HybridSkinCancerModel(backbone_name="efficientnet_b3", pretrained=False).eval()
    image = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    topo = torch.randn(1, config.TOPO_FEATURE_DIM)

    heatmap = get_gradcam_heatmap(model, image, topo, backbone_name="efficientnet_b3")
    print("Grad-CAM heatmap shape:", heatmap.shape, "range:", heatmap.min(), heatmap.max())
    print("OK")

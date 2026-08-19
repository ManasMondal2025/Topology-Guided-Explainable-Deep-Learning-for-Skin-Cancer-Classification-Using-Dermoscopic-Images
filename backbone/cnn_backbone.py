"""
backbone/cnn_backbone.py
---------------------------
Shared base class for CNN backbones loaded via `timm` (EfficientNet-B3,
MobileNetV3, and any other timm CNN dropped in later). Factored out because
EfficientNet and MobileNetV3 wrap identically -- only the timm model name
and default output dimension differ.

Note on dimension probing: `model.num_features` is NOT always reliable as
the actual pooled-output dimension for every timm architecture -- some
models (e.g. mobilenetv3_large_100) report a `num_features` that doesn't
match the tensor shape actually returned when `num_classes=0`, because of
an extra projection layer between the backbone stem and the classifier that
`num_features` doesn't account for. To be safe across architectures, this
class determines the true output dimension with a one-time dummy forward
pass at construction time, rather than trusting the reported attribute.

Both dummy-forward-pass probes below (`_probe_output_dim`,
`_layer_output_is_spatial`) build their probe tensor on the SAME device as
the backbone's own parameters (`next(self.backbone.parameters()).device`),
not a hardcoded CPU tensor -- otherwise, calling these after the model has
been moved to CUDA (e.g. in inference.py, after `model.to(device)`) raises
a CPU/GPU tensor mismatch RuntimeError. `_probe_output_dim` runs at
`__init__` time (before `.to(device)` is typically called, so it rarely
hits this), but `get_last_conv_layer()` -> `_layer_output_is_spatial()` is
called from explainability/gradcam.py at inference time, AFTER the model
is already on GPU -- so it hits this every time on a CUDA setup.
"""

import torch
import torch.nn as nn
import timm

import config


class CNNBackbone(nn.Module):
    """
    Generic timm CNN feature extractor + linear projection to
    `config.DEEP_FEATURE_DIM`, so any CNN backbone is a drop-in replacement
    for ViTBackbone in models/hybrid_model.py.
    """

    def __init__(self, model_name: str, pretrained: bool = True,
                 out_dim: int = None, freeze_backbone: bool = False):
        super().__init__()
        out_dim = out_dim or config.DEEP_FEATURE_DIM

        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        native_dim = self._probe_output_dim()

        self.needs_projection = native_dim != out_dim
        self.projection = nn.Linear(native_dim, out_dim) if self.needs_projection else nn.Identity()

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.model_name = model_name
        self.out_dim = out_dim

    def _probe_output_dim(self) -> int:
        """One-time dummy forward pass to determine the TRUE pooled output
        dimension (see module docstring for why `.num_features` alone isn't
        trustworthy for every architecture)."""
        was_training = self.backbone.training
        self.backbone.eval()
        device = next(self.backbone.parameters()).device
        with torch.no_grad():
            dummy = torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
            out = self.backbone(dummy)
        if was_training:
            self.backbone.train()
        return out.shape[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: Tensor[B, 3, H, W] -> Tensor[B, out_dim]"""
        features = self.backbone(x)
        return self.projection(features)

    def get_last_conv_layer(self):
        """Returns the final SPATIALLY-RESOLVED convolutional layer -- used
        as the Grad-CAM target layer for CNN backbones (see
        explainability/gradcam.py).

        Some timm architectures (e.g. mobilenetv3_large_100) apply
        `conv_head` AFTER global average pooling has already collapsed
        spatial dimensions to 1x1 -- Grad-CAM at that layer would have
        nothing to localize (a degenerate, all-zero heatmap). This is
        detected with a one-time probe; if `conv_head`'s output isn't
        spatial, this falls back to the last pre-pooling stage (`.blocks`),
        which IS spatially resolved.
        """
        if hasattr(self.backbone, "conv_head") and self._layer_output_is_spatial(self.backbone.conv_head):
            return self.backbone.conv_head
        if hasattr(self.backbone, "blocks"):
            return self.backbone.blocks
        raise AttributeError(
            f"Could not find a spatially-resolved conv layer for '{self.model_name}' to use as a "
            "Grad-CAM target. Inspect the model (print(self.backbone)) and adjust "
            "get_last_conv_layer() accordingly."
        )

    def _layer_output_is_spatial(self, layer: nn.Module) -> bool:
        """True if `layer`'s output has spatial dims > 1x1 for a dummy forward pass.
        Builds the dummy input on the backbone's OWN device (see module
        docstring) -- this is called at inference time, after the model may
        already be on CUDA, so a hardcoded CPU tensor here would crash."""
        captured = {}

        def hook(module, inp, output):
            captured["shape"] = output.shape

        handle = layer.register_forward_hook(hook)
        was_training = self.backbone.training
        self.backbone.eval()
        device = next(self.backbone.parameters()).device
        with torch.no_grad():
            self.backbone(torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device))
        if was_training:
            self.backbone.train()
        handle.remove()

        shape = captured.get("shape")
        return shape is not None and len(shape) == 4 and shape[2] > 1 and shape[3] > 1

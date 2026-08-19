"""
backbone/efficientnet.py
--------------------------
EfficientNet-B3 alternative deep-feature backbone (Sec. 6, Phase 5 /
Sec. 8: "Alternative: EfficientNet-B3"). Thin wrapper around
backbone/cnn_backbone.py::CNNBackbone -- see that module for the shared
implementation and a note on why output dimension is probed dynamically
rather than trusted from `.num_features`.

EfficientNet uses compound scaling (jointly scaling depth/width/input
resolution) to get a better accuracy-per-FLOP trade-off than hand-scaled
CNNs, including MobileNet (backbone/mobilenet.py) -- see README.md's
"Backbone comparison" section for the full rationale and benchmark numbers.
"""

import torch

import config
from backbone.cnn_backbone import CNNBackbone


class EfficientNetBackbone(CNNBackbone):
    def __init__(self, model_name: str = None, pretrained: bool = True,
                 out_dim: int = None, freeze_backbone: bool = False):
        super().__init__(
            model_name=model_name or config.EFFICIENTNET_MODEL_NAME,
            pretrained=pretrained, out_dim=out_dim, freeze_backbone=freeze_backbone,
        )


if __name__ == "__main__":
    model = EfficientNetBackbone(pretrained=False)
    dummy = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    out = model(dummy)
    print("EfficientNet backbone output shape:", out.shape)
    assert out.shape == (2, config.DEEP_FEATURE_DIM)
    print("OK")

"""
backbone/mobilenet.py
------------------------
MobileNetV3-Large deep-feature backbone -- a third backbone option
alongside ViT-B/16 (backbone/vit.py) and EfficientNet-B3
(backbone/efficientnet.py). Thin wrapper around
backbone/cnn_backbone.py::CNNBackbone.

Included specifically so this project's backbone choice can be settled
empirically rather than by architectural argument alone: much of the
skin-cancer literature (e.g. Lilhore et al. 2024, Sci Rep 14:4299) reports
strong results with MobileNet-V3, and this project's own topology-guided
pipeline can now be benchmarked against it directly, on the SAME
leak-free, lesion-grouped train/val/test split (see README.md's "Backbone
comparison" section).

MobileNetV3 was designed via neural architecture search specifically to
minimize on-device inference LATENCY (mobile/edge deployment), not to
maximize accuracy under a generous compute budget -- unlike EfficientNet,
which optimizes accuracy-per-FLOP without a mobile-latency constraint. On
a workstation GPU (Sec. 10: RTX A4500), that mobile-oriented design
tends to cost a small but consistent accuracy gap versus EfficientNet at
a comparable parameter count. Still worth measuring directly, though --
hence this option.
"""

import torch

import config
from backbone.cnn_backbone import CNNBackbone


class MobileNetBackbone(CNNBackbone):
    def __init__(self, model_name: str = None, pretrained: bool = True,
                 out_dim: int = None, freeze_backbone: bool = False):
        super().__init__(
            model_name=model_name or config.MOBILENET_MODEL_NAME,
            pretrained=pretrained, out_dim=out_dim, freeze_backbone=freeze_backbone,
        )


if __name__ == "__main__":
    model = MobileNetBackbone(pretrained=False)
    dummy = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    out = model(dummy)
    print("MobileNetV3 backbone output shape:", out.shape)
    assert out.shape == (2, config.DEEP_FEATURE_DIM)
    print("OK")

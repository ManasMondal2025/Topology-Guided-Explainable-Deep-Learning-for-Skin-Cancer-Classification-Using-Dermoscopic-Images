"""
models/hybrid_model.py
------------------------
Wires together the full architecture described in Sec. 5 of the proposal:

    Dermoscopic Image -> [ViT-B/16, EfficientNet-B3, or MobileNetV3] -> Deep Features
    Lesion Mask (U-Net) -> [Persistent Homology]        -> Topological Features
    (Deep Features, Topological Features) -> Fusion Module -> MLP -> Benign/Malignant
"""

import torch
import torch.nn as nn

import config
from backbone.vit import ViTBackbone
from backbone.efficientnet import EfficientNetBackbone
from backbone.mobilenet import MobileNetBackbone
from fusion.gated_fusion import GatedFusion
from fusion.attention_fusion import AttentionFusion


def build_backbone(name: str = None, pretrained: bool = True):
    name = name or config.BACKBONE_NAME
    if name == "vit_b16":
        return ViTBackbone(pretrained=pretrained)
    elif name == "efficientnet_b3":
        return EfficientNetBackbone(pretrained=pretrained)
    elif name == "mobilenet_v3":
        return MobileNetBackbone(pretrained=pretrained)
    raise ValueError(f"Unknown backbone name: {name}")


def build_fusion(fusion_type: str = None, topo_dim: int = None):
    fusion_type = fusion_type or config.FUSION_TYPE
    topo_dim = topo_dim or config.TOPO_FEATURE_DIM
    if fusion_type == "gated":
        return GatedFusion(topo_dim=topo_dim)
    elif fusion_type == "attention":
        return AttentionFusion(topo_dim=topo_dim)
    raise ValueError(f"Unknown fusion type: {fusion_type}")


class HybridSkinCancerModel(nn.Module):
    """
    Full topology-guided, explainable hybrid model.

    forward(image, topo_features) -> logits[B, num_classes]

    `image` is the preprocessed dermoscopic image tensor fed to the deep
    backbone; `topo_features` is the fixed-length vector produced by
    topology/extract.py::extract_topological_features on the corresponding
    U-Net lesion mask.
    """

    def __init__(self, backbone_name: str = None, fusion_type: str = None,
                 topo_dim: int = None, num_classes: int = None,
                 pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        num_classes = num_classes or config.NUM_CLASSES
        topo_dim = topo_dim or config.TOPO_FEATURE_DIM

        self.backbone = build_backbone(backbone_name, pretrained=pretrained)
        self.fusion = build_fusion(fusion_type, topo_dim=topo_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.fusion.out_dim, self.fusion.out_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.fusion.out_dim // 2, num_classes),
        )

    def forward(self, image: torch.Tensor, topo_features: torch.Tensor) -> torch.Tensor:
        deep_features = self.backbone(image)
        fused = self.fusion(deep_features, topo_features)
        logits = self.classifier(fused)
        return logits

    def forward_deep_features(self, image: torch.Tensor) -> torch.Tensor:
        """Exposed separately so Grad-CAM can hook the backbone in isolation."""
        return self.backbone(image)


if __name__ == "__main__":
    model = HybridSkinCancerModel(pretrained=False)
    dummy_image = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    dummy_topo = torch.randn(2, config.TOPO_FEATURE_DIM)
    logits = model(dummy_image, dummy_topo)
    print("HybridSkinCancerModel output shape:", logits.shape)
    assert logits.shape == (2, config.NUM_CLASSES)
    from utils.common import count_parameters
    print(f"Trainable parameters: {count_parameters(model):,}")
    print("OK")

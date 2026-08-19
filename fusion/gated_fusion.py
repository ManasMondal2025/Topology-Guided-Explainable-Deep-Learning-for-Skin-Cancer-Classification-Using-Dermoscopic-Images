"""
fusion/gated_fusion.py
------------------------
Gated fusion of deep (ViT/EfficientNet/MobileNetV3) features and topological
(Persistent Homology) features (Sec. 6, Phase 6 of the proposal).

Both feature streams are first projected into a shared hidden space, then
a learned sigmoid gate decides, per-dimension, how much to trust the deep
stream vs. the topological stream before the two are combined. This lets
the network down-weight topological features on lesions where the U-Net
mask is noisy, and vice versa.
"""

import torch
import torch.nn as nn

import config


class GatedFusion(nn.Module):
    def __init__(self, deep_dim: int = None, topo_dim: int = None, hidden_dim: int = None):
        super().__init__()
        deep_dim = deep_dim or config.DEEP_FEATURE_DIM
        topo_dim = topo_dim or config.TOPO_FEATURE_DIM
        hidden_dim = hidden_dim or config.FUSION_HIDDEN_DIM

        self.deep_proj = nn.Sequential(nn.Linear(deep_dim, hidden_dim), nn.ReLU(inplace=True))
        self.topo_proj = nn.Sequential(nn.Linear(topo_dim, hidden_dim), nn.ReLU(inplace=True))

        # gate is computed from the concatenation of both projected streams
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.out_dim = hidden_dim

    def forward(self, deep_features: torch.Tensor, topo_features: torch.Tensor) -> torch.Tensor:
        """
        deep_features: Tensor[B, deep_dim]
        topo_features: Tensor[B, topo_dim]
        Returns: Tensor[B, hidden_dim]
        """
        deep_h = self.deep_proj(deep_features)
        topo_h = self.topo_proj(topo_features)

        gate = self.gate(torch.cat([deep_h, topo_h], dim=1))
        fused = gate * deep_h + (1 - gate) * topo_h
        return fused


if __name__ == "__main__":
    fusion = GatedFusion()
    deep_feat = torch.randn(4, config.DEEP_FEATURE_DIM)
    topo_feat = torch.randn(4, config.TOPO_FEATURE_DIM)
    out = fusion(deep_feat, topo_feat)
    print("GatedFusion output shape:", out.shape)
    assert out.shape == (4, config.FUSION_HIDDEN_DIM)
    print("OK")

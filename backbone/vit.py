"""
backbone/vit.py
-----------------
Vision Transformer (ViT-B/16) deep feature extractor (Sec. 6, Phase 5).
Pretrained on ImageNet via `timm` and fine-tuned end-to-end on HAM10000.

Reference: Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers
for Image Recognition at Scale" (https://arxiv.org/abs/2010.11929).
The proposal also references arxiv.org/abs/1811.11314 (BAM: Bottleneck
Attention Module) as background reading on attention-based feature
refinement, which motivated the optional attention-fusion module in
fusion/attention_fusion.py.
"""

import torch
import torch.nn as nn
import timm

import config


class ViTBackbone(nn.Module):
    """
    Wraps a timm ViT-B/16 model and exposes a `config.DEEP_FEATURE_DIM`
    (default 768) pooled embedding per image, matching the native ViT-B/16
    hidden size so no extra projection is needed by default.
    """

    def __init__(self, model_name: str = None, pretrained: bool = True,
                 out_dim: int = None, freeze_backbone: bool = False):
        super().__init__()
        model_name = model_name or config.VIT_MODEL_NAME
        out_dim = out_dim or config.DEEP_FEATURE_DIM

        # num_classes=0 -> timm returns the pooled feature embedding directly
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        native_dim = self.vit.num_features

        self.needs_projection = native_dim != out_dim
        if self.needs_projection:
            self.projection = nn.Linear(native_dim, out_dim)
        else:
            self.projection = nn.Identity()

        if freeze_backbone:
            for p in self.vit.parameters():
                p.requires_grad = False

        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: Tensor[B, 3, H, W] -> Tensor[B, out_dim]"""
        features = self.vit(x)  # [B, native_dim]
        return self.projection(features)

    def get_last_attention_layer(self):
        """Returns the last transformer block's normalization layer — used
        as the target layer for Grad-CAM on ViTs (see explainability/gradcam.py)."""
        return self.vit.blocks[-1].norm1

    def unfreeze_last_n_blocks(self, n: int) -> None:
        """
        Freezes the ENTIRE backbone, then unfreezes only the last `n`
        transformer blocks (plus the final LayerNorm, which is small and
        directly consumes those blocks' output). This is the middle ground
        between `freeze_backbone=True` (0 trainable backbone params -- no
        domain adaptation at all) and full fine-tuning (all 12 blocks
        trainable -- highest overfitting risk on a small dataset).

        n=0 is equivalent to freeze_backbone=True. n >= 12 (all blocks)
        is equivalent to full fine-tuning.
        """
        n = max(0, min(n, len(self.vit.blocks)))
        for p in self.vit.parameters():
            p.requires_grad = False
        if n > 0:
            for block in self.vit.blocks[-n:]:
                for p in block.parameters():
                    p.requires_grad = True
            if hasattr(self.vit, "norm"):
                for p in self.vit.norm.parameters():
                    p.requires_grad = True

    def trainable_backbone_summary(self) -> str:
        total = sum(p.numel() for p in self.vit.parameters())
        trainable = sum(p.numel() for p in self.vit.parameters() if p.requires_grad)
        pct = 100.0 * trainable / total if total > 0 else 0.0
        return f"{trainable:,}/{total:,} backbone params trainable ({pct:.1f}%)"


if __name__ == "__main__":
    model = ViTBackbone(pretrained=False)  # pretrained=False for a fast offline smoke test
    dummy = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    out = model(dummy)
    print("ViT backbone output shape:", out.shape)
    assert out.shape == (2, config.DEEP_FEATURE_DIM)

    model.unfreeze_last_n_blocks(2)
    print(model.trainable_backbone_summary())
    out2 = model(dummy)
    assert out2.shape == (2, config.DEEP_FEATURE_DIM)
    print("OK")

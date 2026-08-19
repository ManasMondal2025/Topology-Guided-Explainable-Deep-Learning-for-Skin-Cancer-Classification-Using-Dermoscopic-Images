"""
fusion/attention_fusion.py
-----------------------------
Attention-based fusion of deep (ViT/EfficientNet/MobileNetV3) features and topological
(Persistent Homology) features (Sec. 6, Phase 6 — alternative to
gated_fusion.py).

Both streams are projected to a shared hidden dimension and treated as a
2-token sequence; a multi-head self-attention layer lets each stream
attend to the other before the tokens are pooled into a single fused
vector. This is a lightweight analogue of the attention-refinement idea
referenced in the proposal's background reading (BAM: Bottleneck
Attention Module, arxiv.org/abs/1811.11314).
"""

import torch
import torch.nn as nn

import config


class AttentionFusion(nn.Module):
    def __init__(self, deep_dim: int = None, topo_dim: int = None,
                 hidden_dim: int = None, num_heads: int = 4):
        super().__init__()
        deep_dim = deep_dim or config.DEEP_FEATURE_DIM
        topo_dim = topo_dim or config.TOPO_FEATURE_DIM
        hidden_dim = hidden_dim or config.FUSION_HIDDEN_DIM

        self.deep_proj = nn.Linear(deep_dim, hidden_dim)
        self.topo_proj = nn.Linear(topo_dim, hidden_dim)

        # learned type embeddings so the attention block can distinguish
        # the "deep" token from the "topological" token
        self.token_type_embedding = nn.Parameter(torch.zeros(2, hidden_dim))
        nn.init.normal_(self.token_type_embedding, std=0.02)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.out_dim = hidden_dim

    def forward(self, deep_features: torch.Tensor, topo_features: torch.Tensor) -> torch.Tensor:
        """
        deep_features: Tensor[B, deep_dim]
        topo_features: Tensor[B, topo_dim]
        Returns: Tensor[B, hidden_dim] (mean-pooled fused representation)
        """
        deep_h = self.deep_proj(deep_features) + self.token_type_embedding[0]
        topo_h = self.topo_proj(topo_features) + self.token_type_embedding[1]

        tokens = torch.stack([deep_h, topo_h], dim=1)  # [B, 2, hidden_dim]
        attn_out, _ = self.attention(tokens, tokens, tokens)
        tokens = self.norm(tokens + attn_out)
        tokens = self.norm2(tokens + self.ffn(tokens))

        fused = tokens.mean(dim=1)  # pool the 2 tokens -> [B, hidden_dim]
        return fused


if __name__ == "__main__":
    fusion = AttentionFusion()
    deep_feat = torch.randn(4, config.DEEP_FEATURE_DIM)
    topo_feat = torch.randn(4, config.TOPO_FEATURE_DIM)
    out = fusion(deep_feat, topo_feat)
    print("AttentionFusion output shape:", out.shape)
    assert out.shape == (4, config.FUSION_HIDDEN_DIM)
    print("OK")

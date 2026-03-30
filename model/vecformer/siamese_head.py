"""
VecFormerSiameseHead — Top-Down Siamese Search via Cross-Attention.

Given a legend query (a small set of line-primitive tokens) and a full
floor-plan (a large set of line-primitive tokens), this module uses
multi-head cross-attention to let the legend *attend into* the plan,
then scores every plan primitive for similarity to the query.

Architecture
------------
    legend_tokens  (Q, D)  ─┐
                             ├─► CrossAttention (Q→K/V) ─► FFN ─► pool ─► (D,)
    plan_tokens    (N, D)  ─┘                                           │
                                                                        ▼
    plan_tokens    (N, D)  ───────────────────────────────► dot-product ─► sigmoid ─► (N,) scores

The legend tokens act as *queries*, the plan tokens as *keys* and
*values*.  After cross-attention + feed-forward refinement, the legend
representation is mean-pooled into a single vector and dot-producted
against every plan primitive to produce a per-primitive similarity score.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseCrossAttentionBlock(nn.Module):
    """One block of: cross-attn → add-norm → FFN → add-norm."""

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,   # (B, Q, D)  legend tokens
        kv: torch.Tensor,      # (B, N, D)  plan tokens
    ) -> torch.Tensor:
        attn_out, _ = self.cross_attn(
            query=query, key=kv, value=kv,
        )
        query = self.norm1(query + self.dropout(attn_out))
        query = self.norm2(query + self.dropout(self.ffn(query)))
        return query


class VecFormerSiameseHead(nn.Module):
    """Siamese search head that scores every plan primitive against a legend query.

    Parameters
    ----------
    input_dim : int
        Dimension of backbone primitive features (default 64 for VecFormer).
    embed_dim : int
        Internal embedding dimension for the cross-attention blocks.
    n_heads : int
        Number of attention heads.
    n_blocks : int
        Number of stacked cross-attention blocks.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 64,
        embed_dim: int = 256,
        n_heads: int = 8,
        n_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Project backbone features (64-d) up to the cross-attention dim.
        self.legend_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.plan_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # Cross-attention blocks: legend queries attend to plan keys/values.
        self.blocks = nn.ModuleList([
            SiameseCrossAttentionBlock(embed_dim, n_heads, dropout)
            for _ in range(n_blocks)
        ])

        # Scoring projection: maps plan tokens to a space where we can
        # dot-product against the pooled legend representation.
        self.score_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        legend_tokens: torch.Tensor,
        plan_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        legend_tokens : (Q, input_dim)
            Per-primitive features from the legend crop backbone pass.
        plan_tokens : (N, input_dim)
            Per-primitive features from the full floor-plan backbone pass.

        Returns
        -------
        scores : (N,)
            Similarity score in [0, 1] for every plan primitive.
        """
        # Add batch dimension for nn.MultiheadAttention (batch_first=True).
        legend = self.legend_proj(legend_tokens).unsqueeze(0)  # (1, Q, D)
        plan = self.plan_proj(plan_tokens).unsqueeze(0)        # (1, N, D)

        # Cross-attention: legend queries attend to plan keys/values.
        for block in self.blocks:
            legend = block(query=legend, kv=plan)
        # legend is now (1, Q, D) — each legend token has "looked at" the plan.

        # Pool legend into a single query vector.
        legend_vec = legend.squeeze(0).mean(dim=0)  # (D,)

        # Score every plan primitive.
        plan_scored = self.score_proj(plan.squeeze(0))  # (N, D)
        logits = plan_scored @ legend_vec                # (N,)
        scores = torch.sigmoid(logits)                   # (N,)

        return scores

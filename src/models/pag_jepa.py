"""
PAG-JEPA: Platonic Attribute Grounding Joint-Embedding Predictive Architecture
Non-autoregressive model that maps video representations directly into the
orthogonal Platonic attribute space, eliminating next-token hallucination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class TemporalPooling(nn.Module):
    """
    Lightweight 2-layer Transformer encoder that pools a sequence of frame embeddings
    (e.g., 8 frames from DINOv2-small, d=384) into a single spatio-temporal video token.
    Total parameters: ~1.5M (VRAM: < 50 MB).
    """
    def __init__(self, embed_dim: int = 384, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d) sequence of frame embeddings
        returns: (B, d) aggregated video embedding
        """
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) # (B, T+1, d)
        out = self.transformer(x)
        return self.layer_norm(out[:, 0]) # CLS token output (B, d)


class OrthogonalProjector(nn.Module):
    """
    Orthogonal Procrustes Projection Head.
    Parameterizes the linear mapping g_theta(z_v) = z_v * R,
    enforcing the Stiefel Manifold constraint R^T R = I_d.
    """
    def __init__(self, embed_dim: int = 384):
        super().__init__()
        self.embed_dim = embed_dim
        # Initialize with random orthogonal matrix
        Q, _ = torch.linalg.qr(torch.randn(embed_dim, embed_dim))
        self.weight = nn.Parameter(Q)

    def project_orthogonal(self):
        """Projects current weights onto the Stiefel manifold O(d) via SVD (Procrustes step)."""
        with torch.no_grad():
            U, _, Vh = torch.linalg.svd(self.weight.data)
            self.weight.data = U @ Vh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize and rotate
        x_norm = F.normalize(x, p=2, dim=-1)
        return x_norm @ self.weight


class PAG_JEPA(nn.Module):
    """
    Main Platonic Attribute Grounding JEPA Model.
    Bypasses language decoders, operating directly in the shared Platonic latent space.
    """
    def __init__(
        self,
        embed_dim: int = 384,
        temporal_layers: int = 2,
        temporal_heads: int = 4,
        grounding_threshold: float = 0.62
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.grounding_threshold = grounding_threshold
        
        # Temporal aggregator over frame embeddings
        self.temporal_pool = TemporalPooling(embed_dim, temporal_layers, temporal_heads)
        
        # Orthogonal Procrustes rotation head
        self.projector = OrthogonalProjector(embed_dim)

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        """
        frame_features: (B, T, d) from frozen visual encoder (DINOv2)
        returns: (B, d) projected Platonic video embedding z_v'
        """
        z_v = self.temporal_pool(frame_features)
        z_v_proj = self.projector(z_v)
        return F.normalize(z_v_proj, p=2, dim=-1)

    def predict_attributes(
        self,
        z_v_proj: torch.Tensor,
        codebook_embeddings: torch.Tensor,
        codebook_names: list
    ) -> list:
        """
        Non-autoregressive inference via Platonic codebook matching.
        If maximum cosine similarity < grounding_threshold, marks attribute as 'Not Observable'.
        """
        # Cosine similarity: (B, d) @ (K, d)^T -> (B, K)
        sims = z_v_proj @ F.normalize(codebook_embeddings, p=2, dim=-1).T
        max_sims, indices = torch.max(sims, dim=-1)

        results = []
        for sim, idx in zip(max_sims, indices):
            if sim.item() < self.grounding_threshold:
                results.append({
                    "attribute": "Not Observable",
                    "confidence": sim.item(),
                    "grounded": False
                })
            else:
                results.append({
                    "attribute": codebook_names[idx.item()],
                    "confidence": sim.item(),
                    "grounded": True
                })
        return results

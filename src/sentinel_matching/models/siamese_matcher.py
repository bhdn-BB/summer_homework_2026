"""Timm-based Siamese architecture for seasonal image matching."""

from __future__ import annotations

import timm
import torch
from torch import nn


class GeMPool(nn.Module):
    """Generalized mean pooling for sharper regional representations."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return torch.nn.functional.adaptive_avg_pool2d(x.clamp_min(self.eps).pow(self.p), 1).pow(1.0 / self.p).flatten(1)


class TimmSiameseMatcher(nn.Module):
    """Multi-scale timm encoder with a pairwise relation head."""

    def __init__(self, backbone: str = "convnext_tiny", embedding_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=pretrained, features_only=True)
        channels = self.encoder.feature_info.channels()
        self.poolers = nn.ModuleList([GeMPool() for _ in channels])
        self.scale_weights = nn.Parameter(torch.ones(len(channels)))
        self.projections = nn.ModuleList([nn.Linear(channel, embedding_dim) for channel in channels])
        self.embedding = nn.Sequential(nn.LayerNorm(embedding_dim), nn.GELU(), nn.Dropout(0.2))
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(embedding_dim, 1),
        )

    def encode(self, image):
        features = self.encoder(image)
        weights = torch.softmax(self.scale_weights, dim=0)
        pooled = torch.stack([projection(pool(feature)) for feature, pool, projection in zip(features, self.poolers, self.projections)], dim=1)
        representation = (pooled * weights.view(1, -1, 1)).sum(dim=1)
        return self.embedding(representation)

    def forward(self, image_left, image_right):
        left = self.encode(image_left)
        right = self.encode(image_right)
        return self.forward_embeddings(left, right)

    def forward_embeddings(self, left, right):
        """Return the binary matching logit for two encoded images."""
        relation = torch.cat([torch.abs(left - right), left * right, (left + right) / 2.0], dim=1)
        return self.classifier(relation).squeeze(1)

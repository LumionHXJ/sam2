# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Prototype character encoder using ViT-Tiny backbone."""

import math
from typing import Optional

import timm
import torch
from torch import nn


def get_1d_sine_pe(embed_dim: int, num_tokens: int, temperature: float = 10000.0) -> torch.Tensor:
    """Generate 1D sine positional encoding.

    Learnable positional encoding from Attention Is All You Need, adapted for 1D sequences.

    Args:
        embed_dim: Dimension of the embedding.
        num_tokens: Number of tokens to encode.
        temperature: Temperature for the frequency scaling.

    Returns:
        torch.Tensor: Positional encoding with shape [num_tokens, embed_dim].
    """
    half_dim = embed_dim // 2
    freqs = torch.arange(half_dim, dtype=torch.float32)
    freqs = 1.0 / (temperature ** (freqs / half_dim))

    positions = torch.arange(num_tokens, dtype=torch.float32)
    freqs = positions[:, None] * freqs[None, :]

    pe = torch.zeros(num_tokens, embed_dim, dtype=torch.float32)
    pe[:, 0::2] = freqs.sin()
    pe[:, 1::2] = freqs.cos()

    return pe


class PrototypeEncoder(nn.Module):
    """Encodes prototype character images into spatial embedding tokens.

    Uses a Vision Transformer (ViT-Tiny) pretrained on ImageNet-21K
    to encode prototype character images into spatial token embeddings that
    preserve the spatial structure of the character. The output consists of
    196 tokens (14x14) that can be used as additional prompts in SAM 2.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        output_dim: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        model_name: str = "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    ):
        """
        Args:
            embed_dim: Dimension of ViT backbone output.
            output_dim: Dimension of projected output embedding per token.
            pretrained: Whether to load pretrained weights.
            freeze_backbone: Whether to freeze backbone parameters.
            model_name: Name of the timm model to use.
        """
        super().__init__()

        # Create ViT-Tiny backbone using timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )

        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Token-wise projection to SAM embedding dimension (256)
        # Replaces global pooling with per-token projection
        self.token_proj = nn.Linear(embed_dim, output_dim)

        # Number of spatial tokens for 224x224 input with patch_size=16
        self.num_patches = 196  # 14x14

        # Fixed sine positional encoding for spatial tokens
        pos_embed = get_1d_sine_pe(output_dim, self.num_patches)
        self.register_buffer("pos_embed", pos_embed)  # [196, 256]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Args:
            x: Input images with shape [B, 3, H, W].

        Returns:
            torch.Tensor: Spatial token embeddings with shape [B, num_patches, output_dim].
                         For patch_size=16 and H=W=224, this is [B, 196, 256].
        """
        # Extract features using backbone
        # ViT outputs [B, num_patches + 1, embed_dim] where +1 is the CLS token
        # For 224x224 with patch_size=16: 196 patch tokens + 1 CLS token = 197 total
        features = self.backbone.forward_features(x)  # [B, 197, 192]

        # Skip CLS token (first token) to get only spatial patch tokens
        patch_tokens = features[:, 1:, :]  # [B, 196, 192]

        # Per-token projection to output dimension
        spatial_features = self.token_proj(patch_tokens)  # [B, 196, 256]

        # Add positional encoding
        spatial_features = spatial_features + self.pos_embed.unsqueeze(0)  # [B, 196, 256]

        return spatial_features

    def get_num_layers(self) -> int:
        """Return the number of layers in the backbone for learning rate decay."""
        # timm ViT models have a 'blocks' attribute
        if hasattr(self.backbone, 'blocks'):
            return len(self.backbone.blocks)
        return 12  # Default to 12 for ViT-Tiny

    def get_layer_id(self, layer_name: str) -> int:
        """Return layer ID for learning rate decay.

        Maps parameter names to layer IDs:
        - 'backbone.*': backbone parameters
        - 'token_proj.*': token projection layer parameters
        """
        num_layers = self.get_num_layers()

        if layer_name.find("token_proj") != -1:
            return num_layers + 1  # Token projection layer gets the highest layer ID
        elif layer_name.find("backbone.pos_embed") != -1 or layer_name.find("backbone.pos_drop") != -1:
            return 0  # Positional encoding gets layer 0
        elif layer_name.find("backbone.patch_embed") != -1:
            return 0  # Patch embedding gets layer 0
        elif layer_name.find("backbone.blocks") != -1:
            # Parse block index: "backbone.blocks.0.norm1.weight" -> 0
            try:
                return int(layer_name.split("backbone.blocks")[1].split(".")[1]) + 1
            except:
                return num_layers + 1
        else:
            # Other layers (norm, etc.) get the highest layer ID
            return num_layers + 1

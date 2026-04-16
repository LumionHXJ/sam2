# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Prototype character encoder using a configurable timm backbone."""

from typing import Optional, Tuple, Union

import timm
import torch
from torch import nn


class PrototypeEncoder(nn.Module):
    """Encodes prototype character images into a single prompt embedding."""

    def __init__(
        self,
        embed_dim: Optional[int] = None,
        output_dim: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        model_name: str = "vit_small_patch14_dinov2",
        checkpoint_path: Optional[str] = None,
        img_size: Union[int, Tuple[int, int]] = 224,
    ):
        """
        Args:
            embed_dim: Optional compatibility override for backbone output dimension.
                If not provided, the dimension is inferred from the backbone output.
            output_dim: Dimension of the projected prototype prompt embedding.
            pretrained: Whether to load pretrained weights.
            freeze_backbone: Whether to freeze backbone parameters.
            model_name: Name of the timm model to use.
            checkpoint_path: Optional checkpoint to pass through to timm.create_model.
            img_size: Input image size for the timm model, as int or (H, W).
        """
        super().__init__()

        if isinstance(img_size, int):
            normalized_img_size = (img_size, img_size)
        else:
            if len(img_size) != 2:
                raise ValueError(
                    f"img_size must be an int or a tuple of length 2, got {img_size!r}"
                )
            normalized_img_size = tuple(int(dim) for dim in img_size)

        create_model_kwargs = {
            "pretrained": pretrained,
            "num_classes": 0,
            "img_size": normalized_img_size,
        }
        if checkpoint_path is not None:
            create_model_kwargs["checkpoint_path"] = checkpoint_path

        self.backbone = timm.create_model(model_name, **create_model_kwargs)

        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.img_size = normalized_img_size
        self.output_dim = output_dim
        self.embed_dim = embed_dim

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.prompt_proj: Optional[nn.Sequential] = None
        inferred_dim = self._infer_backbone_output_dim()
        if inferred_dim is not None:
            self.prompt_proj = self._build_prompt_proj(inferred_dim, output_dim)
            self.embed_dim = inferred_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Args:
            x: Input images with shape [B, 3, H, W].

        Returns:
            torch.Tensor: Prototype prompt embeddings with shape [B, 1, output_dim].
        """
        features = self.backbone(x)
        if not isinstance(features, torch.Tensor):
            raise TypeError(
                "PrototypeEncoder requires `model(x)` to return a torch.Tensor "
                f"with shape [B, D], got {type(features).__name__}."
            )
        if features.ndim != 2:
            raise ValueError(
                "PrototypeEncoder requires `model(x)` to return shape [B, D], "
                f"got tensor shape {tuple(features.shape)}."
            )

        if self.prompt_proj is None:
            self.prompt_proj = self._build_prompt_proj(
                features.shape[-1], self.output_dim
            ).to(
                device=features.device
            )
            self.embed_dim = features.shape[-1]
        elif features.shape[-1] != self.prompt_proj[1].in_features:
            raise ValueError(
                "PrototypeEncoder backbone output dimension does not match the "
                f"projection layer. Expected {self.prompt_proj[1].in_features}, "
                f"got {features.shape[-1]}."
            )

        prototype_prompt = self.prompt_proj(features)
        return prototype_prompt.unsqueeze(1)

    def _infer_backbone_output_dim(self) -> Optional[int]:
        """Best-effort inference of the backbone feature dimension."""
        if self.embed_dim is not None:
            return self.embed_dim
        for attr in ("num_features", "embed_dim"):
            value = getattr(self.backbone, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        return None

    @staticmethod
    def _build_prompt_proj(input_dim: int, output_dim: int) -> nn.Sequential:
        """Build the prototype-to-prompt adapter."""
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

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
        - 'prompt_proj.*': prompt projection layer parameters
        """
        num_layers = self.get_num_layers()

        if layer_name.find("prompt_proj") != -1:
            return num_layers + 1  # Prompt projection layer gets the highest layer ID
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

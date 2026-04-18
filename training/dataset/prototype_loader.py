# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Prototype character image loader for category-guided training."""

import os
from typing import List, Optional, Tuple, Union

import torch
from PIL import Image as PILImage
from torchvision import transforms


class PrototypeLoader:
    """Loads prototype character images for category-guided training.

    This loader retrieves prototype character images (字头) from the database
    and preprocesses them for use as additional prompts in SAM 2.
    """

    def __init__(
        self,
        prototype_root: str,
        img_size: Union[int, Tuple[int, int]] = 224,
        missing_as_zero: bool = True,
    ):
        """
        Args:
            prototype_root: Path to directory containing prototype character images.
            img_size: Size to resize prototype images to (default: 224).
            missing_as_zero: If True, return zero tensor for missing prototypes.
        """
        self.prototype_root = prototype_root
        self.img_size = self._normalize_img_size(img_size)
        height, width = self.img_size
        self.missing_as_zero = missing_as_zero
        self.missing_count = 0
        self.total_count = 0

        # Standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @staticmethod
    def _normalize_img_size(img_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        if isinstance(img_size, int):
            return (img_size, img_size)
        if len(img_size) != 2:
            raise ValueError(
                f"img_size must be an int or a tuple of length 2, got {img_size!r}"
            )
        return tuple(int(dim) for dim in img_size)

    def _load_single_prototype(self, label: Optional[str]) -> torch.Tensor:
        """Load a single prototype image."""
        self.total_count += 1
        assert label is not None, "Label must be provided for prototype loading"
        for ext in ['.png', '.jpg', '.jpeg']:
            img_path = os.path.join(self.prototype_root, f"{label}{ext}")
            if os.path.exists(img_path):
                img = PILImage.open(img_path).convert('RGB')
                return self.transform(img)

    def load_prototypes(self, labels: List[Optional[str]]) -> torch.Tensor:
        """
        Load prototype images for given labels.

        Args:
            labels: List of label strings (or None for missing labels).

        Returns:
            torch.Tensor: Batch of prototype images with shape [B, 3, H, W].
        """
        batch = []
        for label in labels:
            prototype = self._load_single_prototype(label)
            batch.append(prototype)
        return torch.stack(batch)

    def get_missing_stats(self) -> dict:
        """Return statistics about missing prototype images."""
        if self.total_count == 0:
            return {"total": 0, "missing": 0, "missing_rate": 0.0}
        return {
            "total": self.total_count,
            "missing": self.missing_count,
            "missing_rate": self.missing_count / self.total_count,
        }

    def reset_stats(self):
        """Reset missing statistics counters."""
        self.missing_count = 0
        self.total_count = 0

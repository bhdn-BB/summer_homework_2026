"""Dataset classes for supervised seasonal image matching."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from match_images import read_sentinel_scene


class SentinelPairDataset(Dataset):
    """Load SAFE scene pairs and return two tensors plus a binary match label."""

    def __init__(self, manifest_path: str, split: str, image_size: int = 224, bands=("B04", "B03", "B02"), train: bool = False):
        with Path(manifest_path).open(encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.rows = [row for row in rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No rows found for split={split} in {manifest_path}")
        self.bands = tuple(bands)
        self.transform = self._build_transform(image_size, train)

    @staticmethod
    def _build_transform(image_size: int, train: bool):
        operations = [transforms.ToPILImage(), transforms.Resize((image_size, image_size))]
        if train:
            operations.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ])
        operations.extend([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
        return transforms.Compose(operations)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        left = self.transform(read_sentinel_scene(row["left"], self.bands))
        right = self.transform(read_sentinel_scene(row["right"], self.bands))
        return {"image_left": left, "image_right": right, "label": torch.tensor(float(row["label"]))}

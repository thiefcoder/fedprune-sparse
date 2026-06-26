"""
Compact CNN backbone for MNIST experiments.

The model is deliberately sequential and easy to trace, which makes structured
channel pruning deterministic and transparent. Because the architecture has no
skip connections, pruned channels can be propagated directly into the next layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Architecture: Conv1 -> ReLU -> Pool -> Conv2 -> ReLU -> Pool -> Flatten -> FC1 -> ReLU -> FC2.

    For MNIST inputs with shape 1x28x28, two MaxPool(2, 2) operations reduce the
    spatial resolution from 28 to 7.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        image_size: int = 28,
        conv1_out: int = 32,
        conv2_out: int = 64,
        fc1_out: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size

        self.conv1 = nn.Conv2d(in_channels, conv1_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.feat_size = image_size // 4
        self._flatten_dim = conv2_out * self.feat_size * self.feat_size

        self.fc1 = nn.Linear(self._flatten_dim, fc1_out)
        self.fc2 = nn.Linear(fc1_out, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def config_dict(self) -> dict:
        """Return the constructor configuration for reproducing the current shape."""
        return dict(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            image_size=self.image_size,
            conv1_out=self.conv1.out_channels,
            conv2_out=self.conv2.out_channels,
            fc1_out=self.fc1.out_features,
        )

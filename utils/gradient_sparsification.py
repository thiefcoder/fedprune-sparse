"""
Delta sparsification utilities for client-to-server communication.

After local training, each client builds a model delta. This module compresses
that delta before transmission using Top-K, random coordinate selection, or a
cost-weighted strategy that accounts for parameter-level communication cost.

Error feedback is included so omitted delta components are accumulated locally
and can be transmitted in later rounds when they become important.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


# Communication sparsification configuration.
@dataclass
class SparsificationConfig:
    method: str = "topk"
    sparsity_ratio: float = 0.99
    use_error_feedback: bool = True
    # Default cost weights for cost-weighted sparsification.
    cwmp_fc_cost: float = 5.0
    cwmp_conv_cost: float = 1.0


# Error feedback buffer for values omitted in previous communication rounds.
class ErrorFeedbackBuffer:
    """
    Accumulate unsent delta components in a local client buffer.

    This compensation is important under aggressive compression, especially for
    sparsity levels above 95%, because dropped coordinates are not permanently
    discarded.
    """

    def __init__(self):
        self._buffer: Dict[str, torch.Tensor] = {}

    def get(self, name: str, like: torch.Tensor) -> torch.Tensor:
        if name not in self._buffer:
            self._buffer[name] = torch.zeros_like(like)
        return self._buffer[name]

    def accumulate(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Add the previous residual to the current delta tensor."""
        residual = self.get(name, tensor)
        combined = residual + tensor
        return combined

    def update_residual(self, name: str, combined: torch.Tensor, mask: torch.Tensor) -> None:
        """Store values that were not transmitted in the current round."""
        self._buffer[name] = combined * (~mask)

    def reset(self) -> None:
        """Clear all stored residuals."""
        self._buffer.clear()


# Base class for all delta sparsification methods.
class BaseSparsifier:
    def __init__(self, config: SparsificationConfig):
        self.config = config
        self.error_feedback = ErrorFeedbackBuffer() if config.use_error_feedback else None

    def _maybe_compensate(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if self.error_feedback is not None:
            return self.error_feedback.accumulate(name, tensor)
        return tensor

    def _maybe_store_residual(self, name: str, combined: torch.Tensor, mask: torch.Tensor) -> None:
        if self.error_feedback is not None:
            self.error_feedback.update_residual(name, combined, mask)

    def sparsify_tensor(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sparsify_state_dict(self, deltas: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Sparsify every tensor in a model-delta state dictionary."""
        return {name: self.sparsify_tensor(name, tensor) for name, tensor in deltas.items()}


# Top-K sparsification keeps the largest absolute delta values.
class TopKSparsifier(BaseSparsifier):
    """Keep the largest coordinates by absolute value."""

    def sparsify_tensor(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        combined = self._maybe_compensate(name, tensor)
        flat = combined.flatten()
        k = max(1, int((1 - self.config.sparsity_ratio) * flat.numel()))
        if k >= flat.numel():
            mask = torch.ones_like(flat, dtype=torch.bool)
        else:
            threshold = torch.topk(flat.abs(), k, sorted=False).values.min()
            mask = flat.abs() >= threshold
        mask = mask.view_as(combined)
        sparse_tensor = combined * mask
        self._maybe_store_residual(name, combined, mask)
        return sparse_tensor


# Random sparsification is primarily a controlled baseline.
class RandomSparsifier(BaseSparsifier):
    """Keep a random subset of coordinates, independent of magnitude."""

    def sparsify_tensor(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        combined = self._maybe_compensate(name, tensor)
        flat = combined.flatten()
        k = max(1, int((1 - self.config.sparsity_ratio) * flat.numel()))
        perm = torch.randperm(flat.numel(), device=flat.device)[:k]
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask[perm] = True
        mask = mask.view_as(combined)
        sparse_tensor = combined * mask
        self._maybe_store_residual(name, combined, mask)
        return sparse_tensor


# Cost-weighted sparsification ranks delta magnitude relative to parameter cost.
class CostWeightedSparsifier(BaseSparsifier):
    """
    Select coordinates by communication value instead of magnitude alone.

    The score is abs(delta) / cost. Parameters with higher estimated
    communication or access cost must therefore carry a larger update to be
    transmitted.
    """

    def __init__(self, config: SparsificationConfig):
        super().__init__(config)
        self._cost_map: Dict[str, torch.Tensor] = {}

    def register_cost(self, name: str, cost: torch.Tensor) -> None:
        """Manually register a cost tensor and bypass automatic cost inference."""
        self._cost_map[name] = cost

    def auto_register_cost_from_model(self, model: nn.Module) -> None:
        """
        Infer default parameter costs from module type.

        Fully connected layers receive a higher default cost than convolutional
        layers. User-provided costs are preserved.
        """
        fc_param_names = set()
        conv_param_names = set()
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                fc_param_names.add(f"{module_name}.weight")
                fc_param_names.add(f"{module_name}.bias")
            elif isinstance(module, nn.Conv2d):
                conv_param_names.add(f"{module_name}.weight")
                conv_param_names.add(f"{module_name}.bias")

        for name, param in model.named_parameters():
            if name in self._cost_map:
                continue
            if name in fc_param_names:
                cost_value = self.config.cwmp_fc_cost
            elif name in conv_param_names:
                cost_value = self.config.cwmp_conv_cost
            else:
                cost_value = 1.0
            self._cost_map[name] = torch.full_like(param, float(cost_value))

    def sparsify_tensor(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        combined = self._maybe_compensate(name, tensor)
        cost = self._cost_map.get(name)
        if cost is None or cost.shape != combined.shape:
            cost = torch.ones_like(combined)
        score = combined.abs() / cost.clamp(min=1e-8)

        flat_score = score.flatten()
        k = max(1, int((1 - self.config.sparsity_ratio) * flat_score.numel()))
        if k >= flat_score.numel():
            mask = torch.ones_like(flat_score, dtype=torch.bool)
        else:
            threshold = torch.topk(flat_score, k, sorted=False).values.min()
            mask = flat_score >= threshold
        mask = mask.view_as(combined)
        sparse_tensor = combined * mask
        self._maybe_store_residual(name, combined, mask)
        return sparse_tensor

    def energy_cost(self, deltas: Dict[str, torch.Tensor]) -> float:
        """Estimate the weighted cost of nonzero transmitted coordinates."""
        total = 0.0
        for name, tensor in deltas.items():
            cost = self._cost_map.get(name)
            if cost is None or cost.shape != tensor.shape:
                cost = torch.ones_like(tensor)
            nonzero_mask = tensor != 0
            total += (cost * nonzero_mask).sum().item()
        return total


# Factory for configured sparsification strategies.
def build_sparsifier(config: SparsificationConfig) -> BaseSparsifier:
    if config.method == "topk":
        return TopKSparsifier(config)
    elif config.method == "random":
        return RandomSparsifier(config)
    elif config.method == "cost_weighted":
        return CostWeightedSparsifier(config)
    else:
        raise ValueError(f"Unknown sparsification method: {config.method}")

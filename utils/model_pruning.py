"""
Model pruning utilities for reducing client-side local training cost.

The module supports unstructured pruning, which masks individual low-magnitude
weights, and structured pruning, which removes complete filters or neurons and
rebuilds a smaller model for efficient local execution.

Pruning is responsible only for local computation reduction. Communication
compression is implemented separately in utils/gradient_sparsification.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# Pruning strategy configuration.
@dataclass
class PruningConfig:
    """Configuration for pruning mode, ratio, and adaptive ratio bounds."""

    mode: str = "unstructured"
    pruning_ratio: float = 0.5
    prunable_layer_types: tuple = (nn.Linear, nn.Conv2d)
    min_ratio: float = 0.1
    max_ratio: float = 0.8
    per_client_ratio: Optional[Dict[str, float]] = field(default=None)


# Locate modules that are eligible for pruning.
def get_prunable_modules(model: nn.Module, layer_types: tuple) -> List[tuple]:
    """Return named modules whose types are eligible for pruning."""
    return [(name, module) for name, module in model.named_modules() if isinstance(module, layer_types)]


# Unstructured pruning masks individual low-magnitude weights.
class UnstructuredPruner:
    """
    Magnitude-based unstructured pruning.

    Low-magnitude weights are zeroed while tensor shapes remain unchanged.
    Persistent masks prevent pruned weights from being reactivated by optimizer
    updates during local training.
    """

    def __init__(self, model: nn.Module, config: PruningConfig):
        self.model = model
        self.config = config
        self.masks: Dict[str, torch.Tensor] = {}
        self._init_masks()

    def _init_masks(self) -> None:
        for name, module in get_prunable_modules(self.model, self.config.prunable_layer_types):
            self.masks[name] = torch.ones_like(module.weight, dtype=torch.bool)

    @torch.no_grad()
    def prune(self, ratio: Optional[float] = None) -> None:
        """Zero low-magnitude weights according to the requested pruning ratio."""
        ratio = ratio if ratio is not None else self.config.pruning_ratio
        for name, module in get_prunable_modules(self.model, self.config.prunable_layer_types):
            weight = module.weight
            importance = weight.abs().flatten()
            k = int(ratio * importance.numel())
            if k <= 0:
                continue
            threshold = torch.kthvalue(importance, k).values
            new_mask = weight.abs() > threshold
            self.masks[name] &= new_mask
            weight.mul_(self.masks[name])

    @torch.no_grad()
    def apply_masks(self) -> None:
        """Re-apply masks so pruned weights and gradients remain zero."""
        for name, module in get_prunable_modules(self.model, self.config.prunable_layer_types):
            module.weight.mul_(self.masks[name])
            if module.weight.grad is not None:
                module.weight.grad.mul_(self.masks[name])

    def sparsity(self) -> float:
        """Return the fraction of pruned weights across eligible layers."""
        total, zeros = 0, 0
        for mask in self.masks.values():
            total += mask.numel()
            zeros += (~mask).sum().item()
        return zeros / total if total > 0 else 0.0

    def register_step_hook(self, optimizer: torch.optim.Optimizer) -> None:
        """Wrap optimizer.step so masks are enforced after every update."""
        original_step = optimizer.step

        def step_with_mask(*args, **kwargs):
            result = original_step(*args, **kwargs)
            self.apply_masks()
            return result

        optimizer.step = step_with_mask


# Structured pruning removes full filters or neurons and rebuilds the model.
class StructuredPruner:
    """
    L1-norm structured pruning for filters and hidden neurons.

    For the current SimpleCNN architecture, the pruner creates a truly smaller
    model by propagating retained channel indices through downstream layers.
    This produces actual local memory/FLOP reduction without sparse backends.
    """

    def __init__(self, model: nn.Module, config: PruningConfig):
        self.model = model
        self.config = config

    @staticmethod
    def _filter_importance(weight: torch.Tensor) -> torch.Tensor:
        """Measure filter or neuron importance by the L1 norm of its weights."""
        return weight.abs().flatten(start_dim=1).sum(dim=1)

    @torch.no_grad()
    def compute_keep_indices(self, ratio: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """Compute retained filter or neuron indices for each prunable layer."""
        ratio = ratio if ratio is not None else self.config.pruning_ratio
        keep_indices: Dict[str, torch.Tensor] = {}
        for name, module in get_prunable_modules(self.model, self.config.prunable_layer_types):
            importance = self._filter_importance(module.weight)
            n_keep = max(1, int(round((1 - ratio) * importance.numel())))
            n_keep = min(n_keep, importance.numel())
            _, idx = torch.topk(importance, n_keep)
            keep_indices[name] = idx.sort().values
        return keep_indices

    @torch.no_grad()
    def rebuild_pruned_model(self, keep_indices: Dict[str, torch.Tensor]) -> nn.Module:
        """
        Build a smaller model from retained indices.

        Outputs from each retained layer are aligned with inputs of the next
        layer. The original model is left unchanged.
        """
        from models.cnn import SimpleCNN

        if not isinstance(self.model, SimpleCNN):
            raise TypeError(
                "rebuild_pruned_model currently supports only SimpleCNN. "
                "Extend channel-alignment logic for additional architectures."
            )

        old: SimpleCNN = self.model
        conv1_keep = keep_indices["conv1"]
        conv2_keep = keep_indices["conv2"]
        fc1_keep = keep_indices["fc1"]
        fc2_out = old.fc2.out_features

        new_model = SimpleCNN(
            in_channels=old.in_channels,
            num_classes=fc2_out,
            image_size=old.image_size,
            conv1_out=conv1_keep.numel(),
            conv2_out=conv2_keep.numel(),
            fc1_out=fc1_keep.numel(),
        )

        # conv1 is reduced only along output channels.
        new_model.conv1.weight.copy_(old.conv1.weight[conv1_keep])
        if old.conv1.bias is not None:
            new_model.conv1.bias.copy_(old.conv1.bias[conv1_keep])

        # conv2 is reduced along both input and output channels.
        conv2_w = old.conv2.weight[:, conv1_keep, :, :]
        conv2_w = conv2_w[conv2_keep]
        new_model.conv2.weight.copy_(conv2_w)
        if old.conv2.bias is not None:
            new_model.conv2.bias.copy_(old.conv2.bias[conv2_keep])

        # fc1 input columns correspond to contiguous flattened conv2 channel blocks.
        feat_size = old.image_size // 4
        spatial = feat_size * feat_size
        flatten_keep_idx = torch.cat(
            [torch.arange(c.item() * spatial, (c.item() + 1) * spatial) for c in conv2_keep]
        )
        fc1_w = old.fc1.weight[:, flatten_keep_idx]
        fc1_w = fc1_w[fc1_keep]
        new_model.fc1.weight.copy_(fc1_w)
        if old.fc1.bias is not None:
            new_model.fc1.bias.copy_(old.fc1.bias[fc1_keep])

        # The final classifier keeps all output classes and only reduces inputs.
        fc2_w = old.fc2.weight[:, fc1_keep]
        new_model.fc2.weight.copy_(fc2_w)
        if old.fc2.bias is not None:
            new_model.fc2.bias.copy_(old.fc2.bias)

        return new_model

    def prune_and_rebuild(self, ratio: Optional[float] = None) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        """Compute kept indices and return the rebuilt pruned model."""
        keep_indices = self.compute_keep_indices(ratio)
        rebuilt = self.rebuild_pruned_model(keep_indices)
        return rebuilt, keep_indices

    @staticmethod
    def expand_to_global_shape(
        pruned_model: nn.Module, global_template: nn.Module, keep_indices: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Expand a pruned local state_dict back to the global model shape.

        Parameters absent from the pruned model retain their global-template
        values, which makes their client delta exactly zero for that round.
        """
        from models.cnn import SimpleCNN

        if not isinstance(global_template, SimpleCNN):
            raise TypeError("expand_to_global_shape currently supports only SimpleCNN.")

        pruned_state = pruned_model.state_dict()
        full_state = {
            name: param.detach().clone()
            for name, param in global_template.state_dict().items()
        }

        conv1_keep = keep_indices["conv1"]
        conv2_keep = keep_indices["conv2"]
        fc1_keep = keep_indices["fc1"]
        feat_size = global_template.image_size // 4
        spatial = feat_size * feat_size
        flatten_keep_idx = torch.cat(
            [torch.arange(c.item() * spatial, (c.item() + 1) * spatial) for c in conv2_keep]
        )

        full_state["conv1.weight"][conv1_keep] = pruned_state["conv1.weight"]
        full_state["conv1.bias"][conv1_keep] = pruned_state["conv1.bias"]
        full_state["conv2.weight"][conv2_keep.unsqueeze(1), conv1_keep.unsqueeze(0)] = pruned_state["conv2.weight"]
        full_state["conv2.bias"][conv2_keep] = pruned_state["conv2.bias"]
        full_state["fc1.weight"][fc1_keep.unsqueeze(1), flatten_keep_idx.unsqueeze(0)] = pruned_state["fc1.weight"]
        full_state["fc1.bias"][fc1_keep] = pruned_state["fc1.bias"]
        full_state["fc2.weight"][:, fc1_keep] = pruned_state["fc2.weight"]
        full_state["fc2.bias"] = pruned_state["fc2.bias"]

        return full_state

    @staticmethod
    def contribution_masks(
        global_template: nn.Module, keep_indices: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Build masks for parameters that were present in a structured-pruned model.

        True entries are parameters retained and trained by the client. False
        entries were absent from the pruned model and should not dilute server
        aggregation for that parameter.
        """
        from models.cnn import SimpleCNN

        if not isinstance(global_template, SimpleCNN):
            raise TypeError("contribution_masks currently supports only SimpleCNN.")

        masks = {
            name: torch.zeros_like(param, dtype=torch.bool)
            for name, param in global_template.state_dict().items()
        }

        conv1_keep = keep_indices["conv1"]
        conv2_keep = keep_indices["conv2"]
        fc1_keep = keep_indices["fc1"]
        feat_size = global_template.image_size // 4
        spatial = feat_size * feat_size
        flatten_keep_idx = torch.cat(
            [torch.arange(c.item() * spatial, (c.item() + 1) * spatial) for c in conv2_keep]
        )

        masks["conv1.weight"][conv1_keep] = True
        masks["conv1.bias"][conv1_keep] = True
        masks["conv2.weight"][conv2_keep.unsqueeze(1), conv1_keep.unsqueeze(0)] = True
        masks["conv2.bias"][conv2_keep] = True
        masks["fc1.weight"][fc1_keep.unsqueeze(1), flatten_keep_idx.unsqueeze(0)] = True
        masks["fc1.bias"][fc1_keep] = True
        masks["fc2.weight"][:, fc1_keep] = True
        masks["fc2.bias"][:] = True

        return masks


# Per-client capability metadata for adaptive pruning.
@dataclass
class ClientCapability:
    """Approximate simulated client capability."""

    client_id: str
    compute_flops_per_sec: float
    bandwidth_mbps: float


class AdaptivePruningRatioController:
    """
    Maintain client-specific pruning ratios.

    Warm-start mode assigns higher pruning ratios to weaker simulated clients.
    Online mode starts from a midpoint and adjusts ratios from observed local
    round times.
    """

    def __init__(self, config: PruningConfig, warm_start: bool = True):
        self.config = config
        self.warm_start = warm_start
        self._observed_time: Dict[str, float] = {}
        self._current_ratio: Dict[str, float] = {}

    def initialize_warm_start(self, capabilities: List[ClientCapability]) -> Dict[str, float]:
        """
        Initialize ratios from simulated capability scores.

        score = compute_flops_per_sec * bandwidth_mbps. Larger scores represent
        stronger clients and therefore receive lower pruning ratios.
        """
        scores = [c.compute_flops_per_sec * c.bandwidth_mbps for c in capabilities]
        max_score = max(scores) if scores else 1.0
        min_score = min(scores) if scores else 1.0
        score_range = max(max_score - min_score, 1e-8)

        ratios: Dict[str, float] = {}
        for cap, score in zip(capabilities, scores):
            normalized_strength = (score - min_score) / score_range
            ratio = self.config.max_ratio - normalized_strength * (
                self.config.max_ratio - self.config.min_ratio
            )
            ratio = float(min(max(ratio, self.config.min_ratio), self.config.max_ratio))
            ratios[cap.client_id] = ratio
            self._current_ratio[cap.client_id] = ratio

        return ratios

    def update_from_round_time(
        self, client_id: str, round_time: float, target_time: float, step_size: float = 0.05
    ) -> float:
        """
        Update an online pruning ratio from observed client runtime.

        Slower-than-target clients receive more pruning; faster clients can
        reduce pruning to preserve accuracy.
        """
        self._observed_time[client_id] = round_time
        current = self._current_ratio.get(client_id, self.config.pruning_ratio)

        if round_time > target_time:
            current = min(self.config.max_ratio, current + step_size)
        elif round_time < target_time * 0.8:
            current = max(self.config.min_ratio, current - step_size)

        self._current_ratio[client_id] = current
        return current

    def get_ratio(self, client_id: str) -> float:
        return self._current_ratio.get(client_id, self.config.pruning_ratio)

    def get_observed_time(self, client_id: str) -> Optional[float]:
        return self._observed_time.get(client_id)

    def initialize_ratios(self, client_ids: list, ratio: float) -> None:
        for cid in client_ids:
            self._current_ratio[cid] = ratio


# Factory for configured pruning strategies.
def build_pruner(model: nn.Module, config: PruningConfig):
    if config.mode == "unstructured":
        return UnstructuredPruner(model, config)
    elif config.mode == "structured":
        return StructuredPruner(model, config)
    else:
        raise ValueError(f"Unknown pruning mode: {config.mode}")

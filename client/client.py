"""
Client-side logic for the federated learning simulation.

Each client receives the current global model, optionally prunes it before local
training, performs local optimization, and returns only the model delta to the
server. If communication sparsification is enabled, the delta is compressed
before transmission.

Pruning and sparsification are intentionally separated: pruning reduces local
training cost before optimization, while sparsification reduces the outbound
communication payload after optimization.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model_pruning import (
    AdaptivePruningRatioController,
    PruningConfig,
    StructuredPruner,
    UnstructuredPruner,
    build_pruner,
)
from utils.gradient_sparsification import (
    BaseSparsifier,
    CostWeightedSparsifier,
    SparsificationConfig,
    build_sparsifier,
)


@dataclass
class ClientConfig:
    local_epochs: int = 2
    learning_rate: float = 0.01
    device: str = "cpu"
    enable_pruning: bool = True
    enable_sparsification: bool = True


class FederatedClient:
    def __init__(
        self,
        client_id: str,
        train_loader: DataLoader,
        pruning_config: PruningConfig,
        sparsification_config: SparsificationConfig,
        client_config: ClientConfig,
        ratio_controller: Optional[AdaptivePruningRatioController] = None,
    ):
        self.client_id = client_id
        self.train_loader = train_loader
        self.pruning_config = pruning_config
        self.sparsification_config = sparsification_config
        self.client_config = client_config
        self.ratio_controller = ratio_controller

        # Each client owns its own error-feedback residual buffer.
        self._sparsifier: Optional[BaseSparsifier] = None

        # Used by online adaptive pruning to tune the next-round pruning ratio.
        self.last_round_time: float = 0.0

    def _get_sparsifier(self) -> BaseSparsifier:
        if self._sparsifier is None:
            self._sparsifier = build_sparsifier(self.sparsification_config)
        return self._sparsifier

    def _current_pruning_ratio(self) -> float:
        if self.ratio_controller is not None:
            return self.ratio_controller.get_ratio(self.client_id)
        if self.pruning_config.per_client_ratio is not None:
            return self.pruning_config.per_client_ratio.get(
                self.client_id, self.pruning_config.pruning_ratio
            )
        return self.pruning_config.pruning_ratio

    def local_update(
        self, global_model: nn.Module
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Run one full local training round and return the transmitted delta and mask.

        Both returned dictionaries have the same tensor shapes as the global
        model state_dict, even when structured pruning trains a smaller local
        model. The contribution mask marks parameters retained and trained by
        this client.
        """
        start_time = time.perf_counter()
        device = torch.device(self.client_config.device)
        global_state = global_model.state_dict()

        local_model: nn.Module
        unstructured_pruner: Optional[UnstructuredPruner] = None
        structured_keep_indices = None

        # Stage 1: reduce local model cost before optimization, when enabled.
        if self.client_config.enable_pruning:
            ratio = self._current_pruning_ratio()

            if self.pruning_config.mode == "unstructured":
                local_model = copy.deepcopy(global_model).to(device)
                pruner = build_pruner(local_model, self.pruning_config)
                assert isinstance(pruner, UnstructuredPruner)
                pruner.prune(ratio=ratio)
                unstructured_pruner = pruner

            elif self.pruning_config.mode == "structured":
                # Structured pruning rebuilds a smaller model and records kept
                # indices so its state can later be expanded to global shape.
                template = copy.deepcopy(global_model)
                pruner = build_pruner(template, self.pruning_config)
                assert isinstance(pruner, StructuredPruner)
                local_model, structured_keep_indices = pruner.prune_and_rebuild(ratio=ratio)
                local_model = local_model.to(device)

            else:
                raise ValueError(f"Unknown pruning mode: {self.pruning_config.mode}")
        else:
            local_model = copy.deepcopy(global_model).to(device)

        optimizer = torch.optim.SGD(local_model.parameters(), lr=self.client_config.learning_rate)

        # Unstructured masks must be re-applied after each optimizer step.
        if unstructured_pruner is not None:
            unstructured_pruner.register_step_hook(optimizer)

        criterion = nn.CrossEntropyLoss()

        # Stage 2: train on the client's local data partition.
        local_model.train()
        for _ in range(self.client_config.local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = local_model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

        # Compute only the model change relative to the broadcast global model.
        if structured_keep_indices is not None:
            expanded_local_state = StructuredPruner.expand_to_global_shape(
                local_model, global_model, structured_keep_indices
            )
            local_state = expanded_local_state
            contribution_masks = StructuredPruner.contribution_masks(
                global_model, structured_keep_indices
            )
        else:
            local_state = local_model.state_dict()
            contribution_masks = {
                name: torch.ones_like(param, dtype=torch.bool)
                for name, param in global_state.items()
            }

        deltas = {
            name: (local_state[name].cpu() - global_state[name].cpu())
            for name in global_state
        }

        # Stage 3: sparsify the outbound delta to reduce communication cost.
        if self.client_config.enable_sparsification:
            sparsifier = self._get_sparsifier()
            if isinstance(sparsifier, CostWeightedSparsifier):
                sparsifier.auto_register_cost_from_model(global_model)
            deltas = sparsifier.sparsify_state_dict(deltas)

        self.last_round_time = time.perf_counter() - start_time

        if self.ratio_controller is not None and not self.ratio_controller.warm_start:
            target_time = self._estimate_target_round_time()
            self.ratio_controller.update_from_round_time(
                self.client_id, self.last_round_time, target_time
            )

        return deltas, contribution_masks

    def _estimate_target_round_time(self) -> float:
        """Return the baseline time used by online adaptive pruning."""
        if self.ratio_controller is None:
            return self.last_round_time if self.last_round_time > 0 else 1.0
        history_time = self.ratio_controller.get_observed_time(self.client_id)
        if history_time is not None:
            return history_time
        return self.last_round_time if self.last_round_time > 0 else 1.0

    def report_stats(self, deltas: Dict[str, torch.Tensor]) -> Dict[str, object]:
        """Return communication and timing statistics for logging and storage."""
        total, nonzero = 0, 0
        for tensor in deltas.values():
            total += tensor.numel()
            nonzero += (tensor != 0).sum().item()
        return {
            "client_id": self.client_id,
            "transmitted_ratio": nonzero / total if total > 0 else 0.0,
            "round_time_sec": self.last_round_time,
            "pruning_ratio": self._current_pruning_ratio() if self.client_config.enable_pruning else 0.0,
        }

"""
Server-side orchestration for the federated learning simulation.

The server owns the global model, broadcasts it to selected clients, aggregates
client deltas, and evaluates the updated model. In addition to weighted FedAvg,
it supports Krum and coordinate-wise trimmed mean for robust aggregation.
Pruning and sparsification happen on the client side, so the server receives
deltas with the global model shape and optional contribution masks.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class FederatedServer:
    def __init__(self, global_model: nn.Module):
        self.global_model = global_model

    def broadcast(self) -> nn.Module:
        """Return the current global model; clients create their own copies."""
        return self.global_model

    def _normalize_client_weights(
        self, client_weights: Optional[List[float]], num_clients: int
    ) -> List[float]:
        """Return validated FedAvg weights that sum to one."""
        if client_weights is None:
            return [1.0 / num_clients] * num_clients
        if len(client_weights) != num_clients:
            raise ValueError("client_weights must match client_deltas length.")
        if any(weight < 0 for weight in client_weights):
            raise ValueError("client_weights must be non-negative.")

        total = sum(client_weights)
        if total <= 0:
            raise ValueError("client_weights must have a positive total.")
        return [weight / total for weight in client_weights]

    @staticmethod
    def _validate_masks(
        client_deltas: List[Dict[str, torch.Tensor]],
        contribution_masks: Optional[List[Dict[str, torch.Tensor]]],
    ) -> List[Dict[str, torch.Tensor]]:
        """Return masks, defaulting to full contribution for supplied deltas."""
        if contribution_masks is None:
            return [
                {
                    name: torch.ones_like(delta, dtype=torch.bool)
                    for name, delta in deltas.items()
                }
                for deltas in client_deltas
            ]
        if len(contribution_masks) != len(client_deltas):
            raise ValueError("contribution_masks must match client_deltas length.")

        for deltas, masks in zip(client_deltas, contribution_masks):
            for name, delta in deltas.items():
                if name not in masks:
                    raise ValueError(f"Missing contribution mask for parameter '{name}'.")
                if masks[name].shape != delta.shape:
                    raise ValueError(
                        f"Contribution mask for parameter '{name}' does not match its delta shape."
                    )
        return contribution_masks

    @staticmethod
    def _masked_delta(
        deltas: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
        name: str,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return one client update and mask aligned to a global-state tensor.

        Missing parameters are treated as unavailable rather than as zero-valued
        contributions. This preserves structured-pruning semantics.
        """
        if name not in deltas:
            return (
                torch.zeros_like(reference),
                torch.zeros_like(reference, dtype=torch.bool),
            )

        delta = deltas[name]
        if delta.shape != reference.shape:
            raise ValueError(f"Delta for parameter '{name}' does not match the global model shape.")
        mask = masks[name]
        return (
            delta.to(device=reference.device, dtype=reference.dtype),
            mask.to(device=reference.device, dtype=torch.bool),
        )

    def _fedavg_update(
        self,
        client_deltas: List[Dict[str, torch.Tensor]],
        client_weights: List[float],
        contribution_masks: List[Dict[str, torch.Tensor]],
        global_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute the mask-aware, weighted FedAvg update."""
        aggregated = {name: torch.zeros_like(param) for name, param in global_state.items()}
        denominators = {name: torch.zeros_like(param) for name, param in global_state.items()}

        for deltas, masks, weight in zip(client_deltas, contribution_masks, client_weights):
            for name, parameter in global_state.items():
                if not parameter.is_floating_point():
                    continue
                delta, mask = self._masked_delta(deltas, masks, name, parameter)
                numeric_mask = mask.to(dtype=parameter.dtype)
                aggregated[name] += weight * numeric_mask * delta
                denominators[name] += weight * numeric_mask

        for name, parameter in global_state.items():
            if parameter.is_floating_point():
                aggregated[name] = torch.where(
                    denominators[name] > 0,
                    aggregated[name]
                    / denominators[name].clamp_min(torch.finfo(parameter.dtype).eps),
                    aggregated[name],
                )
        return aggregated

    def _krum_selected_index(
        self,
        client_deltas: List[Dict[str, torch.Tensor]],
        contribution_masks: List[Dict[str, torch.Tensor]],
        global_state: Dict[str, torch.Tensor],
        krum_f: int,
    ) -> int:
        """
        Select the Krum client whose update is closest to its honest neighbors.

        Pairwise squared Euclidean distances include only coordinates retained by
        both clients. This makes Krum compatible with structured-pruning masks.
        Standard Krum applies this single selected update directly rather than
        averaging it with other client deltas. Consequently, it can converge
        more slowly than FedAvg on benign, strongly non-IID data because each
        round learns from only one client's local distribution; this is the
        expected robustness-versus-statistical-efficiency trade-off.
        """
        num_clients = len(client_deltas)
        if krum_f < 0:
            raise ValueError("krum_f must be non-negative.")
        if num_clients < 2 * krum_f + 3:
            raise ValueError(
                "Krum requires at least 2 * krum_f + 3 client updates; "
                f"received {num_clients} updates with krum_f={krum_f}."
            )

        neighbor_count = num_clients - krum_f - 2
        pairwise_distances = torch.full(
            (num_clients, num_clients), float("inf"), device=next(self.global_model.parameters()).device
        )
        pairwise_distances.fill_diagonal_(0.0)

        for first_index in range(num_clients):
            for second_index in range(first_index + 1, num_clients):
                distance = torch.zeros(
                    (), device=pairwise_distances.device, dtype=torch.float64
                )
                shared_coordinates = 0
                for name, parameter in global_state.items():
                    if not parameter.is_floating_point():
                        continue
                    first_delta, first_mask = self._masked_delta(
                        client_deltas[first_index],
                        contribution_masks[first_index],
                        name,
                        parameter,
                    )
                    second_delta, second_mask = self._masked_delta(
                        client_deltas[second_index],
                        contribution_masks[second_index],
                        name,
                        parameter,
                    )
                    shared_mask = first_mask & second_mask
                    if shared_mask.any():
                        difference = first_delta[shared_mask] - second_delta[shared_mask]
                        distance += torch.sum(difference.to(torch.float64).square())
                        shared_coordinates += shared_mask.sum().item()

                if shared_coordinates > 0:
                    pairwise_distances[first_index, second_index] = distance
                    pairwise_distances[second_index, first_index] = distance

        scores = []
        for client_index in range(num_clients):
            distances = torch.cat(
                (
                    pairwise_distances[client_index, :client_index],
                    pairwise_distances[client_index, client_index + 1 :],
                )
            )
            scores.append(torch.topk(distances, k=neighbor_count, largest=False).values.sum())

        return int(torch.argmin(torch.stack(scores)).item())

    def _trimmed_mean_update(
        self,
        client_deltas: List[Dict[str, torch.Tensor]],
        contribution_masks: List[Dict[str, torch.Tensor]],
        global_state: Dict[str, torch.Tensor],
        trim_ratio: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute a mask-aware, coordinate-wise trimmed-mean update.

        For each parameter coordinate, unavailable structured-pruning entries do
        not participate. The lowest and highest ``floor(n * trim_ratio)`` valid
        updates are removed before taking the mean.
        """
        if not 0.0 <= trim_ratio < 0.5:
            raise ValueError("trim_ratio must satisfy 0.0 <= trim_ratio < 0.5.")

        aggregated: Dict[str, torch.Tensor] = {}
        for name, parameter in global_state.items():
            if not parameter.is_floating_point():
                aggregated[name] = torch.zeros_like(parameter)
                continue
            values, masks = zip(
                *[
                    self._masked_delta(deltas, contribution_mask, name, parameter)
                    for deltas, contribution_mask in zip(client_deltas, contribution_masks)
                ]
            )
            stacked_values = torch.stack(values)
            stacked_masks = torch.stack(masks)
            contribution_count = stacked_masks.sum(dim=0)
            trim_count = torch.floor(contribution_count.to(torch.float64) * trim_ratio).to(torch.long)

            # Sorting unavailable values last lets rank-based indexing operate
            # independently for every coordinate without per-element Python loops.
            sorted_values = torch.sort(
                stacked_values.masked_fill(~stacked_masks, float("inf")), dim=0
            ).values
            ranks = torch.arange(
                len(client_deltas), device=parameter.device
            ).view((-1,) + (1,) * parameter.ndim)
            retained = (ranks >= trim_count) & (ranks < contribution_count - trim_count)
            retained_count = retained.sum(dim=0)
            summed = torch.where(retained, sorted_values, torch.zeros_like(sorted_values)).sum(dim=0)

            aggregated[name] = torch.where(
                retained_count > 0,
                summed / retained_count.clamp_min(1).to(parameter.dtype),
                torch.zeros_like(parameter),
            )
        return aggregated

    def aggregate(
        self,
        client_deltas: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
        contribution_masks: Optional[List[Dict[str, torch.Tensor]]] = None,
        aggregation: str = "fedavg",
        krum_f: int = 0,
        trim_ratio: float = 0.0,
    ) -> None:
        """
        Aggregate client deltas and apply the update to the global model.

        ``fedavg`` performs the existing mask-aware weighted FedAvg operation.
        ``krum`` selects one update with the smallest Krum neighbor-distance
        score and applies that selected update directly; it ignores
        ``client_weights`` because standard Krum is unweighted. This robust
        single-update rule can be less accurate than FedAvg without attacks,
        particularly under strong non-IID client label skew.
        ``trimmed_mean`` averages each coordinate after symmetrically trimming
        outliers; it also ignores ``client_weights`` for the standard unweighted
        robust estimator. Contribution masks ensure structured-pruned clients
        only affect coordinates that they retained and trained.
        """
        if not client_deltas:
            return
        if aggregation not in {"fedavg", "krum", "trimmed_mean"}:
            raise ValueError(
                "aggregation must be one of: 'fedavg', 'krum', or 'trimmed_mean'."
            )

        contribution_masks = self._validate_masks(client_deltas, contribution_masks)
        global_state = self.global_model.state_dict()

        if aggregation == "fedavg":
            client_weights = self._normalize_client_weights(client_weights, len(client_deltas))
            aggregated = self._fedavg_update(
                client_deltas, client_weights, contribution_masks, global_state
            )
        elif aggregation == "krum":
            selected_index = self._krum_selected_index(
                client_deltas, contribution_masks, global_state, krum_f
            )
            aggregated = self._fedavg_update(
                [client_deltas[selected_index]],
                [1.0],
                [contribution_masks[selected_index]],
                global_state,
            )
        else:
            aggregated = self._trimmed_mean_update(
                client_deltas, contribution_masks, global_state, trim_ratio
            )

        new_state = {name: global_state[name] + aggregated[name] for name in global_state}
        self.global_model.load_state_dict(new_state)

    def evaluate(self, test_loader, device: str = "cpu") -> float:
        """Evaluate the global model on the provided test loader."""
        self.global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                output = self.global_model(x)
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0

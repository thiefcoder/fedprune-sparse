"""
Server-side orchestration for the federated learning simulation.

The server owns the global model, broadcasts it to selected clients, aggregates
client deltas with FedAvg, and evaluates the updated model. Pruning and
sparsification happen on the client side, so the server always receives deltas
with the global model shape.
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

    def aggregate(
        self,
        client_deltas: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
    ) -> None:
        """
        Average client deltas and apply the aggregated update to the global model.

        If no client weights are supplied, each selected client contributes
        equally. Passing sample-count weights recovers weighted FedAvg.
        """
        if not client_deltas:
            return

        n = len(client_deltas)
        if client_weights is None:
            client_weights = [1.0 / n] * n
        else:
            total = sum(client_weights)
            client_weights = [w / total for w in client_weights]

        global_state = self.global_model.state_dict()
        aggregated = {name: torch.zeros_like(param) for name, param in global_state.items()}

        for deltas, weight in zip(client_deltas, client_weights):
            for name, delta in deltas.items():
                aggregated[name] += weight * delta.to(aggregated[name].device)

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

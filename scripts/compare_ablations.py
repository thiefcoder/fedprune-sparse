"""
Run a high-level ablation suite for FedPrune-Sparse.

The suite compares plain FedAvg, pruning-only, sparsification-only, and the full
hybrid method. Each experiment writes metrics, logs, summaries, model weights,
and figures to a separate results subdirectory.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(label: str, run_name: str, extra_args: list[str]) -> None:
    print("\n" + "=" * 80)
    print(f"Experiment: {label}")
    print("=" * 80)
    cmd = [sys.executable, str(ROOT / "run_simulation.py"), "--run_name", run_name] + extra_args
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run FedPrune-Sparse ablation experiments.")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--clients_per_round", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=2)
    args, unknown = parser.parse_known_args()

    common = [
        "--rounds", str(args.rounds),
        "--num_clients", str(args.num_clients),
        "--clients_per_round", str(args.clients_per_round),
        "--local_epochs", str(args.local_epochs),
    ] + unknown

    run(
        "Baseline FedAvg without pruning or sparsification",
        "baseline_fedavg",
        common + ["--no_pruning", "--no_sparsification", "--no_adaptive_ratio"],
    )

    run(
        "Unstructured model pruning only",
        "pruning_only_unstructured",
        common + ["--no_sparsification", "--pruning_mode", "unstructured", "--pruning_ratio", "0.5"],
    )

    run(
        "Top-K delta sparsification only",
        "sparsification_only_topk",
        common + ["--no_pruning", "--no_adaptive_ratio", "--sparsify_method", "topk", "--sparsity_ratio", "0.95"],
    )

    run(
        "Hybrid structured adaptive pruning with cost-weighted sparsification",
        "hybrid_structured_cwmp",
        common + [
            "--pruning_mode", "structured",
            "--pruning_ratio", "0.4",
            "--adaptive_ratio",
            "--sparsify_method", "cost_weighted",
            "--sparsity_ratio", "0.95",
        ],
    )


if __name__ == "__main__":
    main()

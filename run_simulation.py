"""
Main entry point for FedPrune-Sparse experiments.

This script assembles the full simulation pipeline: MNIST preparation,
non-IID client partitioning, client/server construction, federated training,
metric logging, checkpointing, and publication-ready plot generation.

Each run writes its configuration, round-level metrics, structured log,
summary, final model weights, and figures into a dedicated subdirectory under
the results directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.append(str(Path(__file__).resolve().parent))

from client.client import ClientConfig, FederatedClient
from models.cnn import SimpleCNN
from server.server import FederatedServer
from utils.gradient_sparsification import SparsificationConfig
from utils.model_pruning import (
    AdaptivePruningRatioController,
    ClientCapability,
    PruningConfig,
)


def setup_logging(run_dir: Path) -> logging.Logger:
    """Create a run-specific English logger for console and file output."""
    logger = logging.getLogger("fedprune_sparse")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def partition_non_iid(dataset, num_clients: int, classes_per_client: int = 2, seed: int = 0):
    """
    Partition a classification dataset into label-skewed non-IID client subsets.

    Samples are grouped by class, split into shards, shuffled, and assigned so
    each client receives only a small number of classes. This approximates a
    realistic federated label-distribution shift.
    """
    rng = random.Random(seed)
    targets = dataset.targets if hasattr(dataset, "targets") else dataset.labels
    targets = torch.as_tensor(targets)
    num_classes = int(targets.max().item()) + 1

    class_indices = {c: (targets == c).nonzero(as_tuple=True)[0].tolist() for c in range(num_classes)}
    for c in class_indices:
        rng.shuffle(class_indices[c])

    total_shards_needed = num_clients * classes_per_client
    shards_per_class = max(1, total_shards_needed // num_classes + 1)

    shards = []
    for c in range(num_classes):
        idx_list = class_indices[c]
        shard_size = max(1, len(idx_list) // shards_per_class)
        for i in range(shards_per_class):
            start = i * shard_size
            end = start + shard_size if i < shards_per_class - 1 else len(idx_list)
            shard = idx_list[start:end]
            if shard:
                shards.append(shard)

    rng.shuffle(shards)

    if len(shards) < total_shards_needed:
        raise ValueError(
            f"Insufficient shards: available={len(shards)}, required={total_shards_needed}. "
            "Reduce num_clients or classes_per_client."
        )

    client_indices = [[] for _ in range(num_clients)]
    shard_ptr = 0
    for client_id in range(num_clients):
        for _ in range(classes_per_client):
            client_indices[client_id].extend(shards[shard_ptr])
            shard_ptr += 1

    return [Subset(dataset, idx) for idx in client_indices if len(idx) > 0]


def build_simulated_capabilities(num_clients: int, seed: int = 0) -> list[ClientCapability]:
    """
    Create synthetic client capability profiles for adaptive pruning experiments.

    The values are normalized proxies, not hardware-calibrated measurements.
    Lower compute or bandwidth generally results in a higher warm-start pruning
    ratio.
    """
    rng = random.Random(seed)
    capabilities = []
    for i in range(num_clients):
        compute = rng.uniform(0.2, 1.0)
        bandwidth = rng.uniform(0.2, 1.0)
        capabilities.append(
            ClientCapability(
                client_id=f"client_{i}",
                compute_flops_per_sec=compute,
                bandwidth_mbps=bandwidth,
            )
        )
    return capabilities


def save_metric_plots(metrics_history: list[dict], run_dir: Path, logger: logging.Logger) -> list[str]:
    """Save high-resolution metric plots into the run directory."""
    if not metrics_history:
        return []

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Matplotlib is not installed; metric plots were not generated.")
        return []

    rounds = [row["round"] for row in metrics_history]
    accuracy = [row["test_accuracy"] * 100 for row in metrics_history]
    transmitted = [row["avg_transmitted_ratio"] * 100 for row in metrics_history]
    pruning = [row["avg_pruning_ratio"] * 100 for row in metrics_history]
    round_time = [row["avg_round_time_sec"] for row in metrics_history]

    saved_paths: list[str] = []

    def save_current_figure(filename: str) -> None:
        path = run_dir / filename
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        saved_paths.append(str(path))

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, accuracy, marker="o", linewidth=2.0, label="Test Accuracy")
    plt.xlabel("Federated Round")
    plt.ylabel("Accuracy (%)")
    plt.title("Global Test Accuracy Across Federated Rounds")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_current_figure("accuracy_curve.png")

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, transmitted, marker="s", linewidth=2.0, label="Transmitted Delta Ratio")
    plt.plot(rounds, pruning, marker="^", linewidth=2.0, label="Pruning Ratio")
    plt.xlabel("Federated Round")
    plt.ylabel("Ratio (%)")
    plt.title("Compression and Pruning Dynamics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_current_figure("compression_pruning_curve.png")

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, round_time, marker="d", linewidth=2.0, color="#7B3294")
    plt.xlabel("Federated Round")
    plt.ylabel("Average Client Update Time (s)")
    plt.title("Average Local Update Runtime")
    plt.grid(True, alpha=0.3)
    save_current_figure("round_time_curve.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(rounds, accuracy, marker="o")
    axes[0, 0].set_title("Test Accuracy")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 1].plot(rounds, transmitted, marker="s", color="#008837")
    axes[0, 1].set_title("Transmitted Ratio")
    axes[0, 1].set_ylabel("Ratio (%)")
    axes[1, 0].plot(rounds, pruning, marker="^", color="#E66101")
    axes[1, 0].set_title("Pruning Ratio")
    axes[1, 0].set_ylabel("Ratio (%)")
    axes[1, 1].plot(rounds, round_time, marker="d", color="#7B3294")
    axes[1, 1].set_title("Average Round Time")
    axes[1, 1].set_ylabel("Seconds")
    for ax in axes.flat:
        ax.set_xlabel("Federated Round")
        ax.grid(True, alpha=0.3)
    save_current_figure("experiment_dashboard.png")

    return saved_paths


def write_metrics_csv(metrics_history: list[dict], history_path: Path) -> None:
    """Persist round-level metrics in a tabular CSV file."""
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_history[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_history)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run doctoral-level FedPrune-Sparse federated learning experiments on MNIST."
    )
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--clients_per_round", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--classes_per_client", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--pruning_mode", type=str, default="structured", choices=["unstructured", "structured"])
    parser.add_argument("--pruning_ratio", type=float, default=0.4)
    parser.add_argument("--no_pruning", action="store_true")
    parser.add_argument(
        "--adaptive_ratio",
        action="store_true",
        default=True,
        help="Enable client-specific pruning ratios. Use --no_adaptive_ratio to disable.",
    )
    parser.add_argument("--no_adaptive_ratio", dest="adaptive_ratio", action="store_false")
    parser.add_argument(
        "--adaptive_ratio_online",
        action="store_true",
        help="Adjust pruning ratios online using previous client round times.",
    )
    parser.add_argument("--min_pruning_ratio", type=float, default=0.1)
    parser.add_argument("--max_pruning_ratio", type=float, default=0.8)

    parser.add_argument("--sparsify_method", type=str, default="cost_weighted", choices=["topk", "random", "cost_weighted"])
    parser.add_argument("--sparsity_ratio", type=float, default=0.95)
    parser.add_argument("--no_sparsification", action="store_true")
    parser.add_argument("--no_error_feedback", action="store_true")

    parser.add_argument("--data_root", type=str, default="./data", help="Directory for MNIST download/cache.")
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory for outputs and logs.")
    parser.add_argument("--run_name", type=str, default=None, help="Optional run subdirectory name.")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None):
    args = parse_args(argv)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.results_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "metrics.csv"
    config_path = run_dir / "config.json"
    summary_path = run_dir / "summary.json"
    model_path = run_dir / "final_model.pt"

    logger = setup_logging(run_dir)
    logger.info("Starting FedPrune-Sparse experiment.")
    logger.info("Run directory: %s", run_dir)

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    logger.info("Configuration saved to %s", config_path)

    logger.info("Preparing MNIST datasets from %s.", args.data_root)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST(root=args.data_root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=args.data_root, train=False, download=True, transform=transform)

    client_subsets = partition_non_iid(
        train_set, args.num_clients, classes_per_client=args.classes_per_client, seed=args.seed
    )
    train_loaders = [DataLoader(s, batch_size=args.batch_size, shuffle=True) for s in client_subsets]
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    logger.info(
        "Data partitioned into %d non-IID clients with %d classes per client.",
        len(train_loaders),
        args.classes_per_client,
    )

    pruning_config = PruningConfig(
        mode=args.pruning_mode,
        pruning_ratio=args.pruning_ratio,
        min_ratio=args.min_pruning_ratio,
        max_ratio=args.max_pruning_ratio,
    )
    sparsification_config = SparsificationConfig(
        method=args.sparsify_method,
        sparsity_ratio=args.sparsity_ratio,
        use_error_feedback=not args.no_error_feedback,
    )
    client_config = ClientConfig(
        local_epochs=args.local_epochs,
        learning_rate=args.lr,
        device=args.device,
        enable_pruning=not args.no_pruning,
        enable_sparsification=not args.no_sparsification,
    )

    global_model = SimpleCNN(in_channels=1, num_classes=10, image_size=28).to(args.device)
    server = FederatedServer(global_model)

    ratio_controller = None
    if args.adaptive_ratio and not args.no_pruning:
        capabilities = build_simulated_capabilities(len(train_loaders), seed=args.seed)
        ratio_controller = AdaptivePruningRatioController(
            pruning_config, warm_start=not args.adaptive_ratio_online
        )
        if ratio_controller.warm_start:
            initial_ratios = ratio_controller.initialize_warm_start(capabilities)
            logger.info("Initial adaptive pruning ratios:")
            for cid, ratio in initial_ratios.items():
                logger.info("  %s: pruning_ratio=%.3f", cid, ratio)
        else:
            mid_ratio = (pruning_config.min_ratio + pruning_config.max_ratio) / 2
            ratio_controller.initialize_ratios([cap.client_id for cap in capabilities], mid_ratio)
            logger.info("Online adaptive pruning initialized at ratio %.3f.", mid_ratio)

    clients = [
        FederatedClient(
            client_id=f"client_{i}",
            train_loader=train_loaders[i],
            pruning_config=pruning_config,
            sparsification_config=sparsification_config,
            client_config=client_config,
            ratio_controller=ratio_controller,
        )
        for i in range(len(train_loaders))
    ]

    logger.info(
        "Experiment setup | clients=%d | clients_per_round=%d | rounds=%d | local_epochs=%d | "
        "pruning=%s (%s, ratio=%.3f, adaptive=%s) | sparsification=%s (%s, sparsity=%.3f, error_feedback=%s)",
        len(clients),
        args.clients_per_round,
        args.rounds,
        args.local_epochs,
        client_config.enable_pruning,
        args.pruning_mode,
        args.pruning_ratio,
        args.adaptive_ratio,
        client_config.enable_sparsification,
        args.sparsify_method,
        args.sparsity_ratio,
        not args.no_error_feedback,
    )

    metrics_history = []

    for round_idx in range(1, args.rounds + 1):
        selected = random.sample(clients, min(args.clients_per_round, len(clients)))
        selected_ids = [client.client_id for client in selected]
        logger.info("Round %03d started | selected_clients=%s", round_idx, ",".join(selected_ids))

        round_deltas = []
        round_contribution_masks = []
        transmitted_ratios = []
        pruning_ratios = []
        round_times = []
        for client in selected:
            deltas, contribution_masks = client.local_update(server.broadcast())
            round_deltas.append(deltas)
            round_contribution_masks.append(contribution_masks)
            stats = client.report_stats(deltas)
            transmitted_ratios.append(stats["transmitted_ratio"])
            pruning_ratios.append(stats["pruning_ratio"])
            round_times.append(stats["round_time_sec"])
            logger.info(
                "Round %03d client %s | transmitted_ratio=%.6f | pruning_ratio=%.6f | update_time_sec=%.4f",
                round_idx,
                stats["client_id"],
                stats["transmitted_ratio"],
                stats["pruning_ratio"],
                stats["round_time_sec"],
            )

        server.aggregate(round_deltas, contribution_masks=round_contribution_masks)
        acc = server.evaluate(test_loader, device=args.device)
        avg_transmitted = sum(transmitted_ratios) / len(transmitted_ratios)
        avg_pruning = sum(pruning_ratios) / len(pruning_ratios) if pruning_ratios else 0.0
        avg_round_time = sum(round_times) / len(round_times)

        metrics = {
            "round": round_idx,
            "test_accuracy": acc,
            "avg_transmitted_ratio": avg_transmitted,
            "avg_pruning_ratio": avg_pruning,
            "avg_round_time_sec": avg_round_time,
        }
        metrics_history.append(metrics)

        logger.info(
            "Round %03d completed | test_accuracy=%.4f%% | avg_transmitted_ratio=%.4f%% | "
            "avg_pruning_ratio=%.4f%% | avg_round_time_sec=%.4f",
            round_idx,
            acc * 100,
            avg_transmitted * 100,
            avg_pruning * 100,
            avg_round_time,
        )

    if metrics_history:
        write_metrics_csv(metrics_history, history_path)
        torch.save(server.global_model.state_dict(), model_path)
        figure_paths = save_metric_plots(metrics_history, run_dir, logger)

        summary = {
            "final_test_accuracy": metrics_history[-1]["test_accuracy"],
            "final_avg_transmitted_ratio": metrics_history[-1]["avg_transmitted_ratio"],
            "final_avg_pruning_ratio": metrics_history[-1]["avg_pruning_ratio"],
            "final_avg_round_time_sec": metrics_history[-1]["avg_round_time_sec"],
            "best_test_accuracy": max(row["test_accuracy"] for row in metrics_history),
            "rounds": len(metrics_history),
            "results_dir": str(run_dir),
            "metrics_csv": str(history_path),
            "log_file": str(run_dir / "run.log"),
            "figures": figure_paths,
            "model_checkpoint": str(model_path),
        }
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info("Metrics saved to %s", history_path)
        logger.info("Summary saved to %s", summary_path)
        logger.info("Final model checkpoint saved to %s", model_path)
        if figure_paths:
            logger.info("Figures saved: %s", ", ".join(figure_paths))
        logger.info("Experiment completed successfully.")


if __name__ == "__main__":
    main()

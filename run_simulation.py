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


def partition_non_iid_shards(
    dataset, num_clients: int, classes_per_client: int = 2, seed: int = 0
):
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


def partition_non_iid_dirichlet(
    dataset,
    num_clients: int,
    alpha: float = 0.3,
    min_samples_per_client: int = 10,
    seed: int = 0,
    max_attempts: int = 100,
):
    """
    Partition a classification dataset using label-wise Dirichlet proportions.

    Smaller ``alpha`` values create stronger label skew. Each class is
    independently distributed across clients according to a Dirichlet draw,
    while every example is assigned exactly once. The sampler retries until
    every client receives at least ``min_samples_per_client`` examples.
    """
    if alpha <= 0:
        raise ValueError("dirichlet_alpha must be positive.")
    if min_samples_per_client < 1:
        raise ValueError("min_samples_per_client must be at least 1.")
    if num_clients * min_samples_per_client > len(dataset):
        raise ValueError(
            "num_clients * min_samples_per_client cannot exceed the dataset size."
        )

    targets = dataset.targets if hasattr(dataset, "targets") else dataset.labels
    targets = torch.as_tensor(targets)
    num_classes = int(targets.max().item()) + 1

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for _ in range(max_attempts):
            client_indices = [[] for _ in range(num_clients)]

            for class_id in range(num_classes):
                class_indices = (targets == class_id).nonzero(as_tuple=True)[0]
                class_indices = class_indices[torch.randperm(class_indices.numel())]
                proportions = torch.distributions.Dirichlet(
                    torch.full((num_clients,), alpha, dtype=torch.float32)
                ).sample()

                raw_counts = proportions * class_indices.numel()
                counts = torch.floor(raw_counts).to(torch.long)
                remainder = class_indices.numel() - int(counts.sum().item())
                if remainder:
                    _, extra_indices = torch.topk(raw_counts - counts, remainder)
                    counts[extra_indices] += 1

                start = 0
                for client_id, count in enumerate(counts.tolist()):
                    if count:
                        client_indices[client_id].extend(
                            class_indices[start : start + count].tolist()
                        )
                    start += count

            if min(len(indices) for indices in client_indices) >= min_samples_per_client:
                return [Subset(dataset, indices) for indices in client_indices]

    raise RuntimeError(
        "Unable to construct a Dirichlet partition with the requested minimum "
        f"client size after {max_attempts} attempts. Increase dirichlet_alpha "
        "or lower min_samples_per_client."
    )


def partition_non_iid(
    dataset,
    num_clients: int,
    classes_per_client: int = 2,
    seed: int = 0,
    strategy: str = "dirichlet",
    dirichlet_alpha: float = 0.3,
    min_samples_per_client: int = 10,
):
    """Partition data with either the legacy shard method or Dirichlet label skew."""
    if strategy == "dirichlet":
        return partition_non_iid_dirichlet(
            dataset,
            num_clients=num_clients,
            alpha=dirichlet_alpha,
            min_samples_per_client=min_samples_per_client,
            seed=seed,
        )
    if strategy == "shard":
        return partition_non_iid_shards(
            dataset,
            num_clients=num_clients,
            classes_per_client=classes_per_client,
            seed=seed,
        )
    raise ValueError(f"Unknown partition strategy: {strategy}")


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
    parser.add_argument(
        "--partition_strategy",
        type=str,
        choices=["dirichlet", "shard"],
        default="dirichlet",
        help="Client data partitioning method. Dirichlet is the default non-IID setting.",
    )
    parser.add_argument(
        "--dirichlet_alpha",
        type=float,
        default=0.3,
        help="Dirichlet concentration for label skew; smaller values are more non-IID.",
    )
    parser.add_argument(
        "--min_samples_per_client",
        type=int,
        default=10,
        help="Minimum samples per client when using Dirichlet partitioning.",
    )
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
    parser.add_argument(
        "--warmup_rounds",
        type=int,
        default=0,
        help="Initial rounds without pruning or sparsification. Zero preserves prior behavior.",
    )
    parser.add_argument(
        "--compression_ramp_rounds",
        type=int,
        default=0,
        help="Post-warm-up rounds used to linearly increase pruning and sparsification strength.",
    )

    parser.add_argument(
        "--baseline_fedprox_mu",
        type=float,
        default=0.0,
        help="FedProx proximal coefficient. Zero preserves ordinary local SGD.",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="fedavg",
        choices=["fedavg", "krum", "trimmed_mean"],
        help="Server aggregation strategy.",
    )
    parser.add_argument(
        "--krum_f",
        type=int,
        default=0,
        help="Assumed number of malicious or outlier clients for Krum.",
    )
    parser.add_argument(
        "--trim_ratio",
        type=float,
        default=0.0,
        help="Fraction trimmed from each parameter tail for trimmed-mean aggregation.",
    )

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
        train_set,
        args.num_clients,
        classes_per_client=args.classes_per_client,
        seed=args.seed,
        strategy=args.partition_strategy,
        dirichlet_alpha=args.dirichlet_alpha,
        min_samples_per_client=args.min_samples_per_client,
    )
    train_loaders = [DataLoader(s, batch_size=args.batch_size, shuffle=True) for s in client_subsets]
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    logger.info(
        "Data partitioned into %d non-IID clients | strategy=%s | "
        "dirichlet_alpha=%.3f | classes_per_client=%d.",
        len(train_loaders),
        args.partition_strategy,
        args.dirichlet_alpha,
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
        baseline_fedprox_mu=args.baseline_fedprox_mu,
        warmup_rounds=args.warmup_rounds,
        compression_ramp_rounds=args.compression_ramp_rounds,
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
        "pruning=%s (%s, ratio=%.3f, adaptive=%s) | sparsification=%s (%s, sparsity=%.3f, "
        "error_feedback=%s) | warmup_rounds=%d | compression_ramp_rounds=%d",
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
        args.warmup_rounds,
        args.compression_ramp_rounds,
    )
    logger.info(
        "Baselines | fedprox_mu=%.6f | aggregation=%s | krum_f=%d | trim_ratio=%.3f",
        args.baseline_fedprox_mu,
        args.aggregation,
        args.krum_f,
        args.trim_ratio,
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
        sparsity_ratios = []
        round_times = []
        for client in selected:
            deltas, contribution_masks = client.local_update(
                server.broadcast(), round_idx=round_idx
            )
            round_deltas.append(deltas)
            round_contribution_masks.append(contribution_masks)
            stats = client.report_stats(deltas)
            transmitted_ratios.append(stats["transmitted_ratio"])
            pruning_ratios.append(stats["pruning_ratio"])
            sparsity_ratios.append(stats["sparsity_ratio"])
            round_times.append(stats["round_time_sec"])
            logger.info(
                "Round %03d client %s | transmitted_ratio=%.6f | pruning_ratio=%.6f | "
                "sparsity_ratio=%.6f | update_time_sec=%.4f",
                round_idx,
                stats["client_id"],
                stats["transmitted_ratio"],
                stats["pruning_ratio"],
                stats["sparsity_ratio"],
                stats["round_time_sec"],
            )

        server.aggregate(
            round_deltas,
            contribution_masks=round_contribution_masks,
            aggregation=args.aggregation,
            krum_f=args.krum_f,
            trim_ratio=args.trim_ratio,
        )
        acc = server.evaluate(test_loader, device=args.device)
        avg_transmitted = sum(transmitted_ratios) / len(transmitted_ratios)
        avg_pruning = sum(pruning_ratios) / len(pruning_ratios) if pruning_ratios else 0.0
        avg_sparsity = sum(sparsity_ratios) / len(sparsity_ratios) if sparsity_ratios else 0.0
        avg_round_time = sum(round_times) / len(round_times)

        metrics = {
            "round": round_idx,
            "test_accuracy": acc,
            "avg_transmitted_ratio": avg_transmitted,
            "avg_pruning_ratio": avg_pruning,
            "avg_sparsity_ratio": avg_sparsity,
            "avg_round_time_sec": avg_round_time,
        }
        metrics_history.append(metrics)

        logger.info(
            "Round %03d completed | test_accuracy=%.4f%% | avg_transmitted_ratio=%.4f%% | "
            "avg_pruning_ratio=%.4f%% | avg_sparsity_ratio=%.4f%% | avg_round_time_sec=%.4f",
            round_idx,
            acc * 100,
            avg_transmitted * 100,
            avg_pruning * 100,
            avg_sparsity * 100,
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
            "final_avg_sparsity_ratio": metrics_history[-1]["avg_sparsity_ratio"],
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

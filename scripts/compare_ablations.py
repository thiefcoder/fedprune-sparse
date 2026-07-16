"""
Run standard baselines and the FedPrune-Sparse hybrid comparison suite.

The suite evaluates pure FedAvg, a FedProx proximal-coefficient sweep, Krum,
trimmed-mean, and the full structured pruning plus gradient sparsification
hybrid. It saves every run independently, selects the FedProx coefficient with
the highest final test accuracy, and writes shared plots and an analysis report.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(label: str, run_name: str, extra_args: list[str]) -> None:
    """Run one named simulation and stop the suite immediately on failure."""
    print("\n" + "=" * 80)
    print(f"Experiment: {label}")
    print("=" * 80)
    command = [
        sys.executable,
        str(ROOT / "run_simulation.py"),
        "--run_name",
        run_name,
        *extra_args,
    ]
    subprocess.run(command, check=True, cwd=ROOT)


def read_metrics(metrics_path: Path) -> list[dict[str, float]]:
    """Load the round-level metrics required by the comparison artifacts."""
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Expected metrics file was not created: {metrics_path}")

    with metrics_path.open(newline="", encoding="utf-8") as metrics_file:
        reader = csv.DictReader(metrics_file)
        required_columns = {"round", "test_accuracy", "avg_transmitted_ratio"}
        available_columns = set(reader.fieldnames or [])
        missing_columns = required_columns - available_columns
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{metrics_path} is missing required metric columns: {missing}")

        history = [
            {
                "round": float(row["round"]),
                "test_accuracy": float(row["test_accuracy"]),
                "avg_transmitted_ratio": float(row["avg_transmitted_ratio"]),
            }
            for row in reader
        ]

    if not history:
        raise ValueError(f"Metrics file contains no completed rounds: {metrics_path}")
    return history


def metric_summary(history: list[dict[str, float]]) -> dict[str, float]:
    """Return final and peak metrics used for selection and reporting."""
    return {
        "final_accuracy": history[-1]["test_accuracy"],
        "best_accuracy": max(row["test_accuracy"] for row in history),
        "final_transmitted_ratio": history[-1]["avg_transmitted_ratio"],
        "mean_transmitted_ratio": sum(
            row["avg_transmitted_ratio"] for row in history
        )
        / len(history),
    }


def format_mu(mu: float) -> str:
    """Create a stable, path-safe FedProx coefficient identifier."""
    return format(mu, ".12g").replace("-", "neg").replace(".", "p")


def write_fedprox_sweep_csv(
    sweep_rows: list[dict[str, float | str]],
    output_path: Path,
) -> None:
    """Save complete FedProx sweep outcomes in a thesis-friendly CSV file."""
    fieldnames = [
        "mu",
        "run_name",
        "final_test_accuracy",
        "best_test_accuracy",
        "final_avg_transmitted_ratio",
        "mean_avg_transmitted_ratio",
        "selected_for_comparison",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sweep_rows)


def save_comparison_plots(
    experiments: list[tuple[str, str]],
    fedprox_histories: list[tuple[float, list[dict[str, float]]]],
    runs_dir: Path,
    comparison_dir: Path,
) -> list[Path]:
    """Generate shared accuracy, communication, and FedProx-sweep plots."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Matplotlib is required to generate comparison plots.") from error

    histories = {
        label: read_metrics(runs_dir / run_name / "metrics.csv")
        for label, run_name in experiments
    }
    comparison_dir.mkdir(parents=True, exist_ok=True)

    def configure_axes(ylabel: str, title: str) -> None:
        plt.xlabel("Federated Round")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

    accuracy_path = comparison_dir / "accuracy_comparison.png"
    plt.figure(figsize=(9.5, 5.75))
    for label, _ in experiments:
        history = histories[label]
        plt.plot(
            [row["round"] for row in history],
            [row["test_accuracy"] * 100 for row in history],
            marker="o",
            linewidth=2.0,
            markersize=3.5,
            label=label,
        )
    configure_axes("Test Accuracy (%)", "Accuracy Comparison Across Federated Rounds")
    plt.savefig(accuracy_path, dpi=300, bbox_inches="tight")
    plt.close()

    transmission_path = comparison_dir / "transmitted_ratio_comparison.png"
    plt.figure(figsize=(9.5, 5.75))
    for label, _ in experiments:
        history = histories[label]
        plt.plot(
            [row["round"] for row in history],
            [row["avg_transmitted_ratio"] * 100 for row in history],
            marker="s",
            linewidth=2.0,
            markersize=3.5,
            label=label,
        )
    configure_axes(
        "Average Transmitted Delta Ratio (%)",
        "Communication Cost Comparison Across Federated Rounds",
    )
    plt.savefig(transmission_path, dpi=300, bbox_inches="tight")
    plt.close()

    fedprox_sweep_path = comparison_dir / "fedprox_mu_sweep.png"
    plt.figure(figsize=(9.5, 5.75))
    for mu, history in fedprox_histories:
        plt.plot(
            [row["round"] for row in history],
            [row["test_accuracy"] * 100 for row in history],
            marker="o",
            linewidth=2.0,
            markersize=3.5,
            label=rf"FedProx $\mu={mu:g}$",
        )
    configure_axes("Test Accuracy (%)", "FedProx Proximal-Coefficient Sweep")
    plt.savefig(fedprox_sweep_path, dpi=300, bbox_inches="tight")
    plt.close()

    return [accuracy_path, transmission_path, fedprox_sweep_path]


def write_analysis(
    output_path: Path,
    histories: dict[str, list[dict[str, float]]],
    selected_fedprox_mu: float,
    krum_f: int,
    trim_ratio: float,
    dirichlet_alpha: float,
    warmup_rounds: int,
    compression_ramp_rounds: int,
) -> None:
    """Write a concise, reproducible interpretation of all final results."""
    summaries = {label: metric_summary(history) for label, history in histories.items()}

    def result_line(label: str) -> str:
        summary = summaries[label]
        return (
            f"final accuracy **{summary['final_accuracy'] * 100:.2f}%** and final "
            f"transmitted ratio **{summary['final_transmitted_ratio'] * 100:.2f}%**"
        )

    report = f"""# FedPrune-Sparse Baseline Analysis

## Experimental Settings

All experiments use Dirichlet label-skew partitioning with `dirichlet_alpha={dirichlet_alpha:g}`.
The hybrid method uses `warmup_rounds={warmup_rounds}` before pruning and
sparsification are activated, followed by
`compression_ramp_rounds={compression_ramp_rounds}` rounds that linearly
increase their strength. Lower Dirichlet alpha values correspond to more
heterogeneous client label distributions.

## Results

**FedAvg.** Pure FedAvg uses no pruning, no sparsification, and weighted mean aggregation; it reached {result_line("FedAvg")}. This is the non-robust, non-proximal reference for measuring the effects of client drift, robust aggregation, and compression.

**FedProx.** The automatic sweep selected `mu={selected_fedprox_mu:g}` by highest final test accuracy; this selected run reached {result_line("FedProx")}. FedProx adds a proximal penalty to limit local-model drift from the global model, so it is the standard baseline for assessing optimization under non-IID data.

**Krum.** Krum used `krum_f={krum_f}` and reached {result_line("Krum")}. Krum applies the single client delta with the best neighbor-distance score rather than averaging all client deltas. This protects against outlier or malicious updates, but in label-skew non-IID training it can trade clean-data accuracy for robustness because each round learns from only one selected client update.

**Trimmed Mean.** Coordinate-wise trimmed mean used `trim_ratio={trim_ratio:g}` per tail and reached {result_line("Trimmed Mean")}. It discards extreme parameter values before averaging, providing robustness to outlier updates while retaining contributions from multiple clients.

**FedPrune-Sparse Hybrid.** The hybrid uses structured adaptive pruning and cost-weighted gradient sparsification with `warmup_rounds={warmup_rounds}` and `compression_ramp_rounds={compression_ramp_rounds}`; it reached {result_line("FedPrune-Sparse Hybrid")}. The warm-up and ramp avoid applying the strongest compression before the global model has established a useful representation, making the accuracy-versus-communication trade-off more interpretable.
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    """Execute the complete baseline suite and generate shared artifacts."""
    parser = argparse.ArgumentParser(
        description="Run FedPrune-Sparse baseline and hybrid comparison experiments."
    )
    parser.add_argument(
        "--num_rounds",
        "--rounds",
        dest="num_rounds",
        type=int,
        default=50,
        help="Number of federated rounds for every suite experiment.",
    )
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--clients_per_round", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument(
        "--dirichlet_alpha",
        type=float,
        default=0.3,
        help="Label-skew severity passed to run_simulation.py; lower is more non-IID.",
    )
    parser.add_argument(
        "--warmup_rounds",
        type=int,
        default=5,
        help="Hybrid compression warm-up rounds passed to run_simulation.py.",
    )
    parser.add_argument(
        "--compression_ramp_rounds",
        type=int,
        default=5,
        help="Hybrid rounds used to linearly increase compression after warm-up.",
    )
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument(
        "--comparison_name",
        type=str,
        default="comparison",
        help="Subdirectory under results_dir for all comparison outputs.",
    )
    parser.add_argument(
        "--fedprox_mu_values",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.1, 1.0],
        help="FedProx proximal coefficients to sweep.",
    )
    parser.add_argument(
        "--krum_f",
        type=int,
        default=1,
        help="Assumed malicious/outlier client count used by Krum.",
    )
    parser.add_argument(
        "--trim_ratio",
        type=float,
        default=0.1,
        help="Per-tail trimming fraction used by trimmed mean.",
    )
    args, unknown = parser.parse_known_args()

    if args.num_rounds <= 0:
        parser.error("--num_rounds must be positive.")
    if args.dirichlet_alpha <= 0:
        parser.error("--dirichlet_alpha must be positive.")
    if args.warmup_rounds < 0:
        parser.error("--warmup_rounds cannot be negative.")
    if args.compression_ramp_rounds < 0:
        parser.error("--compression_ramp_rounds cannot be negative.")
    if not args.fedprox_mu_values or any(mu <= 0 for mu in args.fedprox_mu_values):
        parser.error("--fedprox_mu_values must contain one or more positive values.")
    if args.krum_f < 0:
        parser.error("--krum_f cannot be negative.")
    if not 0.0 <= args.trim_ratio < 0.5:
        parser.error("--trim_ratio must be in [0.0, 0.5).")

    comparison_dir = Path(args.results_dir) / args.comparison_name
    runs_dir = comparison_dir / "runs"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    common = [
        "--rounds",
        str(args.num_rounds),
        "--num_clients",
        str(args.num_clients),
        "--clients_per_round",
        str(args.clients_per_round),
        "--local_epochs",
        str(args.local_epochs),
        "--dirichlet_alpha",
        str(args.dirichlet_alpha),
        "--results_dir",
        str(runs_dir),
        *unknown,
    ]
    clean_baseline_args = [
        "--no_pruning",
        "--no_sparsification",
        "--no_adaptive_ratio",
    ]

    run(
        "Baseline FedAvg without pruning or sparsification",
        "baseline_fedavg",
        common + clean_baseline_args + ["--aggregation", "fedavg"],
    )

    fedprox_histories: list[tuple[float, list[dict[str, float]]]] = []
    fedprox_rows: list[dict[str, float | str]] = []
    for mu in args.fedprox_mu_values:
        run_name = f"fedprox_mu_{format_mu(mu)}"
        run(
            f"Pure FedProx sweep (mu={mu:g}) without pruning or sparsification",
            run_name,
            common
            + clean_baseline_args
            + [
                "--aggregation",
                "fedavg",
                "--baseline_fedprox_mu",
                str(mu),
            ],
        )
        history = read_metrics(runs_dir / run_name / "metrics.csv")
        fedprox_histories.append((mu, history))
        summary = metric_summary(history)
        fedprox_rows.append(
            {
                "mu": mu,
                "run_name": run_name,
                "final_test_accuracy": summary["final_accuracy"],
                "best_test_accuracy": summary["best_accuracy"],
                "final_avg_transmitted_ratio": summary["final_transmitted_ratio"],
                "mean_avg_transmitted_ratio": summary["mean_transmitted_ratio"],
                "selected_for_comparison": "false",
            }
        )

    selected_index = max(
        range(len(fedprox_rows)),
        key=lambda index: float(fedprox_rows[index]["final_test_accuracy"]),
    )
    fedprox_rows[selected_index]["selected_for_comparison"] = "true"
    selected_fedprox_mu = float(fedprox_rows[selected_index]["mu"])
    selected_fedprox_run = str(fedprox_rows[selected_index]["run_name"])
    sweep_csv_path = comparison_dir / "fedprox_mu_sweep.csv"
    write_fedprox_sweep_csv(fedprox_rows, sweep_csv_path)

    run(
        "Krum robust aggregation without pruning or sparsification",
        "baseline_krum",
        common
        + clean_baseline_args
        + [
            "--aggregation",
            "krum",
            "--krum_f",
            str(args.krum_f),
        ],
    )
    run(
        "Trimmed-mean robust aggregation without pruning or sparsification",
        "baseline_trimmed_mean",
        common
        + clean_baseline_args
        + [
            "--aggregation",
            "trimmed_mean",
            "--trim_ratio",
            str(args.trim_ratio),
        ],
    )
    run(
        "Hybrid structured adaptive pruning with cost-weighted sparsification",
        "hybrid_structured_cwmp",
        common
        + [
            "--pruning_mode",
            "structured",
            "--pruning_ratio",
            "0.4",
            "--adaptive_ratio",
            "--sparsify_method",
            "cost_weighted",
            "--sparsity_ratio",
            "0.95",
            "--warmup_rounds",
            str(args.warmup_rounds),
            "--compression_ramp_rounds",
            str(args.compression_ramp_rounds),
            "--aggregation",
            "fedavg",
        ],
    )

    experiments = [
        ("FedAvg", "baseline_fedavg"),
        (f"FedProx (mu={selected_fedprox_mu:g})", selected_fedprox_run),
        ("Krum", "baseline_krum"),
        ("Trimmed Mean", "baseline_trimmed_mean"),
        ("FedPrune-Sparse Hybrid", "hybrid_structured_cwmp"),
    ]
    saved_paths = save_comparison_plots(
        experiments,
        fedprox_histories,
        runs_dir,
        comparison_dir,
    )
    analysis_histories = {
        "FedAvg": read_metrics(runs_dir / "baseline_fedavg" / "metrics.csv"),
        "FedProx": read_metrics(runs_dir / selected_fedprox_run / "metrics.csv"),
        "Krum": read_metrics(runs_dir / "baseline_krum" / "metrics.csv"),
        "Trimmed Mean": read_metrics(runs_dir / "baseline_trimmed_mean" / "metrics.csv"),
        "FedPrune-Sparse Hybrid": read_metrics(
            runs_dir / "hybrid_structured_cwmp" / "metrics.csv"
        ),
    }
    analysis_path = comparison_dir / "analysis.md"
    write_analysis(
        analysis_path,
        analysis_histories,
        selected_fedprox_mu,
        args.krum_f,
        args.trim_ratio,
        args.dirichlet_alpha,
        args.warmup_rounds,
        args.compression_ramp_rounds,
    )

    print(f"\nSelected FedProx coefficient: mu={selected_fedprox_mu:g}")
    print(f"FedProx sweep results: {sweep_csv_path}")
    print(f"Analysis report: {analysis_path}")
    print("Comparison plots saved:")
    for path in saved_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()

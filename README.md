# FedPrune-Sparse

FedPrune-Sparse is a reproducible federated learning simulation for studying the joint effect of client-side model pruning and client-to-server delta sparsification under non-IID data. The project is structured for thesis/dissertation experiments: each run stores configuration, detailed English logs, round-level metrics, model checkpoints, and publication-ready figures under `results/`.

Languages: English | [فارسی](README.fa.md)

## Research Objective

Standard FedAvg can be expensive for resource-constrained clients because every selected client trains the full model and transmits a dense update. FedPrune-Sparse separates these two costs:

| Cost Source | Mechanism |
|---|---|
| Local computation | Prune the client model before local training. |
| Uplink communication | Sparsify the local model delta before transmission. |

This separation supports controlled ablations:

- Plain FedAvg.
- FedProx without pruning or sparsification.
- Krum and trimmed-mean robust aggregation without pruning or sparsification.
- Hybrid pruning plus sparsification.

## Federated Round Flow

```text
Global model broadcast by server
        │
        ▼
Client-side model copy
        │
        ▼
Optional model pruning
        │
        ▼
Local training on non-IID client data
        │
        ▼
Delta = local weights - global weights
        │
        ▼
Optional delta sparsification with error feedback
        │
        ▼
Compressed delta sent to server
        │
        ▼
Selected aggregation and global model update
```

## Project Structure

```text
fedprune-sparse/
├── client/
│   └── client.py
├── data/
│   └── MNIST/
├── models/
│   └── cnn.py
├── scripts/
│   └── compare_ablations.py
├── server/
│   └── server.py
├── utils/
│   ├── gradient_sparsification.py
│   └── model_pruning.py
├── run_simulation.py
├── requirements.txt
├── README.md
└── README.fa.md
```

| Path | Purpose |
|---|---|
| `models/cnn.py` | Compact MNIST CNN designed for transparent structured pruning. |
| `utils/model_pruning.py` | Unstructured pruning, structured pruning, and adaptive per-client pruning ratios. |
| `utils/gradient_sparsification.py` | Top-K, random, and cost-weighted sparsification with error feedback. |
| `client/client.py` | Client workflow: pruning, FedProx-capable local training, delta computation, and sparsification. |
| `server/server.py` | Global model broadcast, FedAvg/Krum/trimmed-mean aggregation, and evaluation. |
| `run_simulation.py` | End-to-end experiment runner with logs, metrics, checkpoints, and plots. |
| `scripts/compare_ablations.py` | Automated 50-round baseline, FedProx-sweep, and hybrid comparison suite. |

## Implemented Methods

### Model Pruning

- **Unstructured pruning:** masks low-magnitude weights in `Conv2d` and `Linear` layers while preserving tensor shapes.
- **Structured pruning:** removes complete filters or hidden neurons and rebuilds a smaller `SimpleCNN`, reducing local memory/FLOP cost without requiring sparse tensor backends.
- **Adaptive pruning:** assigns client-specific pruning ratios either from synthetic capability profiles or online round-time feedback.

### Delta Sparsification

- **Top-K:** keeps the largest absolute delta coordinates.
- **Random:** keeps random coordinates as a baseline.
- **Cost-weighted:** ranks coordinates by `abs(delta) / cost`, with higher default costs for fully connected parameters.
- **Error feedback:** stores unsent delta residuals locally and reuses them in future rounds.

### Standard Federated Learning Baselines

- **FedProx:** adds the proximal penalty
  `mu / 2 * ||w_local - w_global||^2` to each local loss. It is a standard
  baseline for reducing client drift under non-IID data while retaining the
  FedAvg communication pattern.
- **Krum:** selects the client delta with the lowest sum of squared Euclidean
  distances to its nearest client updates. It is a robust aggregation baseline
  for resisting malicious or outlier client updates. Krum requires at least
  `2 * f + 3` selected client updates, where `f` is `--krum_f`.
- **Trimmed mean:** removes the lowest and highest values per parameter
  coordinate before averaging. This robust aggregation baseline reduces the
  influence of outlier or malicious updates.

All aggregation strategies honor contribution masks from structured pruning:
clients only influence parameters they retained and trained.

### Non-IID Partitioning and Compression Schedule

- **Dirichlet label skew:** the default `--partition_strategy dirichlet` samples
  label proportions for each client from a Dirichlet distribution.
  `--dirichlet_alpha 0.3` is deliberately heterogeneous; lower positive values
  create stronger non-IID label skew. Use `--partition_strategy shard` for the
  legacy shard-based partitioning behavior.
- **Warm-up and ramp:** `--warmup_rounds` disables pruning and sparsification
  initially. `--compression_ramp_rounds` then linearly increases their
  configured strengths. Both default to `0`, preserving ordinary-run behavior.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use a CUDA GPU, install the PyTorch build that matches your CUDA version before installing the remaining dependencies.

## Quick Start

Run the default dissertation-oriented hybrid experiment:

```bash
python run_simulation.py
```

Run a faster smoke experiment:

```bash
python run_simulation.py \
  --rounds 2 \
  --num_clients 4 \
  --clients_per_round 2 \
  --local_epochs 1 \
  --run_name smoke_test
```

Run a plain FedAvg baseline:

```bash
python run_simulation.py \
  --no_pruning \
  --no_sparsification \
  --no_adaptive_ratio \
  --run_name baseline_fedavg
```

Run pure FedProx:

```bash
python run_simulation.py \
  --no_pruning \
  --no_sparsification \
  --no_adaptive_ratio \
  --baseline_fedprox_mu 0.01 \
  --run_name baseline_fedprox
```

Run Krum robust aggregation:

```bash
python run_simulation.py \
  --no_pruning \
  --no_sparsification \
  --no_adaptive_ratio \
  --aggregation krum \
  --krum_f 1 \
  --clients_per_round 5 \
  --run_name baseline_krum
```

Run the complete baseline and hybrid comparison suite:

```bash
python scripts/compare_ablations.py
```

The suite defaults to 50 rounds and runs FedAvg, a FedProx
`mu={0.001, 0.01, 0.1, 1.0}` sweep, Krum, trimmed mean, and the hybrid. It
selects the FedProx coefficient with the greatest final accuracy for the main
comparison. The hybrid defaults to five uncompressed warm-up rounds followed
by a five-round compression ramp.

```bash
python scripts/compare_ablations.py \
  --num_rounds 50 \
  --dirichlet_alpha 0.3 \
  --warmup_rounds 5 \
  --compression_ramp_rounds 5
```

To satisfy Krum, keep `--clients_per_round` at least `2 * f + 3`, where `f`
is `--krum_f`.

## Results and Logs

Each run creates a subdirectory under `results/`. If `--run_name` is provided, it is used as the subdirectory name; otherwise a timestamped name is generated.

Example:

```bash
python run_simulation.py --run_name hybrid_structured_cwmp
```

Output directory:

```text
results/hybrid_structured_cwmp/
├── config.json
├── metrics.csv
├── run.log
├── summary.json
├── final_model.pt
├── accuracy_curve.png
├── compression_pruning_curve.png
├── round_time_curve.png
└── experiment_dashboard.png
```

| File | Description |
|---|---|
| `config.json` | Full command-line configuration for the run. |
| `metrics.csv` | Round-by-round accuracy, compression, pruning, and timing metrics. |
| `run.log` | Detailed English execution log with per-round and per-client statistics. |
| `summary.json` | Final and best metrics plus artifact paths. |
| `final_model.pt` | Final global model checkpoint. |
| `*.png` | High-resolution plots saved at 300 DPI for analysis and reporting. |

## Key Metrics

| Metric | Meaning |
|---|---|
| `test_accuracy` | Global model accuracy on the MNIST test set. |
| `avg_transmitted_ratio` | Average fraction of nonzero delta coordinates transmitted by selected clients. |
| `avg_pruning_ratio` | Average pruning ratio among selected clients. |
| `avg_sparsity_ratio` | Effective delta sparsity ratio after warm-up and compression ramp. |
| `avg_round_time_sec` | Average local update time for selected clients. |

## Important Arguments

| Argument | Default | Description |
|---|---:|---|
| `--rounds` | `20` | Number of federated rounds. |
| `--num_clients` | `20` | Total simulated clients. |
| `--clients_per_round` | `10` | Selected clients per round. |
| `--local_epochs` | `2` | Local epochs per selected client. |
| `--partition_strategy` | `dirichlet` | `dirichlet` label skew or legacy `shard` partitioning. |
| `--dirichlet_alpha` | `0.3` | Dirichlet concentration; lower positive values are more non-IID. |
| `--min_samples_per_client` | `10` | Minimum accepted client samples for Dirichlet partitioning. |
| `--pruning_mode` | `structured` | `structured` or `unstructured`. |
| `--pruning_ratio` | `0.4` | Base pruning ratio. |
| `--adaptive_ratio` | enabled | Enables per-client pruning ratios. |
| `--no_adaptive_ratio` | disabled | Disables adaptive pruning ratios. |
| `--sparsify_method` | `cost_weighted` | `topk`, `random`, or `cost_weighted`. |
| `--sparsity_ratio` | `0.95` | Fraction of delta coordinates to zero before transmission. |
| `--no_error_feedback` | disabled | Disables residual error feedback. |
| `--warmup_rounds` | `0` | Initial rounds without pruning or sparsification. |
| `--compression_ramp_rounds` | `0` | Rounds used to linearly activate compression after warm-up. |
| `--baseline_fedprox_mu` | `0.0` | FedProx proximal coefficient; `0.0` preserves ordinary local SGD. |
| `--aggregation` | `fedavg` | Server aggregation: `fedavg`, `krum`, or `trimmed_mean`. |
| `--krum_f` | `0` | Assumed number of malicious/outlier clients for Krum. |
| `--trim_ratio` | `0.0` | Fraction removed from each value tail for trimmed mean; must be below `0.5`. |
| `--results_dir` | `./results` | Output directory for all artifacts. |
| `--run_name` | timestamp | Optional run subdirectory name. |
| `--device` | `cpu` | Training/evaluation device, e.g. `cuda`. |

## Reproducibility Notes

- The random seed controls data shard assignment, client sampling, model initialization, and random sparsification.
- `config.json` records all experiment settings.
- `metrics.csv` is suitable for downstream statistical analysis.
- `run.log` contains detailed English logs for auditing each experiment.
- The ablation script uses consistent shared arguments so methods are compared under the same global configuration.
- `scripts/compare_ablations.py` writes individual runs to
  `results/comparison/runs/` and saves `accuracy_comparison.png` and
  `transmitted_ratio_comparison.png` at 300 DPI in `results/comparison/`.
- The suite also writes `fedprox_mu_sweep.csv`, `fedprox_mu_sweep.png`, and
  `analysis.md` to `results/comparison/`.

The comparison output structure is:

```text
results/comparison/
├── analysis.md
├── accuracy_comparison.png
├── fedprox_mu_sweep.csv
├── fedprox_mu_sweep.png
├── transmitted_ratio_comparison.png
└── runs/
    ├── baseline_fedavg/
    ├── baseline_krum/
    ├── baseline_trimmed_mean/
    ├── fedprox_mu_*/
    └── hybrid_structured_cwmp/
```

## Citation-Oriented Summary

FedPrune-Sparse evaluates a hybrid resource-aware federated learning pipeline in which structured or unstructured pruning reduces local client computation, while sparse delta communication with error feedback reduces uplink bandwidth. The project compares this hybrid method against standard FedAvg, FedProx for non-IID client drift, and Krum/trimmed-mean robust aggregation for outlier resistance. Structured-pruned local updates are expanded to global tensor shape and protected by contribution masks before aggregation.

# Lagos Road Congestion Model

A reproducible simulation study comparing temporal forecasting, autoencoder anomaly features, graph adjustment and active learning for a 20-segment Lagos study network.

## Full reviewer-requested benchmarks

The completed extension includes **STGCN, DCRNN and Graph WaveNet**, each fitted across the same five seeds. Read [the benchmark report](REVIEWER_BENCHMARK_RESULTS.md), [the fixed protocol](BENCHMARK_PROTOCOL.md), and [the draft reviewer response](REVIEWER_1_BENCHMARK_RESPONSE.md).

| Full architecture | Validation accuracy (%) | Validation MAE, 0-100 index | Validation RMSE, 0-100 index |
| --- | ---: | ---: | ---: |
| STGCN | 73.9881 +/- 0.7274 | 4.6555 +/- 0.0259 | 6.2913 +/- 0.0078 |
| DCRNN | 73.6774 +/- 0.4017 | 4.6942 +/- 0.0253 | 6.3021 +/- 0.0122 |
| Graph WaveNet | 73.8642 +/- 0.3388 | 4.7072 +/- 0.0393 | 6.3481 +/- 0.0126 |

The extension contains 15 fitted full-architecture checkpoints, 30 prediction files, matched CPU model profiling and seven passing architecture/artifact checks. The primary comparison uses fixed validation; reuse of the earlier temporal holdout is labeled retrospective supplementary evaluation. Original study results and selection remain recorded below.

```powershell
.venv/Scripts/python.exe run_reviewer_benchmarks.py --prepare
.venv/Scripts/python.exe -u run_reviewer_benchmarks.py
.venv/Scripts/python.exe validate_reviewer_benchmarks.py
```

## Results

The experiment uses a fixed reconstructed simulation dataset and five model seeds: **42, 123, 456, 789, 1024**. A pre-specified validation-MAE rule selected the plain LSTM for final temporal evaluation.

| Held-out metric | Mean +/- sample SD |
| --- | ---: |
| Ordinal-class accuracy | 74.8167 +/- 0.4406% |
| MAE, 0-100 index | 4.7409 +/- 0.0217 |
| RMSE, 0-100 index | 6.3843 +/- 0.0199 |

The full evaluation includes matched random-acquisition controls, 50 degree-preserving graph rewires, a reduced STGCN comparator and CPU inference profiling. All **23 study checks passed**; an independent full-size seed-42 training repeat reproduced validation predictions exactly.

Read [the complete experimental report](REVISION_RESULTS.md) for comparisons and interpretation. These are reconstructed-data simulation results, with ordinal hourly chronology. The source audit is retained alongside the completed study.

## Data and evaluation

- 20 road-segment nodes and 8,432 generated hourly observations per node.
- Twelve-hour inputs produce **8,420 network snapshots**, or **168,400 segment targets**.
- Snapshot split: **4,420 training / 589 validation / 885 acquisition / 2,526 final test**.
- Each 295-snapshot acquisition exposes 5,900 simulated segment labels.
- Preprocessing is fitted on initial-training inputs; the final test follows model selection.

## Repository contents

| Path | Contents |
| --- | --- |
| [REVISION_RESULTS.md](REVISION_RESULTS.md) | Methods, measured results and interpretation |
| [README_REVISION.md](README_REVISION.md) | Execution guide and prediction schema |
| [reconstructed_v1/DATA_DICTIONARY.md](reconstructed_v1/DATA_DICTIONARY.md) | Feature definitions, array shapes and units |
| `reconstructed_v1/protocol/` | Fixed configurations, split indices and hashes |
| `reconstructed_v1/data/` | Model-ready dataset and preprocessing parameters |
| `reconstructed_v1/checkpoints/` | Fitted model weights |
| `reconstructed_v1/predictions/` | Saved seed-specific predictions |
| `reconstructed_v1/results/` | Metrics, statistical comparisons and figures |
| `reconstructed_v1/logs/` | Epoch-level fitting records |
| `Traffic_Congestion_Reproducibility_Package/` | Supplied reconstructed source material |
| `protocol/`, `results/`, `REVISION_AUDIT_INITIAL.md` | Initial source-data audit |

## Setup and verification

The recorded execution environment used Python 3.14.7 and CPU PyTorch. Exact installed versions are in [requirements.installed.txt](requirements.installed.txt).

```powershell
py -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.installed.txt
.venv/Scripts/python.exe reconstructed_study.py verify
.venv/Scripts/python.exe run_reconstructed_checks.py
```

The checks inspect the supplied fitted artifacts, prediction alignment, acquisition accounting and independent-repeat record. Initial source-archive audit tests are a separate scope from the completed study gate.

## Training and report generation

```powershell
.venv/Scripts/python.exe -u run_reconstructed_experiments.py
.venv/Scripts/python.exe summarize_reconstructed_study.py
```

The distributed protocol is already prepared and locked. Completed checkpoints support resumption. Run a single training process at a time and retain the final-test marker. Full CPU training can take several hours; saved results can be inspected without refitting. The compiler creates a local distribution ZIP; duplicate ZIP files and local Python environments are excluded from Git.

## Figures

- [Equal-budget acquisition comparison](reconstructed_v1/results/active_learning/label_efficiency_plot.pdf)
- [Study data accounting](reconstructed_v1/results/data_audit/audit_flowchart.pdf)

The study evaluates forecasting and model inference. Dashboard and routing application tests are separate evaluation tasks.

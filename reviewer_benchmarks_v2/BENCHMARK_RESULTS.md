# Reviewer benchmark comparison: STGCN, DCRNN and Graph WaveNet

This extension evaluates all three architecture families requested by Reviewer 1 using the fixed 20-node simulation dataset and the five predefined model seeds. It supplements the completed study with full architectures while retaining the earlier results and original model-selection decision.

## Architecture and training specification

- **STGCN:** two spatiotemporal blocks with channel paths 14-32-64 and 64-32-128, width-three temporal convolutions, order-two Chebyshev graph filters (three polynomial terms), residual channel alignment, node/channel layer normalization, dropout 0.3 and a four-step output convolution. The output uses all 12 real input hours.
- **DCRNN:** a two-layer encoder and two-layer decoder, each with 64 diffusion-GRU units; forward and reverse directed random-walk supports; diffusion order two. Each direction uses its own Chebyshev polynomial recurrence. A zero GO symbol starts the one-step decoder; scheduled sampling concerns later decoder steps and is inapplicable to this horizon.
- **Graph WaveNet:** the authors' four-block, two-layer-per-block architecture, 32 residual/dilation channels, 256 skip channels, 512 output-hidden channels, diffusion order two, forward/reverse supports and learned adaptive adjacency from 10-dimensional node embeddings. Its 13-step receptive field receives one left-padding step plus the 12 observed hours.

All models receive the same 14 scaled features and forecast the same next-hour congestion score. They use the initial 4,420 training snapshots, fixed 589-snapshot validation set, Adam with learning rate 0.0005, unmasked MSE loss on the 0-10 scale, batches of 32 network snapshots, maximum 80 epochs, patience 10 and gradient clipping at norm 5. Best raw validation-MSE checkpoints are retained. Hyperparameters were fixed before training; no outcome-driven search was performed. These are architecture-preserving task adaptations to a multivariate one-hour congestion target, with a common training protocol rather than the original papers' dataset-specific optimization recipes.

A1 is the plain LSTM; A4 is AE-LSTM-GCN before acquisition; A5 is the final entropy-acquisition pipeline. A1, A4 and all three benchmarks use 4,420 supervised training snapshots. A5 additionally uses the 885 acquisition snapshots, so its comparison represents the refined system at a larger label budget.

## Primary validation comparison

Values are mean +/- sample SD across seeds 42, 123, 456, 789 and 1024. Accuracy is percent; errors are points on the 0-100 reporting index.

| Model | Accuracy (%) | MAE (index) | RMSE (index) |
|---|---:|---:|---:|
| A1 | 73.8506 +/- 0.4815 | 4.6795 +/- 0.0195 | 6.3025 +/- 0.0098 |
| A4 | 73.7284 +/- 0.1237 | 4.9974 +/- 0.0679 | 6.6785 +/- 0.0924 |
| A5 | 74.1138 +/- 0.4720 | 4.9004 +/- 0.0425 | 6.5065 +/- 0.0515 |
| STGCN | 73.9881 +/- 0.7274 | 4.6555 +/- 0.0259 | 6.2913 +/- 0.0078 |
| DCRNN | 73.6774 +/- 0.4017 | 4.6942 +/- 0.0253 | 6.3021 +/- 0.0122 |
| GraphWaveNet | 73.8642 +/- 0.3388 | 4.7072 +/- 0.0393 | 6.3481 +/- 0.0126 |

Among the three requested architectures, **STGCN** achieved the lowest mean validation MAE, **4.6555** index points. Full comparisons and effect sizes appear in `results/validation_paired_tests.csv`. Two-sided paired t tests use five model-seed pairs; Holm correction covers all 27 validation comparisons (three existing pipelines x three benchmarks x three metrics).

- A1 vs STGCN: MAE difference +0.0240 index points (higher for A1); Holm-adjusted p=1.
- A1 vs DCRNN: MAE difference -0.0146 index points (lower for A1); Holm-adjusted p=1.
- A1 vs GraphWaveNet: MAE difference -0.0277 index points (lower for A1); Holm-adjusted p=1.
- A4 vs STGCN: MAE difference +0.3419 index points (higher for A4); Holm-adjusted p=0.004749.
- A4 vs DCRNN: MAE difference +0.3032 index points (higher for A4); Holm-adjusted p=0.01844.
- A4 vs GraphWaveNet: MAE difference +0.2902 index points (higher for A4); Holm-adjusted p=0.0171.
- A5 vs STGCN: MAE difference +0.2450 index points (higher for A5); Holm-adjusted p=0.005024.
- A5 vs DCRNN: MAE difference +0.2063 index points (higher for A5); Holm-adjusted p=0.01844.
- A5 vs GraphWaveNet: MAE difference +0.1932 index points (higher for A5); Holm-adjusted p=0.06422.

Positive MAE differences favor the benchmark; negative differences favor the first named pipeline. Statistical interpretation is limited to the fixed simulation and five fitting seeds.

## Supplementary temporal-test comparison

The original v1 temporal holdout had already been examined before this extension was defined. It is therefore reported as a **retrospective supplementary comparison**, distinct from the original selected-model held-out evaluation. All 15 benchmark fits and validation epoch selections finished before these test predictions were computed. These results do not alter the original A1 selection.

| Model | Accuracy (%) | MAE (index) | RMSE (index) |
|---|---:|---:|---:|
| A1 | 74.8167 +/- 0.4406 | 4.7409 +/- 0.0217 | 6.3843 +/- 0.0199 |
| STGCN | 74.9220 +/- 0.4252 | 4.7042 +/- 0.0201 | 6.3530 +/- 0.0134 |
| DCRNN | 74.7043 +/- 0.2693 | 4.7401 +/- 0.0235 | 6.3603 +/- 0.0121 |
| GraphWaveNet | 74.6512 +/- 0.3268 | 4.7571 +/- 0.0387 | 6.4107 +/- 0.0195 |

Supplementary paired tests comparing A1 with each benchmark are corrected as a separate nine-test family in `results/retrospective_test_paired_tests.csv`. A4 and A5 retain their development-only evaluation in the original protocol.

## Matched CPU model profiling

All six configurations were timed using the same normalized 12-hour input tensor, CPU, four PyTorch threads, float32, 20 warmups and 100 measured repetitions. A batch of one represents one 20-node network update. Timings include all model components for A4/A5 and exclude data ingestion and preprocessing; they are separate from the earlier JSON-inclusive A4 timings.

| Model | Parameters | Mean model time (ms), mean +/- SD over seeds | Mean seed-wise p95 (ms) |
|---|---:|---:|---:|
| A1 | 205,953 | 4.011 +/- 0.862 | 7.437 |
| A4 | 211,960 | 7.692 +/- 2.458 | 15.004 |
| A5 | 211,960 | 5.563 +/- 1.656 | 7.896 |
| STGCN | 202,772 | 5.831 +/- 5.091 | 13.209 |
| DCRNN | 383,873 | 41.495 +/- 13.493 | 66.861 |
| GraphWaveNet | 300,401 | 23.339 +/- 6.086 | 30.725 |

Batch-one and batch-20 measurements are in `results/matched_model_timing.csv`; fit times, executed epochs and selected epochs are in `results/validation_by_seed.csv`.

## Implementation provenance and checks

Reference sources: [STGCN](https://github.com/VeritasYin/STGCN_IJCAI-18), [DCRNN](https://github.com/liyaguang/DCRNN), [Graph WaveNet](https://github.com/nnzhan/Graph-WaveNet). Pinned commits:

- STGCN: `4ab75ab69af354d6a0ab86aefa082d389b187127`
- DCRNN: `602afd9d767d3aa1c9b3eac51710d6aeee12c227`
- GraphWaveNet: `6b162e80c59a1d494809252eca055cff93dc66b1`

STGCN and DCRNN are PyTorch ports of the referenced architecture families. The DCRNN polynomial basis is reset independently for each directional support. Graph WaveNet retains the authors' model with three Conv1d declarations replaced by Conv2d for compatibility with its four-dimensional tensors and two-dimensional kernels. Input and output dimensions are adapted to this study. Reference files, source hashes, patches and licenses are preserved in `benchmark_vendor/`.

**Seven checks passed:** four architecture/operator checks and three completed-artifact checks. These cover finite forward/backward computation, earliest-input influence, adaptive-adjacency gradients, directed supports, numerical diffusion/Chebyshev basis calculations, seeded initialization, all training IDs, prediction alignment, recomputed metrics and replay of all 15 fitted checkpoints. Check records are in `results/final_checks.log`.

The protocol, code fingerprints, epoch logs, 15 fitted checkpoints and 30 full-schema prediction files are retained in `reviewer_benchmarks_v2/`. Source data, feature scaling, original splits and the original study's selected-model predictions remain fixed.

Epoch durations are wall-clock execution records and include long host-execution pauses in some fits. Use the separately measured model-only timing table for computational profiling.

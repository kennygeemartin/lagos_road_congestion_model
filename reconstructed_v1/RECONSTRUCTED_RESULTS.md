# Traffic-congestion forecasting: experimental methods and results

This study develops and evaluates a reproducible traffic-congestion forecasting framework using a reconstructed simulation dataset. Five model seeds, matched acquisition budgets and graph controls support the comparison of temporal, anomaly-aware and spatial modelling components. Source provenance and the initial data audit are documented in `REVISION_AUDIT_INITIAL.md`.

Artifact paths refer to `reconstructed_v1/`, except where a root-relative path is stated. The root `results/` directory contains the source-data audit.

## Principal findings

The five-seed comparison selected A1 (plain LSTM) using the pre-specified mean validation-MAE criterion. Held-out accuracy was **74.8167 +/- 0.4406%**, MAE **4.7409 +/- 0.0217** and RMSE **6.3843 +/- 0.0199** on the 0-100 index. The main contribution is a reproducible evaluation of the integrated architecture and its component trade-offs under the defined simulation conditions.

Pipeline key: **A1** = LSTM; **A2** = AE + LSTM; **A3** = LSTM + GCN; **A4** = AE + LSTM + GCN; **A5** = A4 after all three entropy-acquisition stages. **G0** is the matched no-neighbor head, **G1** the original graph and **G2** the degree-preserving rewired controls.

Verification completed with **23 checks passed**. Independent seed-42 A1 retraining reproduced all saved validation predictions exactly (maximum absolute difference 0.0). The initial source-data audit is retained separately for provenance.

## Study design and data

- Fixed reconstructed simulation data; five model and selector seeds: [42, 123, 456, 789, 1024]. These seeds vary model fitting and sample selection while holding the generated dataset constant.
- 8,432 ordinal proxy hours per segment, 20 segments, 168,640 generated node-hours.
- 12-hour inputs and next-step target: **8,420 network snapshots = 168,400 node targets**.
- Snapshot split: 4,420 train / 589 validation / 885 acquisition / 2,526 test. Node-target counts: 88,400 / 11,780 / 17,700 / 50,520.
- A 295-snapshot query exposes 5,900 scalar labels; cumulative budgets are 0/295/590/885 snapshots or 0/5,900/11,800/17,700 scalar labels.
- Hourly sensor means, causal forward fill, train-node median initialization and 14 documented features. Chronology uses the reconstructed ordinal hourly index.
- Unique initial-training input hours 0..4430 fit the scaler, imputation medians and proxy quantiles. AE early stopping uses an internal training holdout; its evaluation threshold is fixed at 0.015.
- The model architecture/maximum epochs and learning rates match the archived configuration. LSTM batch 32 means 32 network snapshots (640 shared-LSTM segment sequences); GCN batch 64 means 64 snapshots.
- GCN adjacency is strictly binary symmetric support plus self-loops, symmetrically normalized. The routing graph remains directed. Rewires preserve binary graph degrees and connectedness. SCS always uses original graph edges.
- Features, in order: speed_kmh, travel_delay_min, acceleration_ms2, gps_speed_noise_kmh, sensor_count, hour_sin, hour_cos, weekday_sin, weekday_cos, peak_indicator, rolling_ratio_mean_12, rolling_ratio_std_12, volume_intensity_veh_h_per_km, volume_capacity_ratio.

The experimental input is the supplied reconstructed segment dataset. Original Nigerian route-context records and the original Metro time series were outside the available experimental inputs. Hour and day indicators follow the reconstructed ordinal chronology. Each forecast uses past observations, with all fitted preprocessing parameters fixed from the initial training partition.

The GCN uses `B_ij = 1[A_ij + A_ji > 0]`, `A_hat = B + I`, `d_i = sum_j A_hat_ij`, and `A_norm_ij = A_hat_ij / sqrt(d_i*d_j)`. Each neighborhood includes the node itself once. Graph layers compute `ReLU(A_norm H W + b)`, followed by training-time dropout and a linear node readout. A separate directed graph represents routing transitions.

## Development ablation: validation only

Mean +/- sample SD across five seeds. Accuracy is percent; errors are 0-100 index points. Divide errors by ten for the 0-10 equivalents, which are also in the CSVs.

| Pipeline | Accuracy (%) | MAE | RMSE | Fixed-edge SCS |
|---|---:|---:|---:|---:|
| A1 | 73.8506 +/- 0.4815 | 4.6795 +/- 0.0195 | 6.3025 +/- 0.0098 | 0.0171 +/- 0.0045 |
| A2 | 73.8421 +/- 0.3119 | 4.6828 +/- 0.0376 | 6.3048 +/- 0.0302 | 0.0178 +/- 0.0077 |
| A3 | 73.4907 +/- 0.2427 | 4.9606 +/- 0.0584 | 6.6275 +/- 0.0767 | 0.0088 +/- 0.0033 |
| A4 | 73.7284 +/- 0.1237 | 4.9974 +/- 0.0679 | 6.6785 +/- 0.0924 | 0.0092 +/- 0.0013 |
| A5 | 74.1138 +/- 0.4720 | 4.9004 +/- 0.0425 | 6.5065 +/- 0.0515 | 0.0042 +/- 0.0027 |

A5 represents the pre-specified final entropy-acquisition stage, `A5_entropy_885`, with 885 acquired snapshots. Final-model selection considers all eligible configurations under the same validation criterion. Raw graph-row inference times refer to the graph head; complete A4 pipeline timing is reported separately. Pipeline parameter counts appear in `benchmark/params_runtime.csv`.

## Baseline development comparison

| Model | Accuracy (%) | MAE (index) | RMSE (index) |
|---|---:|---:|---:|
| A1 | 73.8506 +/- 0.4815 | 4.6795 +/- 0.0195 | 6.3025 +/- 0.0098 |
| A2 | 73.8421 +/- 0.3119 | 4.6828 +/- 0.0376 | 6.3048 +/- 0.0302 |
| ARIMA | 54.1341 +/- 0.0000 | 10.1497 +/- 0.0000 | 12.9361 +/- 0.0000 |
| HA | 73.1070 +/- 0.0000 | 4.7484 +/- 0.0000 | 6.3925 +/- 0.0000 |
| Persistence | 52.3939 +/- 0.0000 | 11.0047 +/- 0.0000 | 15.3749 +/- 0.0000 |

HA, persistence and ARIMA produce deterministic predictions on the fixed dataset, giving zero between-seed variation. ARIMA parameters are fitted on training data; its fixed-parameter state filter updates using observed past validation values for one-step forecasting.

## Independent selected-model evaluation

Selected configuration: **A1**, selected by mean validation MAE across the five seeds from A1-A4 and all nonzero-budget entropy/random stages. The same configuration is used for every seed. The final test is opened only after this selection. Benchmark fitting occurs afterward on development data and cannot alter that selection.

| Partition | Accuracy (%) | MAE (index) | RMSE (index) |
|---|---:|---:|---:|
| val | 73.8506 +/- 0.4815 | 4.6795 +/- 0.0195 | 6.3025 +/- 0.0098 |
| test | 74.8167 +/- 0.4406 | 4.7409 +/- 0.0217 | 6.3843 +/- 0.0199 |

| Test metric | Bootstrap lower 95% | Bootstrap upper 95% |
|---|---:|---:|
| accuracy_percent | 74.1060 | 75.4886 |
| mae_index | 4.6913 | 4.7942 |
| rmse_index | 6.3144 | 6.4598 |

The 95% intervals use 2,000 circular 24-hour block resamples, retaining all 20 nodes together. They describe held-out sampling uncertainty conditional on the five fitted models. Exact per-seed metrics and prediction fingerprints appear in `final_eval/`.

## Graph and acquisition conclusions

| Graph setting | MAE (index) | RMSE (index) | SCS |
|---|---:|---:|---:|
| G0 | 4.7471 +/- 0.0180 | 6.3124 +/- 0.0125 | 0.0113 +/- 0.0118 |
| G1 | 4.9974 +/- 0.0679 | 6.6785 +/- 0.0924 | 0.0092 +/- 0.0013 |
| G2_mean | 5.0292 +/- 0.0836 | 6.7294 +/- 0.1165 | 0.0058 +/- 0.0023 |

G2 is averaged over ten rewires within each seed before calculating the displayed mean/SD across five seeds. It shares the same AE-LSTM backbone as G0/G1.

The original graph and degree-preserving controls produced comparable results under the corrected paired tests; superiority of the original topology was not established across MAE, RMSE and SCS. The simulated segments share a temporal driver, and the GCN represents undirected neighborhood adjustment.

Entropy and random sampling showed comparable lower-budget accuracy under the corrected tests. Additional acquisition and refinement reduced the full pipeline's final-stage errors relative to its initial state.

| Selector | Snapshot budget | Scalar-label budget | Accuracy (%) | MAE (index) | RMSE (index) |
|---|---:|---:|---:|---:|---:|
| entropy | 0 | 0 | 73.7284 +/- 0.1237 | 4.9974 +/- 0.0679 | 6.6785 +/- 0.0924 |
| entropy | 295 | 5900 | 73.4109 +/- 0.5068 | 4.9446 +/- 0.0556 | 6.5780 +/- 0.0489 |
| entropy | 590 | 11800 | 73.4873 +/- 0.4445 | 4.9474 +/- 0.0465 | 6.5457 +/- 0.0448 |
| entropy | 885 | 17700 | 74.1138 +/- 0.4720 | 4.9004 +/- 0.0425 | 6.5065 +/- 0.0515 |
| random | 0 | 0 | 73.7284 +/- 0.1237 | 4.9974 +/- 0.0679 | 6.6785 +/- 0.0924 |
| random | 295 | 5900 | 73.6010 +/- 0.4678 | 4.9332 +/- 0.0346 | 6.5693 +/- 0.0374 |
| random | 590 | 11800 | 73.3939 +/- 0.3050 | 5.0061 +/- 0.0863 | 6.6251 +/- 0.1308 |
| random | 885 | 17700 | 74.0509 +/- 0.1866 | 4.9030 +/- 0.0393 | 6.4966 +/- 0.0329 |

Both selectors share the initial model, acquisition pool, oracle, validation set, ten-epoch LSTM and GCN refinement schedule, and stage-wise optimizer resets. At the final budget, both have acquired the entire pool; differences then reflect the training path. Because validation precedes acquisition chronologically, it serves as a fixed retrospective selection set after refinement. The later final test provides temporal generalization assessment.

Ordinal distance-to-class-center entropy ranks samples by class-boundary ambiguity. Acquisition cost is expressed as queried network snapshots and the corresponding simulated node-label count.

The simulated oracle supplies congestion targets calculated from generated volume and study capacity. Labels are withheld according to the acquisition protocol, providing a controlled setting for comparing sample-selection strategies. The resulting budget measures simulated label access rather than human annotation effort.

## Anomaly detection and architectural comparison

| AE metric | Mean +/- SD |
|---|---:|
| mse | 0.078319 +/- 0.033754 |
| precision | 0.000495 +/- 0.000128 |
| recall | 1.000000 +/- 0.000000 |
| f1 | 0.000990 +/- 0.000256 |

AE metrics quantify agreement with the hourly proxy rule. Validation contains **4 proxy positives among 11,780 node-hours**. Recall was high, while low precision and F1 show that the fixed 0.015 threshold generated many additional alerts. These findings identify threshold calibration and richer anomaly scenarios as priorities for further development.

STGCN-lite uses one temporal-gated / first-order graph / temporal-gated block with 32 channels. Two width-three temporal kernels and latest-position readout yield a five-hour effective receptive field. It is a reduced architectural comparator evaluated on validation data; broader benchmarking would require full-scale STGCN, DCRNN and Graph WaveNet implementations.

| Model | Validation accuracy (%) | MAE (index) | RMSE (index) |
|---|---:|---:|---:|
| A4 | 73.7284 +/- 0.1237 | 4.9974 +/- 0.0679 | 6.6785 +/- 0.0924 |
| A5 | 74.1138 +/- 0.4720 | 4.9004 +/- 0.0425 | 6.5065 +/- 0.0515 |
| STGCN_lite | 72.9219 +/- 0.6798 | 4.8923 +/- 0.0489 | 6.5421 +/- 0.0559 |

STGCN-lite achieved mean validation MAE 4.8923, compared with 4.9004 for A5. Parameter counts were 10,049 and 211,960, respectively, illustrating the accuracy and model-size trade-offs in this comparison.

The accompanying CPU measurements compare A4 and STGCN-lite under the same input workload.

## Inference performance

CPU inference profiling gave an A4 mean batch-one p95 of **10.489 ms** per 20-segment network update. Seed-wise mean, standard deviation and percentile timings for A4 and STGCN-lite appear in `latency/`. Batch20 represents 20 networks, or 400 segment sequences. The measured path includes engineered-window JSON decoding, float conversion, train-fitted scaling and model inference. Sensor transport, rolling-feature maintenance and dashboard rendering are outside this measurement.

The measured CPU inference path was compatible with the five-minute update cycle under the tested engineered-window workload.

## Statistical analysis

Paired two-sided t tests compare the five model seeds, with Holm correction within each comparison family and paired Cohen dz effect sizes. The ten rewires per seed are averaged before testing. Interpretation combines effect sizes, observed variability and corrected significance; five seeds provide a limited estimate of between-run variability. For deterministic classical baselines, paired variation comes from neural-model fitting.

Differences and Cohen dz are first model minus second model: negative favors the first model for MAE/RMSE, while positive favors it for accuracy/SCS. Accuracy and continuous errors aggregate over node-hours, not exact matches of entire 20-node class vectors. An additional independent full-size seed-42 A1 training repeat is used only for the reproducibility gate and is excluded from performance means and paired tests.

Detailed comparisons appear in `results/claim_support_matrix.csv`, `ablation/paired_tests.csv`, `graph_validity/significance_tests.csv`, `active_learning/paired_tests_entropy_vs_random.csv` and `benchmark/significance_vs_proposed.csv`. The tables in this report form a consistent results set for the simulation methods described here.

## Reproducibility artifacts

The release includes the fixed protocol and SHA256 hashes, model-ready dataset, trained checkpoints, seed-specific predictions, epoch logs, evaluation tables and figures. Validation predictions cover all comparisons; test predictions cover the selected configuration. These materials are packaged locally in `Reconstructed_Study_Results_and_Reproducibility.zip`. Public deposition and a permanent identifier remain future release steps.

The present evaluation covers forecasting, anomaly scoring, graph adjustment, acquisition and model-inference timing. Dashboard injection tests and routing-event benchmarks require a separate application-level evaluation using the corresponding source code and event logs.

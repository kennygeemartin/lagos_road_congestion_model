# Draft response to Reviewer 1: spatiotemporal benchmarks

**Comment:** Include comparisons with STGCN, DCRNN and Graph WaveNet.

**Response:** We have added all three requested architecture families to the experimental evaluation. The comparison includes a two-block Chebyshev STGCN, a two-layer diffusion-GRU encoder-decoder DCRNN, and Graph WaveNet with dilated temporal convolutions and adaptive adjacency. Each uses the same 14 predictors, 12-hour input window, one-hour target, initial-training partition and fixed validation set, with five predefined model seeds. Architectural details, implementation adaptations, source versions and training settings are documented in the benchmark report.

Mean validation MAE was 4.6555 +/- 0.0259 for STGCN, 4.6942 +/- 0.0253 for DCRNN and 4.7072 +/- 0.0393 for Graph WaveNet, compared with 4.6795 +/- 0.0195 for the plain LSTM, 4.9974 +/- 0.0679 for the initial AE-LSTM-GCN pipeline and 4.9004 +/- 0.0425 after entropy-guided refinement. Errors are points on the 0-100 reporting index. Paired tests with Holm correction found no significant MAE difference between the plain LSTM and each benchmark. The integrated pipeline did not establish a general error advantage over the benchmark architectures.

The expanded evaluation includes parameter counts, matched CPU model-only timings, per-seed predictions and seven architecture/artifact checks. A5's additional acquisition-label budget is stated explicitly. The fixed-validation comparison is primary; evaluation on the previously examined temporal test is identified as retrospective supplementary evidence. The original selected-model evaluation remains unchanged.

**Supporting files:** `REVIEWER_BENCHMARK_RESULTS.md`, `BENCHMARK_PROTOCOL.md`, and `reviewer_benchmarks_v2/results/`.

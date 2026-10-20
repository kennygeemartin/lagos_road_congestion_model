# Reviewer-requested architecture benchmarks

The extension evaluates STGCN, DCRNN and Graph WaveNet on the completed study's fixed data and split. Its machine-readable protocol is `reviewer_benchmarks_v2/protocol.json`, with a pre-training SHA256 lock in `protocol_sha256.json`.

## Comparison design

All models use the same 14 initial-training-scaled predictors, 12 observed hourly inputs, 20 nodes and one-hour congestion target. They fit on 4,420 initial-training network snapshots and select the best epoch by MSE on the fixed 589-snapshot validation set. Model seeds are 42, 123, 456, 789 and 1024. Training uses Adam, learning rate 0.0005, MSE on the 0-10 score, batch size 32 networks, at most 80 epochs, patience 10 and gradient clipping at norm 5. There is no hyperparameter search.

The primary comparison is on validation, against A1 (LSTM), A4 (AE-LSTM-GCN) and A5 (A4 after three entropy-acquisition stages). A1/A4 and the benchmarks share the initial 4,420 supervised training snapshots. A5 uses 885 additional acquisition snapshots. Report its larger label budget alongside its results.

All 15 benchmark fits finish before the supplementary temporal-test comparison. The existing temporal test was previously examined, so this extension labels that comparison retrospective. The original A1 selection and original test predictions remain fixed. A4/A5 retain the original protocol's development-only evaluation.

## Full architectures

- STGCN: two T-G-T blocks, channel paths 14-32-64 and 64-32-128; temporal kernel three; three Chebyshev polynomial terms; residual channel alignment; normalization; dropout 0.3; four-step temporal output collapse. All 12 real input hours influence the prediction.
- DCRNN: two-layer encoder and two-layer decoder with 64 units; directed forward and reverse random walks; diffusion depth two; zero GO decoder input. Each direction has an independently reset polynomial basis. A one-step output requires no subsequent teacher-forced decoder steps.
- Graph WaveNet: four blocks with two dilated layers each; residual/dilation channels 32, skip channels 256, end-hidden channels 512; fixed forward/reverse supports and learned adaptive adjacency using 10-dimensional embeddings; graph diffusion order two; dropout 0.3. Its 13-step receptive field includes one left-padding step and 12 observed hours.

These are multivariate, one-step task adaptations of the full architecture families. STGCN and DCRNN are PyTorch ports. Graph WaveNet uses the authors' implementation with three Conv1d-to-Conv2d compatibility declarations for four-dimensional tensors. Source versions, reference files and licenses are retained in `benchmark_vendor/`; no full-architecture model is substituted with STGCN-lite.

Sources: https://github.com/VeritasYin/STGCN_IJCAI-18, https://github.com/liyaguang/DCRNN, https://github.com/nnzhan/Graph-WaveNet.

## Artifacts and commands

```powershell
.venv/Scripts/python.exe run_reviewer_benchmarks.py --prepare
.venv/Scripts/python.exe -u run_reviewer_benchmarks.py
```

Preparation verifies the original study, runs four architecture checks and locks the extension before fitting. Completed checkpoints and per-epoch resume states support interruption recovery. Run a single trainer at a time. Progress appears in `reviewer_benchmarks_v2/progress.json` and `reviewer_benchmark_run.log`.

Training generates checkpoint weights, epoch histories, full-schema validation/test predictions and matched CPU model-only timing. Final compilation recomputes metrics, runs architecture and artifact checks, produces paired tests and writes `REVIEWER_BENCHMARK_RESULTS.md`. All 27 validation tests share one Holm correction family; the nine retrospective test comparisons form a separate family. Model-only timings measure full pipelines under an identical normalized input, excluding ingestion and scaling.

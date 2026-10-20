# Traffic-congestion forecasting study

Start with **REVISION_RESULTS.md** for the experimental design, five-seed findings, statistical comparisons and reproducibility details. The study evaluates an AE-LSTM-GCN forecasting framework on a fixed reconstructed traffic simulation dataset.

## Study organization

- `reconstructed_v1/protocol/`: pre-specified study definitions, split indices and integrity hashes.
- `reconstructed_v1/data/`: model-ready data and preprocessing parameters.
- `reconstructed_v1/DATA_DICTIONARY.md`: array shapes, feature units and label definitions.
- `reconstructed_v1/checkpoints/`: fitted AE, temporal and graph model weights.
- `reconstructed_v1/predictions/`: seed-specific validation predictions and selected-model test predictions.
- `reconstructed_v1/results/`: baseline, ablation, graph-control, acquisition, benchmark, timing and final-test evaluations.
- `reconstructed_v1/logs/`: epoch-level training records.
- `reconstructed_v1/RECONSTRUCTED_RESULTS.md`: the study report.
- `Reconstructed_Study_Results_and_Reproducibility.zip`: local reproducibility release.

The five model and selector seeds are 42, 123, 456, 789 and 1024. Data generation is held fixed across seeds. The dataset contains 8,420 network snapshots, each containing 20 segment targets. The selected plain LSTM achieved held-out accuracy of 74.8167 +/- 0.4406%, MAE of 4.7409 +/- 0.0217 and RMSE of 6.3843 +/- 0.0199 on the 0-100 index.

## Reproduction workflow

Use PowerShell from the project directory with the Python environment specified in `requirements.installed.txt`.

```powershell
.venv/Scripts/python.exe reconstructed_study.py verify
.venv/Scripts/python.exe -u run_reconstructed_experiments.py
.venv/Scripts/python.exe summarize_reconstructed_study.py
.venv/Scripts/python.exe run_reconstructed_checks.py
```

The distributed study is already prepared and locked. On a fresh workspace, `reconstructed_study.py prepare` and `reconstructed_study.py lock` assemble and fix the protocol before fitting. Existing fitted checkpoints support resumption. Run one training process at a time. Final-test markers preserve the evaluation boundary; an incomplete marked evaluation should be investigated before resumption.

The result compiler prepares tables, paired comparisons, figures, CPU measurements and the release archive. It performs an independent full-size seed-42 A1 training repeat before checking reproducibility. That repeat reproduced all validation predictions exactly and is excluded from the five-seed performance summary. The completed study passed all 23 implementation and scientific checks.

`polish_revision_writing.py` applies presentation edits to the report while preserving every numerical table. The result compiler uses the same presentation function.

## Prediction format

`pred_{experiment}_{seed}_{val|test}.npz` files contain non-pickle arrays. `y_true_index`, `y_pred_index`, `y_true_class` and `y_pred_class` have shape `(time,20)`; `probs` has shape `(time,20,5)`; `indices` and `timestamp` have shape `(time,)`; `segment_id` lists nodes 1 through 20. Timestamps represent ordinal reconstructed hours. Errors are reported on the 0-100 index and the equivalent 0-10 scale; class boundaries are 20, 40, 60 and 80 on the reporting index.

## Source provenance and audit

`Traffic_Congestion_Reproducibility_Package/` holds the supplied source material. `REVISION_AUDIT_INITIAL.md`, the root `protocol/` and the root `results/` preserve its initial audit. The completed study has its own fixed protocol and results in `reconstructed_v1/`. Original Metro timestamps and Nigerian route-context records were outside the available experimental inputs.

`tests/test_protocol.py` checks the initial source archive's prerequisites. Use `run_reconstructed_checks.py` for the completed study's 23-check gate; the two checks have different scopes.

The release contains CPU measurements from the actual execution environment. Public archival deposition and a permanent identifier are subsequent release steps.

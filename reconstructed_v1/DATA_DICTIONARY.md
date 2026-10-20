# Newly defined reconstructed-data study

This study uses the fixed reconstructed archive. It does not recover the original UCI Metro vector, calendar dates, Nigerian route records or original feature-generation code. Five seeds vary model fitting and selection only. The original split CSV is preserved exactly.

`data/model_ready.npz` contains non-pickle NumPy arrays:

| Array | Shape | Meaning |
|---|---|---|
| features | 8432 x 20 x 14 | Initial-train-fitted min-max features; later values may exceed [0,1] |
| raw_features | 8432 x 20 x 14 | Assembled, unscaled features after causal filling |
| y_hourly | 8432 x 20 | Congestion score on the original 0-10 scale |
| ratio | 8432 x 20 | Observed generated hourly volume/capacity |
| proxy_hourly | 8432 x 20 | New hourly proxy labels using train-only node quantiles |
| scaler_min / scaler_max / scaler_scale | 14 each | Global-per-feature fit over unique initial-training input node-hours |
| sensor_train_medians | 20 x 4 | Node-specific train medians for sensor columns before filling |
| target_time_index | 8420 | Ordinal proxy target hours, 12 through 8431 |
| initial_train | 4420 | Snapshot IDs 0 through 4419 |
| fixed_validation | 589 | Snapshot IDs 4420 through 5008 |
| acquisition_pool | 885 | Snapshot IDs 5009 through 5893 |
| final_test | 2526 | Snapshot IDs 5894 through 8419 |
| node_ids | 20 | Road-segment IDs 1 through 20 |
| train_input_stop | scalar | Exclusive end of scaler/median/quantile support: 4431 |

For snapshot ID s, target hour t=`target_time_index[s]`, inputs are hours t-12 through t-1, and the target is `y_hourly[t]`. The target's current volume/speed/delay is never inserted into that snapshot's input window. Sensor features are mean-aggregated by node/hour. Missing readings carry the last observed value forward within node; only leading gaps fall back to that node's training median. Sensor count is zero for missing hours and is never forward-filled.

The exact 14-column order and scaling rules are locked in `protocol/feature_config.json`. Ratio rolling features include the current observed input hour and earlier hours only. This is a next-hour forecasting problem, not classification of the same-hour volume used as input.

Each acquired snapshot exposes 20 scalar targets. The three rounds are 295/590/885 cumulative snapshots, equivalent to 5,900/11,800/17,700 node labels. Both units must accompany label-efficiency plots.

The AE uses initial-training input rows with ratio below 0.4. Its chronological normal-feature subset is split 80/20 internally for fitting/early stopping. All other models use the fixed validation split for early stopping. Active-learning refinement uses exactly ten LSTM and ten graph epochs per round, resets Adam at each round for both selectors, and uses the final epoch rather than selecting an intermediate epoch. The frozen scaler and AE do not receive acquisition refitting.

The validation period precedes the acquisition period, as specified by the supplied split. After acquisition, validation is a fixed retrospective selection set, while the final test is later than all training/acquisition records. Do not describe the acquired-stage validation as a strictly future, prospective forecasting holdout.

All GCN comparisons use the same fixed original undirected edge set for SCS. Ten rewires per seed preserve the binary graph's degrees and connectedness. The source generator uses a shared temporal driver with independent node disturbances; it does not simulate causal congestion spillover along the road graph.

The adapted STGCN-lite has a single temporal-gated / first-order graph / temporal-gated block, 32 channels, kernel width three in each temporal layer, and latest-position scalar readout. Its effective temporal receptive field is five hours. Disclose that adaptation rather than presenting it as a full canonical STGCN reproduction.

Original-family checkpoints are named `A1`, `A2`, `A3`, `A4`; active stages include selector and snapshot budget. AE checkpoint training-row IDs refer to the normal-feature subset, rather than snapshot IDs. Graph `G1` reuses the identical fitted A4 head. Validation prediction files exist for each reported comparison. Only the validation-selected final configuration has saved test predictions.

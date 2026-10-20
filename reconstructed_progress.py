"""Refresh the manuscript-facing status from real study artifacts, without fitting."""
import json
import shutil
from pathlib import Path
import pandas as pd
from reconstructed_study import ROOT,STUDY,verify

def refresh():
    verify(require_locked=True)
    legacy=ROOT/'REVISION_AUDIT_INITIAL.md'
    if not legacy.exists():shutil.copyfile(ROOT/'REVISION_RESULTS.md',legacy)
    if (STUDY/'results/experiment_status.csv').exists():
        return
    logs=sorted((STUDY/'logs').glob('*.csv'),key=lambda p:p.stat().st_mtime,reverse=True)
    checkpoints=sorted((STUDY/'checkpoints').glob('*.pt'))
    current='Classical baseline preparation'
    if logs:
        try:
            frame=pd.read_csv(logs[0])
            row=frame.iloc[-1]
            current=f"{logs[0].stem}: completed epoch {int(row.epoch)}, validation MSE {row.val_mse:.6f}, latest epoch {row.seconds:.2f}s"
        except (pd.errors.EmptyDataError,IndexError):current=f'{logs[0].stem}: epoch in progress'
    status=dict(study='new reconstructed-data simulation',full_manuscript_size=True,status='RUNNING',
        current=current,completed_checkpoints=[p.name for p in checkpoints],
        final_test_completed=(STUDY/'results/final_eval/completed.json').exists())
    (STUDY/'progress.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
    report=f'''# New reconstructed-data simulation: full experiments running

The user approved a new reconstructed-data study and explicitly chose the full manuscript-size five-seed experiments rather than a compact pilot. The new protocol is **locked** and its five prerequisite tests pass. These runs do not reproduce the original Metro chronology or authenticate historical manuscript numbers.

**Current stage:** {current}.

**Completed fitted checkpoints:** {len(checkpoints)}. Final test completed: {status['final_test_completed']}.

Maximum epochs and architectures are unchanged: AE 100 with training-internal early stopping; two-layer 128-unit LSTM 80 with early stopping; GCN 50 with early stopping; active-learning LSTM/GCN refinement ten fixed epochs per round. Actual CPU execution can take several hours. Fitted weights are saved when each model completes; epoch losses/timing are written throughout training.

## Resolved scientific definitions

- Fixed supplied reconstructed data; five **model/selector** seeds {{42,123,456,789,1024}}. No claim of five independent regenerated datasets.
- 8,432 ordinal hours per segment; 8,420 **network snapshots**, each with 20 segment targets: 168,400 segment targets total.
- Archived split copied unchanged: 4,420 train / 589 validation / 885 acquisition / 2,526 test snapshots.
- 14 explicit features with hourly sensor means, causal forward filling and train-node median fallback. Full definitions are in `reconstructed_v1/protocol/feature_config.json`.
- Initial training INPUT hours 0..4430 alone fit the scaler, fallback medians and proxy quantiles. AE theta=0.015 is fixed in advance.
- One 295-snapshot acquisition exposes 5,900 scalar node labels. Both units will be reported.
- Binary symmetric GCN adjacency, 50 degree-preserving rewiring controls, matched no-neighbor head, entropy/random equal-budget selectors.
- The minimum five-seed mean validation MAE chooses the final configuration before any test performance is evaluated. The benchmark is development-only after that selection.
- Actual CPU latency only; no RTX3090 result will be claimed.

## Files to inspect during execution

- `reconstructed_run.log`: current training progress.
- `reconstructed_v1/protocol/`: locked config, source hashes, split indices and actual environment.
- `reconstructed_v1/data/split_counts.csv`: snapshot/node-label counts and new hourly proxy support.
- `reconstructed_v1/logs/`: real epoch losses and timing.
- `reconstructed_v1/checkpoints/`: completed real fitted model weights.
- `reconstructed_v1/predictions/`: saved per-seed predictions as experiments complete.
- `reconstructed_v1/results/`: incremental measured metrics, followed by final tables/statistics/figures.

The original data audit remains in `REVISION_AUDIT_INITIAL.md`. Its failed checks concern the original archive's unresolved definitions; the new study has separately documented and tested definitions. Do not run the legacy finalizer to overwrite this new study status.

Do not copy provisional numbers into the manuscript before all five seeds, paired comparisons and independent final evaluation have completed. The completed report will replace this status using saved result artifacts only.
'''
    (ROOT/'REVISION_RESULTS.md').write_text(report,encoding='utf-8')
    print(current,flush=True)

if __name__=='__main__':refresh()

"""Write a candid revision evidence report and run all available checks."""
import contextlib
import hashlib
import importlib.metadata
import io
import json
import platform
import sys
import unittest
import zipfile
from pathlib import Path
from make_data_audit import ROOT,read,write_csv,write_json

def main():
    if (ROOT/'reconstructed_v1/protocol/sha256.json').exists():
        raise SystemExit('New reconstructed study exists. Use reconstructed_progress.py while running, then summarize_reconstructed_study.py; legacy finalizer will not overwrite its report.')
    output=ROOT/'results'
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    log=io.StringIO()
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (output/'test_run.log').write_text(log.getvalue(),encoding='utf-8')
    checks=dict(total=result.testsRun,passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped))
    write_json(output/'test_summary.json',checks)
    versions={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    write_json(output/'installed_environment.json',dict(python=sys.version,platform=platform.platform(),packages=versions))
    import torch
    write_json(output/'hardware.json',dict(cuda_available=torch.cuda.is_available(),torch=torch.__version__,
               cpu=platform.processor(),cuda_devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
               rtx3090_benchmark_performed=False))
    (ROOT/'requirements.installed.txt').write_text('\n'.join(f'{name}=={version}' for name,version in sorted(versions.items(),key=lambda x:x[0].lower()))+'\n',encoding='utf-8')
    families=[
      ('data_audit','IMPLEMENTED AND RUN','CSV tables, target verification, threshold provenance, split audit, flowchart'),
      ('protocol','IMPLEMENTED EVIDENCE SNAPSHOT; NOT LOCKED','Source and protocol SHA256 manifests, split NPZ, binary normalized graph, failing prerequisites'),
      ('ae','MODEL COMPONENT IMPLEMENTED; EXPERIMENT NOT RUN','14-column assembly and aligned sensor/time mapping absent; original thresholds use all data'),
      ('baselines','NOT IMPLEMENTED OR RUN','Depends on resolved sample unit, model-ready features and locked development protocol'),
      ('ablation','MODEL COMPONENTS IMPLEMENTED; EXPERIMENT NOT RUN','AE/LSTM/GCN modules available; no A1-A5 experiment orchestration or model-ready data'),
      ('graph_validity','CONTROLS IMPLEMENTED AND GENERATED; MODELS NOT RUN','50 degree-preserving binary graph controls; no trained G0/G1/G2 performance comparisons'),
      ('active_learning','SELECTORS IMPLEMENTED; EXPERIMENT NOT RUN','Five 3x295 random schedules and entropy selector; label unit and retraining orchestration unresolved'),
      ('final_eval','EVALUATOR IMPLEMENTED; NO STUDY PREDICTIONS','Per-seed val/test metrics, clustered bootstrap and plots available only after genuine fitted predictions'),
      ('benchmark','STGCN-LITE COMPONENT IMPLEMENTED; EXPERIMENT NOT RUN','Disclosed one-block first-order approximation; no study benchmark performance'),
      ('latency','MEASUREMENT FUNCTION IMPLEMENTED; STUDY BENCHMARK NOT RUN','Actual ingestion/feature assembly absent; no CUDA available in installed runtime'),
      ('public_release','LOCAL REVIEW BUNDLE ONLY; NOT PUBLISHED','Protocol gate fails; no genuine checkpoints/predictions or archival destination')]
    write_csv(output/'experiment_status.csv',[dict(family=f,status=s,detail=d) for f,s,d in families])
    claims=[
      ('AE anomaly feature helps','A2 vs A1 paired-seed comparison','UNSUPPORTED: no independent predictions','Describe AE purpose; remove numerical contribution claim until rerun'),
      ('LSTM beats classical baselines','HA/PF/ARIMA/LSTM development table','UNSUPPORTED: metadata only','Do not reuse historical metrics as new validation'),
      ('Spatial adjustment helps','A4 vs A2, A3 vs A1, G1 vs G0/G2; fixed-edge SCS','UNSUPPORTED: controls exist; models not trained','Binary adjacency implementation now available; outcome unknown'),
      ('Active learning improves label efficiency','Entropy vs random at matched budget','UNSUPPORTED: selectors exist; no fitted learning curves','Retain explicit no-random-control limitation until genuine rerun'),
      ('Useful lightweight data-scarce model','STGCN benchmark, parameters and latency','UNSUPPORTED: model components only','Frame as proposed simulation pipeline pending comparisons'),
      ('Held-out performance is credible','Independent per-seed predictions and temporal bootstrap','UNSUPPORTED: reported_results.json is not prediction evidence','Do not repeat 94.5/2.30/3.12 as verified results'),
      ('Five-minute update feasible','100 real pipeline runs after 20 warmups on declared hardware','UNSUPPORTED: no study latency benchmark','Remove universal sub-100ms requirement; report only measured actual hardware'),
      ('Archive has stated split sizes and topology','Observed split/graph files','SUPPORTED','4,420/589/885/2,526 disjoint ordinal split IDs; 20 nodes, 34 directed edges'),
      ('Generated target matches stated piecewise rule','All 168,640 generated rows','SUPPORTED','Zero target/class formula mismatches detected')]
    write_csv(output/'claim_support_matrix.csv',[dict(claim=c,required_evidence=e,status=s,revision_action=a) for c,e,s,a in claims])
    report=f'''# Manuscript revision: evidence audit and implementation status

The supplied archive does not currently support an end-to-end reproducible rerun of the manuscript. This report contains measured audit findings and tested software components. **No AE/LSTM/GCN study metrics, active-learning gains, benchmark scores, p-values, test confidence intervals or latency numbers were generated.** Historical results in `metadata/reported_results.json` were not used as newly measured results.

## Measured data accounting

| Quantity | Observed or computed result |
|---|---:|
| Original Metro rows | 48,204 claimed; original file/vector absent |
| Supplied reconstructed temporal proxy | 8,432 ordinal hours |
| Generated rows per segment | 8,432, for every one of 20 segments |
| Generated segment rows in total | 168,640 |
| L=12 next-step windows per segment | 8,420 |
| Pooled segment sequences | **168,400**, not 8,420 |
| Whole-network snapshots, if that unit is adopted | **8,420**, each containing 20 node targets |
| Split rows | 4,420 train; 589 validation; 885 acquisition; 2,526 test |
| Split uniqueness/order | Passed for ordinal indices; no calendar timestamps available |
| Nodes / directed transitions | 20 / 34 |
| Binary undirected support edges | 19 |
| Generated target/class formula mismatches | 0 of 168,640 rows |
| Synthetic sensor records | 50,000 records; no per-minute timestamp or trajectory identifier supplied |
| Archived positive proxy labels | 20 / 50,000 (0.04%) |

The exact reconciliation is `20 * (8432 - 12) = 168400` segment sequences. Alternatively `8432 - 12 = 8420` network snapshots. The archived split CSV has no segment ID and one row per target time. This is compatible with the network-snapshot interpretation, but it does not establish how the original models were assembled or trained. The original source split was preserved byte-for-byte.

If network snapshots are the intended unit, revise the manuscript to state **8,420 network sequences with 20 segment targets each**. The train/val/acquisition/test node-target counts would then be 88,400 / 11,780 / 17,700 / 50,520. The held-out bootstrap should sample time blocks of snapshots and retain all nodes together; it should not treat 50,520 correlated segment values as independent observations. At each acquisition round, 295 snapshots would expose **5,900 scalar node labels**, unless the oracle queries a separately specified subset of nodes. All three rounds expose 17,700 node labels. Report both units.

## Leakage and AE evidence

For **all 20 nodes**, archived `speed_p10` and `intensity_p90` match quantiles calculated from **all 50,000 sensor records**, including later partitions. They differ from quantiles calculated from initial-training temporal support. This violates the requested train-only fitting policy. The audit's diagnostic training support includes ordinal hours 0..4431: the union of initial-training histories and target hours. Exact fitting support must be finalized with the feature/label mapping; it must never include validation/acquisition/test records.

The archived labels have 9 initial-training positives, 2 validation positives, 2 acquisition positives and 7 final-test positives when sensor rows are assigned by target-time intervals. Another 68 sensor records fall in the initial 12 historical hours and are outside those target intervals. The train-only and full-data diagnostic rules each produce 20 positives overall; equal counts do not validate full-data threshold fitting or prove that individual labels are unchanged.

With only two archived validation positives, a claimed five-seed AE recall near 0.88 requires the original evaluation-unit definition and per-seed labels/predictions. For a fixed two-positive validation set, a deterministic recall in one run can only be 0, 0.5 or 1. Original run-specific regeneration or a different evaluation unit could change that denominator; neither is established by this archive. This finding does not prove the historical result is false, but prevents verification.

The requested locked protocol fits thresholds on initial training only, whereas the methods text says theta was selected by validation Youden J. These are different policies. For a new locked rerun, retain theta=0.015 as pre-specified, or select a threshold using a training-internal holdout and freeze it. Do not claim that a fresh percentile sweep independently reproduces theta=0.015 or J=0.79. The code does not invent those results.

## Provenance and feature gaps

The source README and generation configuration explicitly call the supplied temporal vector a **deterministic reconstructed proxy**, not the original UCI observations. Its `time_index` has no Metro `date_time` mapping. The original Metro-to-cleaned-hours-to-selected-8432-hours chain cannot be established. The supplied Nigerian route source, route-matching code and the original feature assembly are also absent.

The preprocessing config lists **eight** engineered feature names, while the AE architecture requires **14** columns. No exact 14-column ordering, hourly sensor aggregation, interpolation algorithm or feature-generation implementation is supplied. The 50,000 sensor rows do not establish the manuscript's claimed one-minute trajectory chronology. Creating these mappings would define a new simulation implementation, rather than recover the original experiment.

The archive contains one reconstructed dataset (seed 42), while the manuscript says synthetic generation was repeated under each run seed. A new rerun must state whether data are fixed and only model seeds vary, or regenerate data per seed from a fully specified generator. Neither the original five model checkpoints nor per-seed predictions are included.

## Implemented and tested components

1. `make_data_audit.py`: repeatable standard-library audit, all requested CSV audit tables, additional proxy-threshold and class checks, SVG flowchart, source hashes, immutable-by-verification protocol evidence snapshot and portable split NPZ. `audit_flowchart.pdf` and PNG are supplied alongside the SVG.
2. `tests/test_protocol.py`: split completeness/order, source/protocol integrity, binary graph normalization, fixed threshold and explicit prerequisite gates. Training-dependent tests are visibly skipped, not reported as passed.
3. `experiment_controls.py`: all 50 degree-preserving random rewires (10 per seed), connectedness preservation, five no-replacement 3x295 random schedules and probability-only entropy selection. Degrees are preserved on the binary undirected GCN graph; all future SCS comparisons must use the same original graph edge set.
4. `evaluate_predictions.py`: strict genuine-prediction validation, 0-100 and 0-10 errors, ordinal classes/probabilities, fixed-edge SCS, paired t comparisons with Holm correction, paired standardized effect sizes, per-seed val/test summary, temporal-block bootstrap (default 24 hours) and final confusion/residual plots. Final evaluation refuses an unlocked protocol and existing final output. Tests use artificial fixtures only, not manuscript data.
5. `revision_models.py`: newly implemented AE, shared LSTM backbone, matched GCN/no-neighbor head, disclosed STGCN-lite approximation, early-stopped fitting/checkpoint helper, deterministic seed initialization and synchronized caller-supplied end-to-end latency function. These are reusable components, **not a completed A1-A5 experiment runner**. Baseline/ablation/active-learning orchestration and model-ready feature construction remain outstanding.

The GCN uses binary symmetric support with self-loops:

`B = ((A + A.T) > 0); A_hat = B + I; d_i = sum_j A_hat[i,j]; A_norm[i,j] = A_hat[i,j] / sqrt(d_i*d_j)`.

For Equation 12, define the neighborhood as nodes with B[i,j]=1, add the node itself once, and define degrees after self-loop addition. The original directed adjacency remains separate. Existing reported GCN metrics cannot be relabelled as binary-adjacency rerun results.

The STGCN-lite component is a temporal-gated / first-order graph / temporal-gated block with 32 channels. It omits higher-order Chebyshev filters and the second block. Disclose this as an adapted benchmark if it is used; do not call it a canonical STGCN reproduction.

## Verification status

Latest full test run: **{checks['passed']} passed, {checks['failed']} failed, {checks['skipped']} skipped, {checks['errors']} errors** ({checks['total']} total). See `results/test_run.log` for exact evidence. The failures flag the unresolved data/protocol conditions. Passing synthetic model tests do not establish same-seed reproducibility after full training.

`protocol/STATUS.json` explicitly declares `locked=false`. The files are an evidence snapshot, not an approved experiment lock. A SHA256 manifest detects subsequent edits. This is integrity checking, not OS-level write protection. Existing protocol files and the original split are not overwritten by the audit. Once the scientific definitions are resolved, create a separately versioned protocol and lock it before fitting.

Installed hardware/runtime: PyTorch {torch.__version__}; CUDA available: {torch.cuda.is_available()}. This is not the manuscript's Python 3.9 / PyTorch 2.0 / RTX 3090 environment. Actual installed versions are recorded in `requirements.installed.txt`, `results/installed_environment.json` and `results/hardware.json`. No RTX 3090 latency claim can be verified here.

## Manuscript-safe text supported by this audit

> Inspection of the currently supplied reconstructed package identified 8,432 hourly indices per segment. A 12-hour input window and next-step target produce 8,420 network snapshots, equivalent to 168,400 segment-level targets across 20 nodes. The archived split contains 4,420 initial-training, 589 validation, 885 acquisition and 2,526 final-test snapshot indices. These indices are disjoint and ordered by the proxy temporal index. The current package does not contain the original Metro timestamp mapping or per-seed fitted predictions, so it does not independently verify the previously reported performance values.

Use that paragraph only for describing this package audit. A revised Results section still requires genuine rerun outputs. Do not infer AE or graph benefits from architecture descriptions. Do not infer label efficiency from the final full-pool acquisition stage: both entropy and random selection have then seen all 885 pool items; differences can reflect optimization order and path dependence. Lower-budget comparisons are essential.

With only five seeds, an exact two-sided Wilcoxon signed-rank test has minimum attainable p=0.0625 when all five differences have the same sign and there are no zeros. It cannot establish p<0.05 in that setting. The evaluator pre-specifies paired t tests instead; its assumptions and low power must be stated, and tests must not be chosen after observing their p-values. Apply Holm correction within pre-specified comparison families. Do not treat the 50 rewiring runs as 50 independent model seeds: average rewires within seed before paired comparisons to G1.

## Required to complete the study runs

Choose a documented study version: (a) reproduce the original study using its original Metro temporal vector/timestamp mapping, exact 14-feature assembly and generation/aggregation code; or (b) conduct a **new, explicitly reconstructed-data simulation** with a newly documented feature generator, network-snapshot sample unit, oracle cost unit and fixed-data versus regenerated-data seed policy. The pasted methods text supplies scientific intent but does not resolve those executable definitions.

After that choice: finalize feature/label construction, regenerate train-only proxy labels/scalers, rerun audit gates, lock a new protocol, implement the experiment orchestration, then run the requested families in order. Keep the final test sealed until model selection. Use `results/experiment_status.csv` and `results/claim_support_matrix.csv` to track which reviewer claims remain unsupported.

The local ZIP is a review bundle, not a public reproducibility release. Publication was not attempted because the prerequisite tests fail and genuine experiment checkpoints/predictions are absent. The original archive and manuscript were left unchanged.
'''
    (ROOT/'REVISION_RESULTS.md').write_text(report,encoding='utf-8')
    files=[ROOT/x for x in ['make_data_audit.py','experiment_controls.py','evaluate_predictions.py','revision_models.py',
            'finalize_revision_package.py','REVISION_RESULTS.md','README_REVISION.md','requirements.installed.txt']]
    files.extend(p for folder in ['Traffic_Congestion_Reproducibility_Package','protocol','results','tests']
                 for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    bundle=ROOT/'Revision_Evidence_and_Code.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:
            if p.exists():z.write(p,p.relative_to(ROOT))
    print(json.dumps(checks,indent=2))
    print(f'Wrote {bundle.name} and REVISION_RESULTS.md. Protocol remains unlocked.')

if __name__=='__main__':main()

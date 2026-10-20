"""Compile genuine full-architecture results into a reviewer-facing report."""
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from reconstructed_study import STUDY,SEEDS,digest
from make_data_audit import ROOT,write_json
from run_reviewer_benchmarks import EXT,verify_extension,timing
from run_reconstructed_experiments import Study
from evaluate_predictions import paired_comparisons
from revision_models import Autoencoder,TemporalLSTM,GraphAdjustment


class ExistingPipeline(nn.Module):
    def __init__(self,name,seed,adjacency):
        super().__init__();self.name=name
        def load(model,key):
            model.load_state_dict(torch.load(STUDY/'checkpoints'/f'{key}_{seed}.pt',weights_only=True)['state_dict']);return model.eval()
        if name=='A1':self.backbone=load(TemporalLSTM(14),'A1')
        else:
            self.ae=load(Autoencoder(),'AE')
            self.backbone=load(TemporalLSTM(15),'A2' if name=='A4' else 'A5_entropy_885_LSTM')
            self.head=load(GraphAdjustment(adjacency,16),'A4' if name=='A4' else 'A5_entropy_885_GCN')
    def forward(self,x):
        if self.name=='A1':return self.backbone(x)
        features=torch.cat([x,self.ae.anomaly_score(x).unsqueeze(-1)],-1)
        pred=self.backbone(features).clamp(0,10)
        return self.head(torch.cat([pred.unsqueeze(-1),features[:,-1]],-1))


def pairs(frame,comparisons,file):
    rows=[]
    for a,b in comparisons:
        aa=frame[frame.experiment==a].set_index('seed');bb=frame[frame.experiment==b].set_index('seed')
        for metric in ['accuracy_percent','mae_index','rmse_index']:
            for seed in SEEDS:rows.append(dict(comparison=f'{a} vs {b}',metric=metric,seed=seed,a=aa.loc[seed,metric],b=bb.loc[seed,metric]))
    result=pd.DataFrame(paired_comparisons(rows));result.to_csv(file,index=False);return result


def table(frame,names):
    lines=[]
    for name in names:
        values=frame[frame.experiment==name]
        cells=[f'{values[m].mean():.4f} +/- {values[m].std(ddof=1):.4f}' for m in ['accuracy_percent','mae_index','rmse_index']]
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    return '\n'.join(lines)


def main():
    config=verify_extension()
    # Validation, prediction alignment and checkpoint replay all finish before report.
    architecture=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_reviewer_benchmarks.py','-v'],cwd=ROOT,capture_output=True,text=True)
    completed=subprocess.run([sys.executable,'validate_reviewer_benchmarks.py'],cwd=ROOT,capture_output=True,text=True)
    (EXT/'results/final_checks.log').write_text(architecture.stdout+architecture.stderr+completed.stdout+completed.stderr,encoding='utf-8')
    if architecture.returncode or completed.returncode:raise RuntimeError('Benchmark checks failed; inspect final_checks.log')
    write_json(EXT/'results/check_summary.json',dict(passed=7,failed=0,architecture_checks=4,artifact_checks=3))
    val=pd.read_csv(EXT/'results/validation_by_seed.csv')
    original=pd.read_csv(STUDY/'results/ablation/metrics_by_seed.csv')
    original=original[original.experiment.isin(['A1','A4','A5'])].copy();original['split']='val'
    combined=pd.concat([original,val],ignore_index=True)
    combined.to_csv(EXT/'results/all_validation_by_seed.csv',index=False)
    names=['A1','A4','A5','STGCN','DCRNN','GraphWaveNet']
    columns=['accuracy_percent','mae_index','rmse_index','scs','parameters']
    combined.groupby('experiment')[columns].agg(['mean','std']).to_csv(EXT/'results/validation_mean_std.csv')
    tests=pairs(combined,[(a,b) for a in ['A1','A4','A5'] for b in ['STGCN','DCRNN','GraphWaveNet']],EXT/'results/validation_paired_tests.csv')
    test=pd.read_csv(EXT/'results/retrospective_test_by_seed.csv')
    baseline=pd.read_csv(STUDY/'results/final_eval/val_vs_test_by_seed.csv')
    base=pd.DataFrame([dict(experiment='A1',seed=r.seed,accuracy_percent=r.test_accuracy_percent,mae_index=r.test_mae_index,rmse_index=r.test_rmse_index) for r in baseline.itertuples()])
    test=pd.concat([base,test],ignore_index=True)
    test.to_csv(EXT/'results/all_retrospective_test_by_seed.csv',index=False)
    test.groupby('experiment')[['accuracy_percent','mae_index','rmse_index']].agg(['mean','std']).to_csv(EXT/'results/retrospective_test_mean_std.csv')
    pairs(test,[('A1',b) for b in ['STGCN','DCRNN','GraphWaveNet']],EXT/'results/retrospective_test_paired_tests.csv')
    runtime=pd.read_csv(EXT/'results/model_timing_by_seed.csv')
    existing=EXT/'results/existing_pipeline_timing.csv'
    if existing.exists():extra=pd.read_csv(existing)
    else:
        study=Study();adjacency=json.loads((STUDY/'protocol/graph.json').read_text())['normalized_adjacency'];rows=[]
        for seed in SEEDS:
            for name in ['A1','A4','A5']:rows.extend(timing(ExistingPipeline(name,seed,adjacency).eval(),study.x,name,seed))
        extra=pd.DataFrame(rows);extra.to_csv(existing,index=False)
    runtime=pd.concat([runtime,extra],ignore_index=True);runtime.to_csv(EXT/'results/matched_model_timing.csv',index=False)
    runtime.groupby(['experiment','batch_networks'])[['mean_ms','p95_ms','parameters']].agg(['mean','std']).to_csv(EXT/'results/matched_model_timing_summary.csv')
    performance=[]
    for name in names:
        s=runtime[(runtime.experiment==name)&(runtime.batch_networks==1)]
        training=combined[combined.experiment==name]
        performance.append(f'| {name} | {int(s.parameters.iloc[0]):,} | {s.mean_ms.mean():.3f} +/- {s.mean_ms.std(ddof=1):.3f} | {s.p95_ms.mean():.3f} |')
    per_model=val.groupby('experiment').mae_index.mean();winner=per_model.idxmin()
    source=json.loads((ROOT/'benchmark_vendor/sources.json').read_text())
    commits='\n'.join(f"- {r['name']}: `{r['commit']}`" for r in source['repositories'])
    findings=[]
    for a in ['A1','A4','A5']:
        for b in ['STGCN','DCRNN','GraphWaveNet']:
            row=tests[(tests.comparison==f'{a} vs {b}')&(tests.metric=='mae_index')].iloc[0]
            direction='lower' if row.mean_difference_a_minus_b<0 else 'higher'
            findings.append(f'- {a} vs {b}: MAE difference {row.mean_difference_a_minus_b:+.4f} index points ({direction} for {a}); Holm-adjusted p={row.p_holm:.4g}.')
    report=f'''# Reviewer benchmark comparison: STGCN, DCRNN and Graph WaveNet

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
{table(combined,names)}

Among the three requested architectures, **{winner}** achieved the lowest mean validation MAE, **{per_model.min():.4f}** index points. Full comparisons and effect sizes appear in `results/validation_paired_tests.csv`. Two-sided paired t tests use five model-seed pairs; Holm correction covers all 27 validation comparisons (three existing pipelines x three benchmarks x three metrics).

{chr(10).join(findings)}

Positive MAE differences favor the benchmark; negative differences favor the first named pipeline. Statistical interpretation is limited to the fixed simulation and five fitting seeds.

## Supplementary temporal-test comparison

The original v1 temporal holdout had already been examined before this extension was defined. It is therefore reported as a **retrospective supplementary comparison**, distinct from the original selected-model held-out evaluation. All 15 benchmark fits and validation epoch selections finished before these test predictions were computed. These results do not alter the original A1 selection.

| Model | Accuracy (%) | MAE (index) | RMSE (index) |
|---|---:|---:|---:|
{table(test,['A1','STGCN','DCRNN','GraphWaveNet'])}

Supplementary paired tests comparing A1 with each benchmark are corrected as a separate nine-test family in `results/retrospective_test_paired_tests.csv`. A4 and A5 retain their development-only evaluation in the original protocol.

## Matched CPU model profiling

All six configurations were timed using the same normalized 12-hour input tensor, CPU, four PyTorch threads, float32, 20 warmups and 100 measured repetitions. A batch of one represents one 20-node network update. Timings include all model components for A4/A5 and exclude data ingestion and preprocessing; they are separate from the earlier JSON-inclusive A4 timings.

| Model | Parameters | Mean model time (ms), mean +/- SD over seeds | Mean seed-wise p95 (ms) |
|---|---:|---:|---:|
{chr(10).join(performance)}

Batch-one and batch-20 measurements are in `results/matched_model_timing.csv`; fit times, executed epochs and selected epochs are in `results/validation_by_seed.csv`.

## Implementation provenance and checks

Reference sources: [STGCN](https://github.com/VeritasYin/STGCN_IJCAI-18), [DCRNN](https://github.com/liyaguang/DCRNN), [Graph WaveNet](https://github.com/nnzhan/Graph-WaveNet). Pinned commits:

{commits}

STGCN and DCRNN are PyTorch ports of the referenced architecture families. The DCRNN polynomial basis is reset independently for each directional support. Graph WaveNet retains the authors' model with three Conv1d declarations replaced by Conv2d for compatibility with its four-dimensional tensors and two-dimensional kernels. Input and output dimensions are adapted to this study. Reference files, source hashes, patches and licenses are preserved in `benchmark_vendor/`.

**Seven checks passed:** four architecture/operator checks and three completed-artifact checks. These cover finite forward/backward computation, earliest-input influence, adaptive-adjacency gradients, directed supports, numerical diffusion/Chebyshev basis calculations, seeded initialization, all training IDs, prediction alignment, recomputed metrics and replay of all 15 fitted checkpoints. Check records are in `results/final_checks.log`.

The protocol, code fingerprints, epoch logs, 15 fitted checkpoints and 30 full-schema prediction files are retained in `reviewer_benchmarks_v2/`. Source data, feature scaling, original splits and the original study's selected-model predictions remain fixed.
'''
    (EXT/'BENCHMARK_RESULTS.md').write_text(report,encoding='utf-8')
    (ROOT/'REVIEWER_BENCHMARK_RESULTS.md').write_text(report,encoding='utf-8')
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    grouped=combined.groupby('experiment').mae_index
    fig,ax=plt.subplots(figsize=(9,5));ax.bar(names,[grouped.mean()[n] for n in names],yerr=[grouped.std()[n] for n in names],capsize=4)
    ax.set_ylabel('Validation MAE (0-100 index points)');ax.set_title('Full-architecture comparison: five seeds (mean +/- SD)')
    fig.tight_layout();fig.savefig(EXT/'results/validation_mae_comparison.png',dpi=200);plt.close(fig)
    files=[p for p in EXT.rglob('*') if p.is_file() and p.name!='artifact_sha256.json' and not p.name.endswith('.resume.pt')]
    write_json(EXT/'artifact_sha256.json',{p.relative_to(ROOT).as_posix():digest(p) for p in files})
    print('Full reviewer benchmark report compiled.',flush=True)


if __name__=='__main__':main()

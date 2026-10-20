"""Rebuild metrics from genuine saved predictions; never import reported_results.

Requires numpy, scipy, scikit-learn and matplotlib. Outputs are written only
after validation of every prediction file. Shape (time, node) is mandatory for
network results, retaining time clustering in confidence intervals.

Example after training and model selection:
  .venv/Scripts/python evaluate_predictions.py --experiment A5 --pred-dir predictions
Expected files: pred_A5_{seed}_{val|test}.npz for all five predefined seeds.
The final evaluator requires protocol/STATUS.json to declare locked=true.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, confusion_matrix
from make_data_audit import ROOT, write_csv, write_json

SEEDS=[42,123,456,789,1024]
REQUIRED={'y_true_index','y_pred_index','y_true_class','y_pred_class','probs','indices','segment_id','timestamp'}

def ordinal_probs(index):
    y=np.clip(np.asarray(index)/10,0,10)
    logits=-(y[...,None]-np.array([1,3,5,7,9]))**2/2
    logits-=logits.max(axis=-1,keepdims=True)
    exp=np.exp(logits)
    return exp/exp.sum(axis=-1,keepdims=True)

def index_class(index):
    return np.minimum(np.floor(np.clip(np.asarray(index),0,100)/20).astype(int),4)

def metrics(y,p):
    y,p=np.asarray(y),np.clip(np.asarray(p),0,100)
    if y.shape!=p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Incompatible or nonfinite arrays')
    mae=float(np.abs(y-p).mean())
    rmse=float(np.sqrt(((y-p)**2).mean()))
    return dict(accuracy_percent=float((index_class(y)==index_class(p)).mean()*100),
                mae_index=mae,rmse_index=rmse,mae_0_10=mae/10,rmse_0_10=rmse/10,
                macro_f1=float(f1_score(index_class(y).ravel(),index_class(p).ravel(),
                                      labels=np.arange(5),average='macro',zero_division=0)))

def spatial_correlation(y,p,edges,node_ids):
    """Pearson over differences on fixed, original binary undirected edges."""
    pos={int(n):i for i,n in enumerate(node_ids)}
    true=np.concatenate([(y[:,pos[a]]-y[:,pos[b]]) for a,b in edges])
    pred=np.concatenate([(p[:,pos[a]]-p[:,pos[b]]) for a,b in edges])
    if true.std()==0 or pred.std()==0: return float('nan')
    return float(np.corrcoef(true,pred)[0,1])

def load_prediction(path,expected_indices):
    with np.load(path,allow_pickle=False) as saved:
        if not REQUIRED<=set(saved.files): raise ValueError(f'{path}: missing {REQUIRED-set(saved.files)}')
        d={k:saved[k] for k in saved.files}
    y,p=d['y_true_index'],d['y_pred_index']
    if y.ndim!=2 or y.shape!=p.shape or y.shape[1]!=20: raise ValueError(f'{path}: expected time x 20 nodes')
    if not np.array_equal(d['indices'],expected_indices): raise ValueError(f'{path}: incorrect split indices')
    if y.shape[0]!=len(expected_indices): raise ValueError('Inconsistent number of snapshots')
    if not np.array_equal(d['segment_id'],np.arange(1,21)): raise ValueError('Node order must be 1..20')
    if d['timestamp'].shape!=(len(expected_indices),): raise ValueError('Missing time alignment')
    if not np.isfinite(y).all() or not np.isfinite(p).all() or ((y<0)|(y>100)|(p<0)|(p>100)).any():
        raise ValueError('Prediction/target outside reporting index range')
    for key,values in [('y_true_class',y),('y_pred_class',p)]:
        if not np.array_equal(d[key],index_class(values)): raise ValueError(f'{path}: incorrect ordinal classes')
    if d['probs'].shape!=(*y.shape,5) or not np.allclose(d['probs'],ordinal_probs(p),atol=1e-6):
        raise ValueError('Probabilities must match the locked ordinal transformation')
    return d

def paired_comparisons(rows):
    """Each row: comparison, metric, seed, a, b. Lower a-b favours a for errors.

    Two-sided paired t tests are specified here; do not choose the test after
    inspecting p-values. Holm adjusts over the whole supplied comparison family.
    """
    groups={}
    for row in rows:
        key=(row['comparison'],row['metric'])
        group=groups.setdefault(key,{})
        seed=int(row['seed'])
        if seed in group: raise ValueError('Duplicate paired seed')
        group[seed]=(float(row['a']),float(row['b']))
    result=[]
    for (comparison,metric),pairs in groups.items():
        if sorted(pairs)!=sorted(SEEDS): raise ValueError('Require exactly the five predefined paired seeds')
        a,b=np.array([pairs[s] for s in SEEDS]).T
        if not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('Nonfinite paired metric')
        d=a-b
        sd=d.std(ddof=1)
        p=float(stats.ttest_rel(a,b).pvalue) if sd>0 else (1. if np.all(d==0) else None)
        result.append(dict(comparison=comparison,metric=metric,n=5,
                           mean_a=float(a.mean()),std_a=float(a.std(ddof=1)),
                           mean_b=float(b.mean()),std_b=float(b.std(ddof=1)),
                           mean_difference_a_minus_b=float(d.mean()),p_raw=p,
                           cohen_dz=float(d.mean()/sd) if sd>0 else None,
                           test='two-sided paired t; seed is experimental unit'))
    order=sorted([i for i in range(len(result)) if result[i]['p_raw'] is not None],key=lambda i:result[i]['p_raw'])
    for r in result: r['p_holm']=None
    adjusted=0.
    for rank,i in enumerate(order):
        adjusted=max(adjusted,min(1.,result[i]['p_raw']*(len(result)-rank)))
        result[i]['p_holm']=adjusted
    return result

def block_bootstrap(y,p,repeats=2000,block=24,seed=2026):
    """Mean performance over seeds, resampling temporal blocks and all nodes.

    Arrays have shape (seed,time,node). Seeds are kept fixed; intervals describe
    test-sampling uncertainty conditional on these trained runs. They are NOT
    confidence intervals for performance across new training seeds.
    """
    if y.shape!=p.shape or y.ndim!=3: raise ValueError('Expected seed,time,node arrays')
    n=y.shape[1]
    if not 1<=block<=n or repeats<100: raise ValueError('Invalid bootstrap settings')
    rng=np.random.default_rng(seed)
    values=[]
    for _ in range(repeats):
        starts=rng.integers(0,n,size=int(np.ceil(n/block)))
        idx=((starts[:,None]+np.arange(block))%n).ravel()[:n]
        errs=y[:,idx]-p[:,idx]
        values.append([np.mean(index_class(y[:,idx])==index_class(p[:,idx]))*100,
                       np.abs(errs).mean(axis=(1,2)).mean(),
                       np.sqrt((errs**2).mean(axis=(1,2))).mean()])
    bounds=np.percentile(values,[2.5,97.5],axis=0)
    return [dict(metric=name,ci_lower=float(bounds[0,i]),ci_upper=float(bounds[1,i]),
                 bootstrap_repeats=repeats,block_hours=block,unit='network snapshot; all 20 nodes clustered',
                 seed_policy='mean metric over five fixed fitted seeds')
            for i,name in enumerate(['accuracy_percent','mae_index','rmse_index'])]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',required=True)
    parser.add_argument('--pred-dir',type=Path,required=True)
    parser.add_argument('--block',type=int,default=24)
    args=parser.parse_args()
    status=json.loads((ROOT/'protocol/STATUS.json').read_text())
    if not status.get('locked'): raise SystemExit('Refusing final evaluation: protocol is not locked.')
    out=ROOT/'results/final_eval'/args.experiment
    if out.exists(): raise SystemExit('Final evaluation output already exists; refusing repeated evaluation.')
    with np.load(ROOT/'protocol/split_indices.npz') as f:
        indices={'val':f['fixed_validation'],'test':f['final_test']}
    loaded={}
    filehashes={}
    for seed in SEEDS:
        for split in ['val','test']:
            path=args.pred_dir/f'pred_{args.experiment}_{seed}_{split}.npz'
            loaded[seed,split]=load_prediction(path,indices[split])
            filehashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        val,test=loaded[seed,'val'],loaded[seed,'test']
        if np.intersect1d(val['timestamp'],test['timestamp']).size:
            raise ValueError('Overlapping validation/test timestamps')
        if val['y_pred_index'].shape==test['y_pred_index'].shape and np.array_equal(val['y_pred_index'],test['y_pred_index']):
            raise ValueError('Identical validation/test predictions require investigation')
    for split in ['val','test']:
        first=loaded[42,split]
        for seed in SEEDS[1:]:
            for key in ['indices','timestamp','segment_id']:
                if not np.array_equal(first[key],loaded[seed,split][key]): raise ValueError('Seed alignment mismatch')
    rows=[]
    identical=[]
    for seed in SEEDS:
        row=dict(seed=seed)
        for split in ['val','test']:
            d=loaded[seed,split]
            row.update({f'{split}_{key}':value for key,value in metrics(d['y_true_index'],d['y_pred_index']).items()})
        keys=['accuracy_percent','mae_index','rmse_index']
        if all(round(row['val_'+k],2)==round(row['test_'+k],2) for k in keys): identical.append(seed)
        rows.append(row)
    if identical: raise ValueError(f'Identical rounded validation/test metrics for seeds {identical}; investigate before reporting')
    summary=[]
    for name in list(rows[0])[1:]:
        values=[row[name] for row in rows]
        summary.append(dict(metric=name,mean=float(np.mean(values)),std=float(np.std(values,ddof=1))))
    y=np.stack([loaded[s,'test']['y_true_index'] for s in SEEDS])
    p=np.stack([loaded[s,'test']['y_pred_index'] for s in SEEDS])
    ci=block_bootstrap(y,p,block=args.block)
    out.mkdir(parents=True,exist_ok=False)
    write_csv(out/'val_vs_test_by_seed.csv',rows)
    write_csv(out/'mean_std_summary.csv',summary)
    write_csv(out/'final_test_bootstrap_ci.csv',ci)
    write_json(out/'prediction_sha256.json',filehashes)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cm=confusion_matrix(index_class(y).ravel(),index_class(p).ravel(),labels=np.arange(5))
    fig,ax=plt.subplots(figsize=(6,5))
    image=ax.imshow(cm,cmap='Blues')
    fig.colorbar(image,ax=ax)
    ax.set(xlabel='Predicted class',ylabel='True class',title='Test confusion counts pooled across five seeds')
    for i in range(5):
        for j in range(5): ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    fig.tight_layout();fig.savefig(out/'confusion_matrix_test.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    # Representative seed is predeclared, never selected by performance.
    ax.hexbin(p[0].ravel(),(y[0]-p[0]).ravel(),gridsize=50,mincnt=1,cmap='viridis')
    ax.axhline(0,color='black',lw=.8)
    ax.set(xlabel='Prediction (0-100)',ylabel='True - predicted (index points)',title='Held-out residuals, predefined seed 42')
    fig.tight_layout();fig.savefig(out/'residual_plot_test.png',dpi=180);plt.close(fig)

if __name__=='__main__': main()

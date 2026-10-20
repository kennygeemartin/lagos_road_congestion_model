"""Five-seed, full-architecture reviewer benchmark extension; immutable v1 inputs.

Run --prepare, then execute without flags. A separate protocol is locked before
training. Original v1 final selection and predictions remain unchanged.
"""
import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from reconstructed_study import STUDY,verify,digest,SEEDS
from run_reconstructed_experiments import Study
from reviewer_benchmark_models import MODELS
from revision_models import seed_everything
from evaluate_predictions import metrics,ordinal_probs,index_class,spatial_correlation
from make_data_audit import ROOT,write_json

EXT=ROOT/'reviewer_benchmarks_v2'


def graphs():
    graph=json.loads((STUDY/'protocol/graph.json').read_text())
    a=np.zeros((20,20),dtype='float32')
    for edge in graph['directed_edges']:a[edge['from_node']-1,edge['to_node']-1]=1
    return a,np.maximum(a,a.T),graph['original_undirected_edges']


def prepare():
    verify(require_locked=True)
    EXT.mkdir(exist_ok=True)
    if (EXT/'protocol.json').exists():verify_extension();return
    for folder in ['checkpoints','logs','predictions','results']:(EXT/folder).mkdir(exist_ok=True)
    sources=[ROOT/name for name in ['reviewer_benchmark_models.py','run_reviewer_benchmarks.py',
        'summarize_reviewer_benchmarks.py','validate_reviewer_benchmarks.py','tests/test_reviewer_benchmarks.py']]
    sources += [p for p in (ROOT/'benchmark_vendor').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    config=dict(extension='reviewer-requested full-architecture comparison',seeds=SEEDS,
        base_protocol_sha256=digest(STUDY/'protocol/sha256.json'),
        model_ready_sha256=digest(STUDY/'data/model_ready.npz'),
        architectures={
            'STGCN':dict(blocks=[[14,32,64],[64,32,128]],spatial_kernel=3,temporal_kernel=3,output_kernel=4,dropout=.3,graph='binary symmetric; scaled normalized Laplacian Chebyshev T0,T1,T2'),
            'DCRNN':dict(encoder_layers=2,decoder_layers=2,units=64,max_diffusion_step=2,graph='directed dual random walk; transposed supports',diffusion_basis='T0 plus per-support T1,T2 Chebyshev recurrence; reset basis for each support',horizon=1,decoder='zero GO symbol; one step; scheduled sampling inapplicable'),
            'GraphWaveNet':dict(blocks=4,layers_per_block=2,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,diffusion_order=2,node_embedding=10,dropout=.3,receptive_field=13,left_padding=1,graph='forward/backward random walks plus adaptive row-softmax adjacency')},
        task=dict(features=14,input_hours=12,output='next-hour scalar congestion per node',nodes=20,chronology='ordinal reconstructed hours',data='fixed v1 model-ready inputs; train-only preprocessing'),
        training=dict(epochs=80,patience=10,batch_networks=32,optimizer='Adam',lr=.0005,weight_decay=0.,gradient_clip_norm=5.,loss='unmasked MSE on 0-10 score',selection='best raw validation MSE epoch',hyperparameter_search='none',torch_threads=4,device='cpu',checkpoint_every_epoch=True),
        fit_snapshot_ids=np.arange(4420).tolist(),validation_snapshot_ids=np.arange(4420,5009).tolist(),
        evaluation=dict(primary='fixed v1 validation; A1/A4/A5 and all three full models',
            secondary='all three fixed full architectures on previously examined v1 temporal test; retrospective supplementary comparison with existing A1 predictions',
            test_is_fresh=False,test_snapshot_ids=np.arange(5894,8420).tolist(),
            candidate_selection='no re-selection of the original A1 final configuration',
            paired_tests='two-sided paired t; five seed pairs; Holm across 3 baseline x 3 comparator x 3 metric validation comparisons; secondary A1-test family separate',
            timing='model-only; same 12-hour 20-node tensor; batch1 and20;20 warmup,100 repeats; no raw ingestion'),
        implementation=dict(STGCN='PyTorch port of reference two-block architecture; multifeature one-step task adaptation',
            DCRNN='PyTorch port of diffusion GRU encoder/decoder; polynomial basis resets independently for each direction',
            GraphWaveNet='authors source with three Conv1d-to-Conv2d compatibility declarations; in_dim14,out_dim1',
            shared_training='common training objective/optimizer/budget adapted to v1; not a reproduction of original traffic-dataset scores'),
        frozen_code_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in sources})
    # Tests finish before any fitting or lock is written.
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_reviewer_benchmarks.py','-v'],cwd=ROOT,capture_output=True,text=True)
    (EXT/'results/architecture_checks.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError(result.stderr)
    write_json(EXT/'protocol.json',config)
    write_json(EXT/'protocol_sha256.json',dict(sha256=digest(EXT/'protocol.json')))
    print('Reviewer benchmark protocol locked after architecture checks.',flush=True)


def verify_extension():
    verify(require_locked=True)
    p=json.loads((EXT/'protocol.json').read_text())
    if digest(EXT/'protocol.json')!=json.loads((EXT/'protocol_sha256.json').read_text())['sha256']:raise RuntimeError('Extension protocol changed')
    if digest(STUDY/'data/model_ready.npz')!=p['model_ready_sha256']:raise RuntimeError('Base data changed')
    if digest(STUDY/'protocol/sha256.json')!=p['base_protocol_sha256']:raise RuntimeError('Base protocol changed')
    for name,sha in p['frozen_code_sha256'].items():
        if digest(ROOT/name)!=sha:raise RuntimeError(f'Benchmark code changed after lock: {name}')
    return p


def status(message,**extra):
    print(time.strftime('%H:%M:%S')+' '+message,flush=True)
    write_json(EXT/'progress.json',dict(status='RUNNING',current=message,**extra))


def atomic_save(value,path):
    temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)


def fit(model,study,name,seed,config):
    t=config['training'];path=EXT/'checkpoints'/f'{name}_{seed}.pt';latest=path.with_suffix('.resume.pt')
    signature=digest(EXT/'protocol.json')
    optimizer=torch.optim.Adam(model.parameters(),lr=t['lr'])
    generator=torch.Generator().manual_seed(seed)
    train=DataLoader(TensorDataset(study.x[study.tr],study.y[study.tr]),batch_size=t['batch_networks'],shuffle=True,generator=generator)
    valid=DataLoader(TensorDataset(study.x[study.val],study.y[study.val]),batch_size=64)
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['protocol_sha256']!=signature:raise RuntimeError('Checkpoint protocol mismatch')
        model.load_state_dict(saved['state_dict']);return model.eval(),saved
    first=0;best=float('inf');best_state=None;stale=0;history=[];elapsed=0.
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=False)
        if saved['protocol_sha256']!=signature:raise RuntimeError('Resume protocol mismatch')
        model.load_state_dict(saved['current_state']);optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['generator']);torch.set_rng_state(saved['torch_rng'])
        np.random.set_state(saved['numpy_rng']);first=saved['next_epoch'];history=saved['history']
        best=saved['best'];best_state=saved['best_state'];stale=saved['stale'];elapsed=saved['elapsed']
    for epoch in range(first,t['epochs']):
        if stale>=t['patience']:break
        tic=time.perf_counter();model.train();total=count=0
        for xb,yb in train:
            optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.mse_loss(model(xb),yb)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),t['gradient_clip_norm']);optimizer.step()
            total+=loss.item()*len(xb);count+=len(xb)
        model.eval();vt=vc=0
        with torch.inference_mode():
            for xb,yb in valid:vt+=torch.nn.functional.mse_loss(model(xb),yb).item()*len(xb);vc+=len(xb)
        value=vt/vc
        if not np.isfinite(value):raise RuntimeError('Nonfinite validation loss')
        seconds=time.perf_counter()-tic;elapsed+=seconds
        history.append(dict(epoch=epoch+1,train_mse=total/count,val_mse=value,seconds=seconds))
        if value<best:best=value;best_state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        pd.DataFrame(history).to_csv(EXT/'logs'/f'{name}_{seed}.csv',index=False)
        atomic_save(dict(protocol_sha256=signature,current_state=model.state_dict(),optimizer=optimizer.state_dict(),
            generator=generator.get_state(),torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),next_epoch=epoch+1,
            history=history,best=best,best_state=best_state,stale=stale,elapsed=elapsed),latest)
        status(f'{name} seed={seed} epoch={epoch+1}/80 val_MSE={value:.6f} seconds={seconds:.2f}')
    model.load_state_dict(best_state);model.eval()
    saved=dict(state_dict=model.state_dict(),protocol_sha256=signature,history=history,training_seconds=elapsed,
        best_epoch=min(history,key=lambda r:r['val_mse'])['epoch'],train_snapshot_ids=study.tr.tolist(),validation_snapshot_ids=study.val.tolist(),seed=seed,name=name)
    atomic_save(saved,path)
    return model,saved


def predict_and_score(model,study,name,seed,ids,split):
    path=EXT/'predictions'/f'pred_{name}_{seed}_{split}.npz'
    if path.exists():
        with np.load(path) as z:
            if not np.array_equal(z['indices'],ids):raise RuntimeError('Prediction IDs changed')
            true=z['y_true_index'];pred=z['y_pred_index']
    else:
        raw,_=study.predict(model,study.x,ids,batch=64);pred=np.clip(raw,0,10)*10;true=study.y[ids].numpy()*10
        np.savez_compressed(path,y_true_index=true,y_pred_index=pred,y_true_class=index_class(true),y_pred_class=index_class(pred),
            probs=ordinal_probs(pred),indices=ids,segment_id=np.arange(1,21),timestamp=study.times[ids],
            evaluation_status=np.array('primary development comparison' if split=='val' else 'retrospective supplementary comparison; v1 holdout previously examined'))
    return dict(experiment=name,seed=seed,split=split,**metrics(true,pred),
        scs=spatial_correlation(true,pred,study.edges,np.arange(1,21)))


def timing(model,x,name,seed):
    rows=[];model.eval()
    with torch.inference_mode():
        for batch in [1,20]:
            sample=x[:batch].contiguous();values=[]
            for i in range(120):
                tic=time.perf_counter_ns();model(sample);ms=(time.perf_counter_ns()-tic)/1e6
                if i>=20:values.append(ms)
            rows.append(dict(experiment=name,seed=seed,batch_networks=batch,mean_ms=np.mean(values),std_ms=np.std(values,ddof=1),
                p50_ms=np.percentile(values,50),p95_ms=np.percentile(values,95),p99_ms=np.percentile(values,99),runs=100,warmup=20,
                parameters=sum(p.numel() for p in model.parameters()),scope='model-only; no JSON ingestion or scaling',device='cpu'))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare();return
    config=verify_extension();torch.set_num_threads(4)
    study=Study();directed,symmetric,_=graphs();rows=[];times=[]
    for seed in SEEDS:
        for name,constructor in MODELS.items():
            seed_everything(seed);model=constructor(symmetric if name=='STGCN' else directed)
            status(f'Starting {name} seed={seed}')
            model,saved=fit(model,study,name,seed,config)
            row=predict_and_score(model,study,name,seed,study.val,'val')
            row.update(training_seconds=saved['training_seconds'],epochs=len(saved['history']),best_epoch=saved['best_epoch'],parameters=sum(p.numel() for p in model.parameters()))
            rows.append(row);pd.DataFrame(rows).to_csv(EXT/'results/validation_by_seed.csv',index=False)
            times.extend(timing(model,study.x,name,seed));pd.DataFrame(times).to_csv(EXT/'results/model_timing_by_seed.csv',index=False)
    # All fits and validation selections finish before supplementary test inference.
    write_json(EXT/'results/RETROSPECTIVE_TEST_STATUS.json',dict(previously_examined=True,used_for_training=False,
        used_for_epoch_selection=False,original_final_selection_unchanged=True,benchmarks=list(MODELS),seeds=SEEDS))
    test=[]
    for seed in SEEDS:
        for name,constructor in MODELS.items():
            seed_everything(seed);model=constructor(symmetric if name=='STGCN' else directed)
            saved=torch.load(EXT/'checkpoints'/f'{name}_{seed}.pt',map_location='cpu',weights_only=True);model.load_state_dict(saved['state_dict']);model.eval()
            test.append(predict_and_score(model,study,name,seed,study.test,'test'))
    pd.DataFrame(test).to_csv(EXT/'results/retrospective_test_by_seed.csv',index=False)
    write_json(EXT/'results/environment.json',dict(python=sys.version,torch=torch.__version__,platform=platform.platform(),processor=platform.processor(),threads=4,cuda=torch.cuda.is_available()))
    subprocess.run([sys.executable,'summarize_reviewer_benchmarks.py'],cwd=ROOT,check=True)
    write_json(EXT/'progress.json',dict(status='COMPLETED',current='All 15 full-architecture fits, predictions, timings, comparisons and checks completed'))
    print('REVIEWER BENCHMARKS COMPLETED',flush=True)


if __name__=='__main__':main()

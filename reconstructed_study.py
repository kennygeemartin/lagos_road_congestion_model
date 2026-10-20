"""Prepare/lock the newly authorized reconstructed-data study (not an original rerun).

The original archive and its split file are never changed. All new study outputs
live under reconstructed_v1/. Fixed archived data; five MODEL seeds, not five
claimed independent synthetic-data generations. Chronology is explicitly ordinal.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from make_data_audit import ROOT,SOURCE,write_json
from experiment_controls import SEEDS,canonical_edges

STUDY=ROOT/'reconstructed_v1'
FEATURES=['speed_kmh','travel_delay_min','acceleration_ms2','gps_speed_noise_kmh',
          'sensor_count','hour_sin','hour_cos','weekday_sin','weekday_cos','peak_indicator',
          'rolling_ratio_mean_12','rolling_ratio_std_12','volume_intensity_veh_h_per_km','volume_capacity_ratio']

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare():
    protocol=STUDY/'protocol';data=STUDY/'data'
    if (protocol/'sha256.json').exists():
        verify();print('Existing immutable reconstructed_v1 study verified; not regenerated.');return
    protocol.mkdir(parents=True,exist_ok=True);data.mkdir(exist_ok=True)
    generated=pd.read_csv(SOURCE/'data/generated_segment_volume_series.csv').sort_values(['time_index','segment_id'])
    sensors=pd.read_csv(SOURCE/'data/synthetic_sensor_dataset.csv')
    split=pd.read_csv(SOURCE/'splits/chronological_split_indices.csv')
    template=pd.read_csv(SOURCE/'data/temporal_template_proxy.csv')
    nodes=pd.read_csv(SOURCE/'graph/graph_nodes.csv')
    edges=pd.read_csv(SOURCE/'graph/graph_edges.csv')
    t=len(template);n=len(nodes)
    if t!=8432 or n!=20 or len(generated)!=t*n:raise ValueError('Unexpected source shape')
    if generated[['time_index','segment_id']].duplicated().any():raise ValueError('Duplicate node-hours')
    ratio=generated.volume_capacity_ratio.to_numpy().reshape(t,n)
    y=generated.congestion_score_0_10.to_numpy().reshape(t,n).astype('float32')
    times=split.target_time_index.to_numpy()
    initial=split[split.partition=='initial_train'].sequence_index.to_numpy()
    input_stop=int(times[initial].max()) # exclusive: all training INPUT hours 0..4430
    columns=['speed_kmh','travel_delay_min','acceleration_ms2','gps_speed_noise_kmh']
    aggregated=sensors.groupby(['time_index','segment_id'])[columns].mean()
    grid=pd.MultiIndex.from_product([range(t),range(1,n+1)],names=['time_index','segment_id'])
    grid_values=aggregated.reindex(grid)
    raw_sensor=grid_values.to_numpy().reshape(t,n,4)
    medians=np.nanmedian(raw_sensor[:input_stop],axis=0)
    if not np.isfinite(medians).all():raise ValueError('A node has no training sensor data')
    filled=raw_sensor.copy()
    for node in range(n):
        filled[:,node,:]=pd.DataFrame(raw_sensor[:,node,:]).ffill().fillna(pd.Series(medians[node])).to_numpy()
    counts=sensors.groupby(['time_index','segment_id']).size().reindex(grid,fill_value=0).to_numpy().reshape(t,n)
    hour=template.hour_of_day.to_numpy();weekday=template.day_of_week.to_numpy()
    repeat=lambda a:np.repeat(a[:,None],n,axis=1)
    rolling=pd.DataFrame(ratio).rolling(12,min_periods=1)
    raw=np.stack([*[filled[:,:,i] for i in range(4)],counts,
        repeat(np.sin(2*np.pi*hour/24)),repeat(np.cos(2*np.pi*hour/24)),
        repeat(np.sin(2*np.pi*weekday/7)),repeat(np.cos(2*np.pi*weekday/7)),
        repeat(((hour>=7)&(hour<10))|((hour>=16)&(hour<19))),
        rolling.mean().to_numpy(),rolling.std(ddof=0).fillna(0).to_numpy(),
        generated.volume_intensity_veh_h_per_km.to_numpy().reshape(t,n),ratio],axis=-1).astype('float32')
    train_values=raw[:input_stop]
    minimum=train_values.min(axis=(0,1));maximum=train_values.max(axis=(0,1))
    scale=np.where(maximum>minimum,maximum-minimum,1)
    scaled=((raw-minimum)/scale).astype('float32')
    speed10=np.quantile(train_values[:,:,0],.1,axis=0)
    intensity90=np.quantile(train_values[:,:,12],.9,axis=0)
    proxy=((raw[:,:,0]<speed10)&(raw[:,:,12]>intensity90)&(raw[:,:,1]>15)).astype('int64')
    arrays=dict(features=scaled,raw_features=raw,y_hourly=y,ratio=ratio.astype('float32'),
                proxy_hourly=proxy,scaler_min=minimum,scaler_max=maximum,scaler_scale=scale,
                sensor_train_medians=medians.astype('float32'),target_time_index=times,
                node_ids=np.arange(1,n+1),train_input_stop=np.array(input_stop))
    for p in ['initial_train','fixed_validation','acquisition_pool','final_test']:
        arrays[p]=split.loc[split.partition==p,'sequence_index'].to_numpy()
    np.savez_compressed(data/'model_ready.npz',**arrays)
    # Source split is copied byte-for-byte; NPZ retains exactly the same IDs.
    (protocol/'chronological_split_indices.csv').write_bytes((SOURCE/'splits/chronological_split_indices.csv').read_bytes())
    np.savez_compressed(protocol/'split_indices.npz',**{p:arrays[p] for p in ['initial_train','fixed_validation','acquisition_pool','final_test']},target_time_index=times)
    a=np.zeros((n,n),dtype='float32')
    for row in edges.itertuples():a[row.from_node-1,row.to_node-1]=1
    binary=((a+a.T)>0).astype('float32');hat=binary+np.eye(n,dtype='float32');degree=hat.sum(1)
    norm=hat/np.sqrt(degree[:,None]*degree[None,:])
    write_json(protocol/'graph.json',dict(nodes=nodes.to_dict('records'),directed_edges=edges.to_dict('records'),
        original_undirected_edges=sorted(canonical_edges(edges.to_dict('records'))),
        binary_adjacency=binary.tolist(),normalized_adjacency=norm.tolist(),
        rule='B = ((A+A.T)>0); A_hat=B+I; normalized=D^(-1/2) A_hat D^(-1/2)',
        scs_edges='same original undirected edges for all graph controls'))
    write_json(protocol/'seeds.json',SEEDS)
    write_json(protocol/'feature_config.json',dict(columns=FEATURES,L=12,stride=1,
        sample_unit='one 12-hour whole-network snapshot; 20 node targets',
        chronological_index='ordinal reconstructed proxy time_index; NOT original Metro date_time',
        sensor_aggregation='arithmetic mean per node/hour; no claimed one-minute trajectory chronology',
        missing_sensors='causal forward-fill within node; initial fallback to node-specific TRAIN median',
        sensor_count='observed records per node/hour, zero if absent; not forward-filled',
        rolling='inclusive trailing 12 observed hourly ratios; ddof=0; min_periods=1',
        scaler='one global min-max transform per feature; only unique initial-training INPUT hours',
        scaler_fit_hours=[0,input_stop-1],out_of_range_features='not clipped',
        seeds='fixed reconstructed data; only model/selector seeds vary',
        static_nigerian_context='not used; unavailable in archive',
        source_generation='archived reconstructed proxy and archived generated volumes retained unchanged'))
    write_json(protocol/'label_rule.json',dict(ratio_breakpoints=[0,.4,.6,.75,.9,1],score_breakpoints=[0,2,4,6,8,10],
        index_multiplier=10,class_index_boundaries=[20,40,60,80],ae_threshold=.015,
        proxy='hourly assembled speed < training-node-p10 AND intensity > training-node-p90 AND delay >15',
        proxy_fit_input_hours=[0,input_stop-1],speed_p10=speed10.tolist(),intensity_p90=intensity90.tolist(),
        note='NEW hourly proxy labels; not the archived sensor-row proxy labels',
        oracle='one acquired snapshot exposes all 20 generated node targets',
        stage_snapshot_budgets=[0,295,590,885],stage_scalar_label_budgets=[0,5900,11800,17700],
        ae_threshold_selection='theta=.015 is pre-specified; percentile/Youden diagnostics use TRAIN only'))
    models=json.loads((SOURCE/'config/model_config.json').read_text())
    models.update(dict(training=dict(device='cpu',torch_threads=4,lstm_patience=10,
        lstm_batch_unit='32 whole-network snapshots (640 segment sequences) per optimizer batch',
        graph_batch_unit='64 snapshots',loss='MSE on original 0-10 scale',
        ae_normal_fit='initial-training input hours with ratio<.4; first80% fit,last20% training-internal early-stop',
        graph_training='two-stage; frozen same-seed LSTM predictions plus last observed 14 features; A4 also anomaly',
        active_retraining='same 10 final-epoch LSTM and GCN epochs per round; warm start; same optimizer-reset policy for both selectors',
        active_entropy='mean node entropy from LSTM ordinal probabilities, query one whole snapshot',
        active_backbone_dropout='same deterministic per-stage seed for both selectors',
        final_selection='minimum five-seed mean validation MAE among A1-A4 and entropy/random stages; no test selection'),
        benchmark=dict(name='STGCN-lite',channels=32,epochs=80,patience=10,lr=.0005,
            adaptation='single T-G-T block, first-order graph convolution; omits higher-order filters and second block',
            evaluation='fixed validation only; does not reopen final test'),
        baseline=dict(HA='node x hour-of-week mean from training target hours, node-mean fallback',
            persistence='last observed hourly congestion score',ARIMA='(2,1,2), fixed train-only parameters; rolling one-step state update on observed past validation values',
            arima_maxiter=100),statistics=dict(test='paired two-sided t across five model seeds',
            correction='Holm within each experiment family across all reported pairwise metric tests',
            bootstrap='2000 circular temporal block resamples; 24-hour blocks; all nodes clustered; five fitted seeds held fixed'),
        latency=dict(warmup=20,runs=100,batch_sizes=[1,20],batch_unit='networks; each network contains20segments',
            hardware='actual CPU only; not RTX3090; not directly comparable to historical GPU latency')))
    write_json(protocol/'model_config.yaml',models) # JSON is valid YAML
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    write_json(protocol/'environment.lock',dict(python=sys.version,platform=platform.platform(),packages=packages))
    sources={str(p.relative_to(ROOT)):digest(p) for p in SOURCE.rglob('*') if p.is_file()}
    write_json(protocol/'source_sha256.json',sources)
    pd.DataFrame([dict(partition=p,snapshots=len(arrays[p]),node_targets=len(arrays[p])*n,
        proxy_positives=int(proxy[times[arrays[p]]].sum()),proxy_node_hours=len(arrays[p])*n)
        for p in ['initial_train','fixed_validation','acquisition_pool','final_test']]).to_csv(data/'split_counts.csv',index=False)
    write_json(protocol/'STATUS.json',dict(locked=False,study='new reconstructed-data simulation',
        authorized='user yes following offered reconstructed-study option',
        original_study_reproduced=False,chronology='ordinal proxy',sample_unit='20-node network snapshot',
        source_split_modified=False,total_snapshots=8420,total_node_targets=168400))
    print('Prepared new study. Run reconstructed protocol tests before locking.',flush=True)

def verify(require_locked=False):
    p=STUDY/'protocol'
    if require_locked and not json.loads((p/'STATUS.json').read_text())['locked']:raise RuntimeError('Study is not locked')
    if (p/'sha256.json').exists():
        for name,sha in json.loads((p/'sha256.json').read_text()).items():
            if digest(STUDY/Path(name.replace('\\','/')))!=sha:raise RuntimeError(f'Immutable study changed: {name}')
    for name,sha in json.loads((p/'source_sha256.json').read_text()).items():
        if digest(ROOT/Path(name.replace('\\','/')))!=sha:raise RuntimeError(f'Archive source changed: {name}')

def lock():
    import subprocess
    verify()
    if (STUDY/'protocol/sha256.json').exists():print('Already locked and verified');return
    subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_reconstructed_protocol.py','-v'],cwd=ROOT,check=True)
    path=STUDY/'protocol/STATUS.json';status=json.loads(path.read_text());status['locked']=True;write_json(path,status)
    files=[p for folder in ['protocol','data'] for p in (STUDY/folder).rglob('*') if p.is_file()]
    write_json(STUDY/'protocol/sha256.json',{str(p.relative_to(STUDY)):digest(p) for p in files})
    print('New reconstructed-data protocol LOCKED; archived protocol untouched.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','lock','verify']);a=parser.parse_args()
    {'prepare':prepare,'lock':lock,'verify':verify}[a.action]()

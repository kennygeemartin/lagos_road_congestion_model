"""Execute the locked reconstructed-v1 study; resume from verified checkpoints.

All experiment comparisons use validation. Only the configuration selected from
mean validation MAE is evaluated on test. Reported historical results are never
loaded. New results are not reproductions of the original manuscript numbers.
"""
import argparse
import copy
import json
import math
import os
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import precision_recall_fscore_support
from reconstructed_study import STUDY,SEEDS,verify,digest
from make_data_audit import write_json,write_csv,ROOT
from experiment_controls import entropy_select,random_schedule
from evaluate_predictions import metrics,ordinal_probs,index_class,spatial_correlation,paired_comparisons,block_bootstrap
from revision_models import Autoencoder,TemporalLSTM,GraphAdjustment,STGCNLite,seed_everything,latency

OUT=STUDY/'results';CKPT=STUDY/'checkpoints';PRED=STUDY/'predictions';LOG=STUDY/'logs'

def announce(message):
    print(time.strftime('%H:%M:%S')+' '+message,flush=True)

class Study:
    def __init__(self):
        verify(require_locked=True)
        torch.set_num_threads(4)
        for path in [OUT,CKPT,PRED,LOG]:path.mkdir(parents=True,exist_ok=True)
        with np.load(STUDY/'data/model_ready.npz') as z:self.d={k:z[k] for k in z.files}
        self.config=json.loads((STUDY/'protocol/model_config.yaml').read_text())
        self.graph=json.loads((STUDY/'protocol/graph.json').read_text())
        self.a=np.array(self.graph['normalized_adjacency'],dtype='float32')
        self.edges=self.graph['original_undirected_edges']
        self.times=self.d['target_time_index'];self.dev=np.arange(5894)
        self.tr=self.d['initial_train'];self.val=self.d['fixed_validation'];self.pool=self.d['acquisition_pool'];self.test=self.d['final_test']
        self.y=torch.from_numpy(self.d['y_hourly'][self.times].copy())
        self.x=torch.from_numpy(np.stack([self.d['features'][t-12:t] for t in self.times]))
        self.protocol_sha=digest(STUDY/'protocol/sha256.json')
        self.rows=[];self.ae_rows=[];self.graph_rows=[];self.al_rows=[];self.benchmark_rows=[]
        self.runtime=[];self.xae={};self.ae={};self.errors={}

    def checkpoint(self,name,seed):return CKPT/f'{name}_{seed}.pt'

    def train(self,model,x,y,train_ids,val_ids,name,seed,epochs,lr,batch,patience=10,fixed=False):
        checkpoint=self.checkpoint(name,seed)
        if checkpoint.exists():
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
            if saved['protocol_sha']!=self.protocol_sha:raise RuntimeError('Checkpoint from another protocol')
            if saved['train_snapshot_ids']!=np.asarray(train_ids).tolist() or saved['validation_snapshot_ids']!=np.asarray(val_ids).tolist():
                raise RuntimeError(f'Checkpoint fitting/validation IDs changed: {name}; refusing unsafe resume')
            model.load_state_dict(saved['state_dict']);model.eval()
            return model,saved['train_seconds'],saved['history']
        seed_everything(seed)
        generator=torch.Generator().manual_seed(seed)
        train=DataLoader(TensorDataset(x[train_ids],y[train_ids]),batch_size=batch,shuffle=True,generator=generator)
        valid=DataLoader(TensorDataset(x[val_ids],y[val_ids]),batch_size=batch*2)
        optimizer=torch.optim.Adam(model.parameters(),lr=lr,betas=(.9,.999))
        best=float('inf');state=None;stale=0;history=[];started=time.perf_counter()
        for epoch in range(epochs):
            tic=time.perf_counter();model.train();total=0.;count=0
            for xb,yb in train:
                optimizer.zero_grad(set_to_none=True)
                loss=torch.nn.functional.mse_loss(model(xb),yb)
                if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss: {name}')
                loss.backward();optimizer.step();total+=loss.item()*len(xb);count+=len(xb)
            model.eval();vt=0.;vc=0
            with torch.inference_mode():
                for xb,yb in valid:
                    vt+=torch.nn.functional.mse_loss(model(xb),yb).item()*len(xb);vc+=len(xb)
            value=vt/vc
            history.append(dict(epoch=epoch+1,train_mse=total/count,val_mse=value,seconds=time.perf_counter()-tic))
            pd.DataFrame(history).to_csv(LOG/f'{name}_{seed}.csv',index=False)
            if name.startswith('REPRO_'):
                from reconstructed_progress import refresh
                refresh()
            announce(f'{name} seed={seed} epoch={epoch+1}/{epochs} val_MSE={value:.6f} seconds={time.perf_counter()-tic:.2f}')
            if value<best:
                best=value;state=copy.deepcopy(model.state_dict());stale=0
            else:stale+=1
            if not fixed and stale>=patience:break
        elapsed=time.perf_counter()-started
        if not fixed:model.load_state_dict(state)
        model.eval()
        saved=dict(state_dict=model.state_dict(),protocol_sha=self.protocol_sha,train_seconds=elapsed,history=history,
                   name=name,seed=seed,train_snapshot_ids=np.asarray(train_ids).tolist(),validation_snapshot_ids=np.asarray(val_ids).tolist())
        temporary=checkpoint.with_suffix('.tmp');torch.save(saved,temporary);temporary.replace(checkpoint)
        return model,elapsed,history

    def predict(self,model,x,ids,batch=128):
        model.eval();values=[];tic=time.perf_counter()
        with torch.inference_mode():
            for start in range(0,len(ids),batch):values.append(model(x[ids[start:start+batch]]).numpy())
        return np.concatenate(values),time.perf_counter()-tic

    def save_prediction(self,name,seed,ids,pred,split='val'):
        true=self.y[ids].numpy()*10;pred=np.clip(pred,0,10)*10
        path=PRED/f'pred_{name}_{seed}_{split}.npz'
        np.savez_compressed(path,y_true_index=true,y_pred_index=pred,y_true_class=index_class(true),
            y_pred_class=index_class(pred),probs=ordinal_probs(pred),indices=np.asarray(ids),
            segment_id=np.arange(1,21),timestamp=self.times[ids],
            chronology=np.array('ordinal reconstructed proxy; not inherited Metro date_time'))
        return metrics(true,pred)

    def score(self,name,seed,model,x,train_seconds,family='ablation',extra=None):
        p,inference=self.predict(model,x,self.val)
        row=dict(experiment=name,seed=seed,evaluation_partition='fixed_validation',
            **self.save_prediction(name,seed,self.val,p),scs=spatial_correlation(self.y[self.val].numpy()*10,np.clip(p,0,10)*10,self.edges,np.arange(1,21)),
            training_seconds=train_seconds,inference_seconds=inference,
            parameters=sum(p.numel() for p in model.parameters()))
        if extra:row.update(extra)
        if family=='ablation':self.rows.append(row)
        elif family=='graph':self.graph_rows.append(row)
        elif family=='al':self.al_rows.append(row)
        elif family=='benchmark':self.benchmark_rows.append(row)
        return row

    def autoencoder(self,seed):
        seed_everything(seed);model=Autoencoder(14)
        stop=int(self.d['train_input_stop'])
        raw=torch.from_numpy(self.d['features'])
        normal=self.d['ratio'][:stop].ravel()<.4
        free=torch.from_numpy(self.d['features'][:stop].reshape(-1,14)[normal].copy())
        cut=int(.8*len(free));normal_ids=np.arange(len(free))
        model,seconds,history=self.train(model,free,free,normal_ids[:cut],normal_ids[cut:],
            'AE',seed,100,.001,64,10)
        with torch.inference_mode():errors=model.anomaly_score(raw).numpy()
        self.ae[seed]=model;self.errors[seed]=errors
        augmented=np.concatenate([self.d['features'],errors[...,None]],axis=-1)
        self.xae[seed]=torch.from_numpy(np.stack([augmented[t-12:t] for t in self.times]))
        labels=self.d['proxy_hourly'][self.times[self.val]].ravel()
        val_errors=errors[self.times[self.val]].ravel();pred=val_errors>.015
        precision,recall,f1,_=precision_recall_fscore_support(labels,pred,average='binary',zero_division=np.nan)
        self.ae_rows.append(dict(seed=seed,mse=float(val_errors.mean()),precision=float(precision),recall=float(recall),f1=float(f1),
            positive_support=int(labels.sum()),n=len(labels),threshold=.015,training_seconds=seconds,agreement='NEW hourly rule-based proxy'))
        grid=[];train_err=errors[:stop].ravel();train_labels=self.d['proxy_hourly'][:stop].ravel()
        for percentile in range(80,100):
            threshold=float(np.percentile(train_err,percentile));positive=train_err>threshold
            tpr=float(positive[train_labels==1].mean()) if np.any(train_labels==1) else np.nan
            tnr=float((~positive[train_labels==0]).mean())
            grid.append(dict(seed=seed,percentile=percentile,threshold=threshold,youden_j=tpr+tnr-1,
                selection_data='initial training inputs only',used_for_test_threshold=False,fixed_test_threshold=.015))
        write_csv(OUT/'ae'/f'threshold_selection_{seed}.csv',grid)
        np.savez_compressed(OUT/'ae'/f'reconstruction_errors_{seed}.npz',train_errors=errors[:stop],
            val_errors=errors[self.times[self.val]],val_labels=labels,threshold=np.array(.015))
        return model,seconds

    def backbone(self,seed,anomaly=False):
        seed_everything(seed);model=TemporalLSTM(15 if anomaly else 14)
        x=self.xae[seed] if anomaly else self.x
        return self.train(model,x,self.y,self.tr,self.val,'A2' if anomaly else 'A1',seed,80,.0005,32,10)[:2]

    def graph_inputs(self,backbone,seed,anomaly,ids=None):
        ids=self.dev if ids is None else np.asarray(ids)
        x=self.xae[seed] if anomaly else self.x
        prediction,_=self.predict(backbone,x,ids)
        last=x[ids,-1].numpy()
        feature=np.concatenate([np.clip(prediction,0,10)[...,None],last],axis=-1).astype('float32')
        # Dev IDs are 0..5893; test-only arrays use local IDs.
        return torch.from_numpy(feature)

    def graph_fit(self,seed,backbone,anomaly,name,adjacency=None):
        x=self.graph_inputs(backbone,seed,anomaly)
        seed_everything(seed);model=GraphAdjustment(self.a if adjacency is None else adjacency,x.shape[-1])
        model,seconds,_=self.train(model,x,self.y,self.tr,self.val,name,seed,50,.001,64,10)
        return model,x,seconds

    def classical(self):
        from statsmodels.tsa.arima.model import ARIMA
        output=OUT/'baselines';output.mkdir(exist_ok=True)
        path=output/'classical_predictions.npz'
        if path.exists():
            with np.load(path) as z:pred={k:z[k] for k in z.files}
        else:
            hourly=self.d['y_hourly'];train_times=self.times[self.tr];val_times=self.times[self.val]
            how=np.arange(len(hourly))%168
            fallback=hourly[train_times].mean(0)
            average={h:hourly[train_times[how[train_times]==h]].mean(0) if np.any(how[train_times]==h) else fallback for h in range(168)}
            ha=np.stack([average[h] for h in how[val_times]])
            persistence=hourly[val_times-1]
            arima=np.empty_like(ha);warning_rows=[]
            for node in range(20):
                announce(f'ARIMA node {node+1}/20')
                with warnings.catch_warnings(record=True) as records:
                    warnings.simplefilter('always')
                    fitted=ARIMA(hourly[:train_times[-1]+1,node],order=(2,1,2)).fit(method_kwargs={'maxiter':100})
                    extended=fitted.extend(hourly[val_times,node])
                    arima[:,node]=extended.get_prediction().predicted_mean
                    warning_rows.extend(dict(node=node+1,message=str(w.message)) for w in records)
                if not np.isfinite(arima[:,node]).all():raise RuntimeError('Nonfinite ARIMA predictions')
            write_json(output/'arima_warnings.json',warning_rows)
            pred=dict(HA=ha,Persistence=persistence,ARIMA=arima)
            np.savez_compressed(path,**pred)
        rows=[]
        for name,values in pred.items():
            for seed in SEEDS:rows.append(dict(experiment=name,seed=seed,evaluation_partition='fixed_validation',**self.save_prediction(name,seed,self.val,values)))
        return rows

    def active(self,seed,base_lstm,base_graph,initial_seconds):
        stage_rows=[]
        for selector in ['entropy','random']:
            backbone=copy.deepcopy(base_lstm);head=copy.deepcopy(base_graph)
            train_ids=self.tr.copy();remaining=self.pool.tolist();cumulative=0
            schedule=random_schedule(self.pool.tolist(),seed)
            for stage in range(4):
                name=f'A5_{selector}_{stage*295}'
                if stage:
                    if selector=='random':chosen=schedule[stage-1]
                    else:
                        p,_=self.predict(backbone,self.xae[seed],np.array(remaining))
                        probabilities=ordinal_probs(np.clip(p,0,10)*10)
                        chosen=entropy_select(remaining,{i:probabilities[j].tolist() for j,i in enumerate(remaining)},295)
                    if len(set(chosen))!=295 or not set(chosen)<=set(remaining):raise RuntimeError('Invalid acquisition')
                    remaining=[i for i in remaining if i not in set(chosen)]
                    train_ids=np.concatenate([train_ids,np.asarray(chosen)])
                    write_json(OUT/'active_learning'/f'acquired_{selector}_{seed}_{stage}.json',dict(snapshot_ids=chosen,
                        scalar_labels=len(chosen)*20,remaining_pool=remaining))
                    # Same stage seed and reset Adam policy for both selectors.
                    backbone,seconds,_=self.train(backbone,self.xae[seed],self.y,train_ids,self.val,
                        name+'_LSTM',seed,10,.0005,32,10,fixed=True)
                    graph_x=self.graph_inputs(backbone,seed,True)
                    head,gs,_=self.train(head,graph_x,self.y,train_ids,self.val,name+'_GCN',seed,10,.001,64,10,fixed=True)
                    cumulative+=seconds+gs
                else:graph_x=self.graph_inputs(backbone,seed,True)
                row=self.score(name,seed,head,graph_x,initial_seconds+cumulative,family='al',
                    extra=dict(selector=selector,stage=stage,cumulative_snapshot_labels=stage*295,cumulative_scalar_labels=stage*5900))
                stage_rows.append(row)
        return stage_rows

    def run_development(self,finalize=True):
        for folder in ['ae','baselines','ablation','graph_validity','active_learning','benchmark']: (OUT/folder).mkdir(exist_ok=True)
        classical=self.classical()
        for seed in SEEDS:
            announce(f'BEGIN SEED {seed}')
            _,ae_seconds=self.autoencoder(seed)
            a1,s1=self.backbone(seed,False);self.score('A1',seed,a1,self.x,s1)
            a2,s2=self.backbone(seed,True);self.score('A2',seed,a2,self.xae[seed],s2+ae_seconds)
            a3,x3,g3=self.graph_fit(seed,a1,False,'A3');self.score('A3',seed,a3,x3,s1+g3)
            a4,x4,g4=self.graph_fit(seed,a2,True,'A4');self.score('A4',seed,a4,x4,s2+ae_seconds+g4)
            # Save incremental real metrics before expensive controls.
            self.flush(classical)
            for label,adj in [('G0',np.eye(20,dtype='float32')),('G1',self.a)]:
                if label=='G1':head,gx,gs=a4,x4,g4
                else:head,gx,gs=self.graph_fit(seed,a2,True,'G0',adj)
                self.score(label,seed,head,gx,s2+ae_seconds+gs,family='graph',extra=dict(rewire=-1))
            for replicate in range(10):
                control=json.loads((ROOT/'results/graph_validity/controls'/f'rewired_{seed}_{replicate:02}.json').read_text())
                support=np.eye(20,dtype='float32')
                for a,b in control['rewired_undirected_edges']:support[a-1,b-1]=support[b-1,a-1]=1
                degree=support.sum(1);adj=support/np.sqrt(degree[:,None]*degree[None,:])
                name=f'G2_{replicate:02}';head,gx,gs=self.graph_fit(seed,a2,True,name,adj)
                self.score(name,seed,head,gx,s2+ae_seconds+gs,family='graph',extra=dict(rewire=replicate))
                self.flush(classical)
            self.active(seed,a2,a4,s2+ae_seconds+g4)
            self.flush(classical)
            # Release the per-seed augmented window tensor before next seed.
            del self.xae[seed]
        if not finalize:
            announce('Development families completed; final test remains sealed.');return
        self.select_and_test()
        self.benchmark()
        self.flush(classical)
        announce('ALL REQUESTED STUDY FAMILIES COMPLETED')

    def flush(self,classical):
        if self.ae_rows:pd.DataFrame(self.ae_rows).to_csv(OUT/'ae/metrics_by_seed.csv',index=False)
        if self.rows:
            frame=pd.DataFrame(self.rows);frame.to_csv(OUT/'ablation/metrics_by_seed.csv',index=False)
            frame.groupby('experiment')[['accuracy_percent','mae_index','rmse_index','scs']].agg(['mean','std']).to_csv(OUT/'ablation/mean_std_summary.csv')
            baseline=pd.concat([pd.DataFrame(classical),frame[frame.experiment.isin(['A1','A2'])]],ignore_index=True)
            baseline.to_csv(OUT/'baselines/dev_metrics_by_seed.csv',index=False)
        if self.graph_rows:pd.DataFrame(self.graph_rows).to_csv(OUT/'graph_validity/original_vs_rewired_by_seed.csv',index=False)
        if self.al_rows:
            frame=pd.DataFrame(self.al_rows);frame.to_csv(OUT/'active_learning/metrics_by_stage_by_seed.csv',index=False)
            frame.groupby(['selector','stage','cumulative_snapshot_labels','cumulative_scalar_labels'])[['accuracy_percent','mae_index','rmse_index']].agg(['mean','std']).to_csv(OUT/'active_learning/learning_curve_entropy_vs_random.csv')
        if self.benchmark_rows:pd.DataFrame(self.benchmark_rows).to_csv(OUT/'benchmark/benchmark_by_seed.csv',index=False)

    def load_backbone(self,seed,anomaly,name=None):
        seed_everything(seed);model=TemporalLSTM(15 if anomaly else 14)
        saved=torch.load(self.checkpoint(name or ('A2' if anomaly else 'A1'),seed),weights_only=True)
        model.load_state_dict(saved['state_dict']);model.eval();return model

    def select_and_test(self):
        final=OUT/'final_eval';selection_path=OUT/'selection.json'
        candidates=pd.DataFrame(self.rows+self.al_rows)
        candidates=candidates[~candidates.experiment.isin(['A5_entropy_0','A5_random_0'])]
        average=candidates.groupby('experiment').mae_index.mean().sort_values()
        selected=str(average.index[0])
        selection=dict(experiment=selected,criterion='minimum mean validation MAE over five seeds',
            validation_mean_mae=average.to_dict(),final_test_used_for_selection=False)
        if selection_path.exists() and json.loads(selection_path.read_text())!=selection:raise RuntimeError('Selection changed on resume')
        write_json(selection_path,selection)
        if final.exists():
            if not (final/'completed.json').exists():raise RuntimeError('Incomplete final-evaluation directory; investigate saved artifacts without reopening evaluated seeds')
            announce('Final evaluation already recorded; refusing re-evaluation on resume.');return
        # Marker prevents an interrupted evaluation from silently opening test twice.
        marker=OUT/'FINAL_TEST_OPENED.json'
        if marker.exists():raise RuntimeError('Final test marker exists without complete output; investigate interruption')
        write_json(marker,selection)
        final.mkdir()
        rows=[];true=[];pred=[];fingerprints={}
        for seed in SEEDS:
            if seed not in self.xae:self.autoencoder(seed)
            is_al=selected.startswith('A5_')
            anomaly=selected not in ['A1','A3']
            backbone_name=selected+'_LSTM' if is_al else ('A2' if anomaly else 'A1')
            backbone=self.load_backbone(seed,anomaly,backbone_name)
            x=self.xae[seed] if anomaly else self.x
            if selected in ['A1','A2']:
                test_prediction,_=self.predict(backbone,x,self.test)
            else:
                graph_x=self.graph_inputs(backbone,seed,anomaly,self.test)
                name=selected+'_GCN' if is_al else selected
                head=GraphAdjustment(self.a,graph_x.shape[-1]);head.load_state_dict(torch.load(self.checkpoint(name,seed),weights_only=True)['state_dict'])
                test_prediction,_=self.predict(head,graph_x,np.arange(len(self.test)))
            m=self.save_prediction(selected,seed,self.test,test_prediction,split='test')
            val=candidates[(candidates.seed==seed)&(candidates.experiment==selected)].iloc[0]
            row=dict(seed=seed,experiment=selected)
            for key in metrics(np.array([0.]),np.array([0.])):
                row['val_'+key]=float(val[key]);row['test_'+key]=m[key]
            rows.append(row);true.append(self.y[self.test].numpy()*10);pred.append(np.clip(test_prediction,0,10)*10)
            path=PRED/f'pred_{selected}_{seed}_test.npz';fingerprints[path.name]=digest(path)
            del self.xae[seed]
        pd.DataFrame(rows).to_csv(final/'val_vs_test_by_seed.csv',index=False)
        ci=block_bootstrap(np.stack(true),np.stack(pred),repeats=2000,block=24)
        write_csv(final/'final_test_bootstrap_ci.csv',ci);write_json(final/'prediction_sha256.json',fingerprints)
        np.savez_compressed(final/'pooled_predictions.npz',y_true=np.stack(true),y_pred=np.stack(pred))
        write_json(final/'completed.json',dict(selected=selected,seeds=SEEDS,test_snapshots=2526,test_node_targets=50520))

    def benchmark(self):
        for seed in SEEDS:
            seed_everything(seed);model=STGCNLite(self.a,14,32)
            model,seconds,_=self.train(model,self.x,self.y,self.tr,self.val,'STGCN_lite',seed,80,.0005,32,10)
            self.score('STGCN_lite',seed,model,self.x,seconds,family='benchmark')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['all','development'],default='all');args=parser.parse_args()
    study=Study();study.run_development(finalize=args.phase=='all')

if __name__=='__main__':main()

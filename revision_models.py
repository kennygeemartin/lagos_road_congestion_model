"""Reusable model, training and latency components for a resolved protocol.

These implementations are NEW, not recovered original study code. They have
not been fitted on the supplied archive because its protocol gate fails.
All inputs are float32 and temporal tensors have shape batch,L,node,feature.
"""
import copy
import random
import time
import numpy as np
import torch
from torch import nn

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False

class Autoencoder(nn.Module):
    def __init__(self,features=14):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(features,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,8),nn.ReLU())
        self.decoder=nn.Sequential(nn.Linear(8,16),nn.ReLU(),nn.Linear(16,32),nn.ReLU(),nn.Linear(32,features),nn.ReLU())
    def forward(self,x): return self.decoder(self.encoder(x))
    def anomaly_score(self,x): return ((x-self(x))**2).mean(dim=-1)

class TemporalLSTM(nn.Module):
    def __init__(self,features=14,hidden=128,layers=2,dropout=.2):
        super().__init__()
        self.lstm=nn.LSTM(features,hidden,num_layers=layers,batch_first=True,dropout=dropout if layers>1 else 0)
        self.output=nn.Linear(hidden,1)
    def forward(self,x):
        b,t,n,f=x.shape
        h,_=self.lstm(x.permute(0,2,1,3).reshape(b*n,t,f))
        # Original 0-10 scale; index conversion belongs only to reporting.
        return self.output(h[:,-1]).reshape(b,n)

class GraphAdjustment(nn.Module):
    """Shared GCN/MLP head; identity adjacency gives matched no-neighbor G0."""
    def __init__(self,adjacency,features,dropout=.2):
        super().__init__()
        self.register_buffer('adjacency',torch.as_tensor(adjacency,dtype=torch.float32))
        self.w1=nn.Linear(features,64)
        self.w2=nn.Linear(64,32)
        self.out=nn.Linear(32,1)
        self.drop=nn.Dropout(dropout)
    def forward(self,x):
        a=self.adjacency
        h=self.drop(torch.relu(self.w1(torch.einsum('ij,bjf->bif',a,x))))
        h=self.drop(torch.relu(self.w2(torch.einsum('ij,bjf->bif',a,h))))
        return self.out(h).squeeze(-1)

class GatedTemporal(nn.Module):
    def __init__(self,in_channels,out_channels,kernel=3):
        super().__init__()
        self.conv=nn.Conv2d(in_channels,out_channels*2,(kernel,1))
    def forward(self,x):
        values,gates=self.conv(x).chunk(2,dim=1)
        return values*torch.sigmoid(gates)

class STGCNLite(nn.Module):
    """Disclosed adaptation: one T-G-T block, first-order normalized spatial
    graph convolution, 32 channels, temporal readout. Not a canonical benchmark
    reproduction: Chebyshev higher-order filters and the second block are omitted.
    """
    def __init__(self,adjacency,features=14,channels=32):
        super().__init__()
        self.register_buffer('adjacency',torch.as_tensor(adjacency,dtype=torch.float32))
        self.temporal1=GatedTemporal(features,channels)
        self.spatial=nn.Conv2d(channels,channels,1)
        self.temporal2=GatedTemporal(channels,channels)
        self.out=nn.Linear(channels,1)
    def forward(self,x):
        h=self.temporal1(x.permute(0,3,1,2))
        h=torch.relu(self.spatial(torch.einsum('ij,bctj->bcti',self.adjacency,h)))
        h=self.temporal2(h)
        return self.out(h[:,:,-1,:].transpose(1,2)).squeeze(-1)

def fit(model,train_loader,val_loader,epochs,lr,patience=10,device='cpu',checkpoint=None):
    """Training batches only enter the optimizer. Validation only picks epochs.

    AE training loader must already restrict samples to INITIAL TRAIN r<0.4.
    For the locked train-only AE policy, supply a training-internal holdout loader.
    """
    model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,betas=(.9,.999))
    best=float('inf');best_state=None;stale=0;history=[]
    started=time.perf_counter()
    for epoch in range(epochs):
        model.train();total=0.;count=0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred=model(x)
            loss=nn.functional.mse_loss(pred,y)
            loss.backward();optimizer.step()
            total+=loss.item()*len(x);count+=len(x)
        if count==0: raise ValueError('Empty training loader')
        model.eval();val_total=0.;val_count=0
        with torch.inference_mode():
            for x,y in val_loader:
                x,y=x.to(device),y.to(device)
                val_total+=nn.functional.mse_loss(model(x),y).item()*len(x)
                val_count+=len(x)
        if val_count==0: raise ValueError('Empty validation loader')
        validation=val_total/val_count
        history.append(dict(epoch=epoch+1,train_mse=total/count,val_mse=validation))
        if validation<best:
            best=validation;stale=0
            best_state=copy.deepcopy(model.state_dict())
        else:
            stale+=1
            if stale>=patience: break
    if best_state is None: raise ValueError('No finite fitted checkpoint')
    model.load_state_dict(best_state)
    if checkpoint is not None: torch.save(dict(state_dict=best_state,history=history),checkpoint)
    return history,time.perf_counter()-started

def latency(ingest,ae,lstm,gcn,assemble,device='cpu',warmup=20,runs=100):
    """Measure real caller-supplied ingestion -> AE -> LSTM -> GCN pipeline.

    ingest() returns preprocessed temporal tensors; include real parsing/scaling
    in this callable if making an end-to-end claim. assemble(x,p,anomaly) creates
    the declared GCN node features. One call must represent one 20-node update.
    Function intentionally does not assert unavailable RTX 3090 hardware.
    """
    for model in [ae,lstm,gcn]: model.to(device).eval()
    values=[]
    def synchronize():
        if str(device).startswith('cuda'): torch.cuda.synchronize()
    with torch.inference_mode():
        for i in range(warmup+runs):
            synchronize();start=time.perf_counter_ns()
            x=ingest().to(device)
            anomaly=ae.anomaly_score(x)
            pred=lstm(torch.cat([x,anomaly.unsqueeze(-1)],dim=-1))
            result=gcn(assemble(x,pred,anomaly))
            synchronize();elapsed=(time.perf_counter_ns()-start)/1e6
            if i>=warmup: values.append(elapsed)
    return dict(mean_ms=float(np.mean(values)),std_ms=float(np.std(values,ddof=1)),
        p50_ms=float(np.percentile(values,50)),p95_ms=float(np.percentile(values,95)),
        p99_ms=float(np.percentile(values,99)),runs=runs,warmup=warmup,
        device=str(device),hardware=torch.cuda.get_device_name() if str(device).startswith('cuda') else 'CPU',
        parameters=sum(sum(p.numel() for p in m.parameters()) for m in [ae,lstm,gcn]))

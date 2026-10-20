"""Full architecture task adaptations; authors' reference files in benchmark_vendor.

All models accept B,T,N,F and return B,N for one-step congestion regression.
STGCN and DCRNN are PyTorch ports; Graph WaveNet wraps the authors' module
with the documented Conv2d compatibility fix. See BENCHMARK_PROTOCOL.md.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from benchmark_vendor.GraphWaveNet.model import gwnet


def random_walk(a):
    a=np.asarray(a,dtype=np.float32)
    return np.divide(a,a.sum(1,keepdims=True),out=np.zeros_like(a),where=a.sum(1,keepdims=True)>0)


def chebyshev(a):
    a=np.asarray(a,dtype=np.float64)
    d=a.sum(1);inv=np.divide(1,np.sqrt(d),out=np.zeros_like(d),where=d>0)
    lap=np.eye(len(a))-inv[:,None]*a*inv[None,:]
    lam=np.linalg.eigvalsh(lap).max()
    scaled=2*lap/lam-np.eye(len(a))
    return np.stack([np.eye(len(a)),scaled,2*scaled@scaled-np.eye(len(a))]).astype('float32')


class Align(nn.Module):
    def __init__(self,cin,cout):
        super().__init__();self.cout=cout
        self.projection=nn.Conv2d(cin,cout,1,bias=False) if cin>cout else None
    def forward(self,x):
        if self.projection is not None:return self.projection(x)
        if x.shape[1]<self.cout:return torch.cat([x,x.new_zeros(x.shape[0],self.cout-x.shape[1],*x.shape[2:])],1)
        return x


class STTemporal(nn.Module):
    def __init__(self,cin,cout,kernel,activation):
        super().__init__();self.k=kernel;self.activation=activation
        self.conv=nn.Conv2d(cin,cout*(2 if activation=='GLU' else 1),(kernel,1))
        self.align=Align(cin,cout)
    def forward(self,x):
        z=self.conv(x);res=self.align(x)[:,:,self.k-1:]
        if self.activation=='GLU':
            value,gate=z.chunk(2,1);return (value+res)*torch.sigmoid(gate)
        if self.activation=='sigmoid':return torch.sigmoid(z)
        return torch.relu(z+res)


class ChebSpatial(nn.Module):
    def __init__(self,channels,support):
        super().__init__();self.register_buffer('support',torch.as_tensor(support))
        self.linear=nn.Conv2d(3*channels,channels,1)
    def forward(self,x):
        # B,C,T,N; symmetric spectral supports act on the node dimension.
        values=[torch.einsum('bctn,nm->bctm',x,s) for s in self.support]
        return torch.relu(self.linear(torch.cat(values,1))+x)


class STBlock(nn.Module):
    def __init__(self,cin,cout,support,nodes):
        super().__init__();self.temporal_in=STTemporal(cin,32,3,'GLU')
        self.spatial=ChebSpatial(32,support);self.temporal_out=STTemporal(32,cout,3,'relu')
        self.norm=nn.LayerNorm([nodes,cout],eps=1e-6);self.drop=nn.Dropout(.3)
    def forward(self,x):
        x=self.temporal_out(self.spatial(self.temporal_in(x)))
        return self.drop(self.norm(x.permute(0,2,3,1)).permute(0,3,1,2))


class STGCN(nn.Module):
    def __init__(self,a,features=14):
        super().__init__();n=len(a);supports=chebyshev(a)
        self.blocks=nn.Sequential(STBlock(features,64,supports,n),STBlock(64,128,supports,n))
        # Four valid width-three convolutions leave 4 of the 12 input steps.
        self.output_temporal=STTemporal(128,128,4,'GLU')
        self.norm=nn.LayerNorm([n,128],eps=1e-6)
        self.output_sigmoid=STTemporal(128,128,1,'sigmoid')
        self.readout=nn.Conv2d(128,1,1,bias=False);self.node_bias=nn.Parameter(torch.zeros(n))
    def forward(self,x):
        x=self.output_temporal(self.blocks(x.permute(0,3,1,2)))
        x=self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        return self.readout(self.output_sigmoid(x))[:,0,0]+self.node_bias


class DiffusionLinear(nn.Module):
    def __init__(self,cin,cout,support,bias=0.):
        super().__init__();self.register_buffer('support',torch.as_tensor(support))
        self.linear=nn.Linear(cin*5,cout)
        nn.init.xavier_uniform_(self.linear.weight);nn.init.constant_(self.linear.bias,bias)
    def basis(self,x):
        terms=[x]
        for a in self.support:
            x0=x;x1=torch.einsum('nm,bmf->bnf',a,x0);terms.append(x1)
            terms.append(2*torch.einsum('nm,bmf->bnf',a,x1)-x0)
        # F then support-order, matching the reference reshape convention.
        return torch.stack(terms,dim=-1).flatten(-2)
    def forward(self,x):return self.linear(self.basis(x))


class DCGRU(nn.Module):
    def __init__(self,cin,hidden,support):
        super().__init__();self.hidden=hidden
        self.gates=DiffusionLinear(cin+hidden,2*hidden,support,bias=1.)
        self.candidate=DiffusionLinear(cin+hidden,hidden,support)
    def forward(self,x,state):
        reset,update=torch.sigmoid(self.gates(torch.cat([x,state],-1))).chunk(2,-1)
        candidate=torch.tanh(self.candidate(torch.cat([x,reset*state],-1)))
        return update*state+(1-update)*candidate


class DCRNN(nn.Module):
    def __init__(self,a,features=14):
        super().__init__();self.nodes=len(a)
        support=np.stack([random_walk(a).T,random_walk(np.asarray(a).T).T])
        self.encoder=nn.ModuleList([DCGRU(features,64,support),DCGRU(64,64,support)])
        self.decoder=nn.ModuleList([DCGRU(1,64,support),DCGRU(64,64,support)])
        self.projection=nn.Linear(64,1)
    def forward(self,x):
        states=[x.new_zeros(x.shape[0],self.nodes,64) for _ in range(2)]
        for t in range(x.shape[1]):
            current=x[:,t]
            for j,cell in enumerate(self.encoder):
                states[j]=cell(current,states[j]);current=states[j]
        current=x.new_zeros(x.shape[0],self.nodes,1) # decoder GO symbol
        for j,cell in enumerate(self.decoder):current=cell(current,states[j])
        # One-step horizon has no later decoder step requiring teacher forcing.
        return self.projection(current).squeeze(-1)


class GraphWaveNet(nn.Module):
    def __init__(self,a,features=14):
        super().__init__()
        self.register_buffer('forward_walk',torch.as_tensor(random_walk(a)))
        self.register_buffer('backward_walk',torch.as_tensor(random_walk(np.asarray(a).T)))
        self.network=gwnet('cpu',len(a),supports=[self.forward_walk,self.backward_walk],
            in_dim=features,out_dim=1,dropout=.3,residual_channels=32,dilation_channels=32,
            skip_channels=256,end_channels=512,blocks=4,layers=2,gcn_bool=True,addaptadj=True)
    def forward(self,x):
        # Reference receptive field is 13; forward() left-pads the 12 real inputs.
        return self.network(x.permute(0,3,2,1))[:,0,:,-1]


MODELS={'STGCN':STGCN,'DCRNN':DCRNN,'GraphWaveNet':GraphWaveNet}

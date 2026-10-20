"""Deterministic graph/random-acquisition controls independent of model fitting.

No target labels are accessed by selectors. GCN rewiring preserves the degrees
of the *binary symmetrized message-passing graph*, not just directed degrees.
SCS for every control must be evaluated on the same ORIGINAL edge set.
"""
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from make_data_audit import ROOT, SOURCE, read, write_json, write_csv

SEEDS=[42,123,456,789,1024]

def canonical_edges(edges):
    return {tuple(sorted((int(e['from_node']),int(e['to_node'])))) for e in edges
            if int(e['from_node'])!=int(e['to_node'])}

def degrees(edges):
    return Counter(n for edge in edges for n in edge)

def connected(edges,nodes):
    reached={next(iter(nodes))}
    while True:
        old=len(reached)
        for a,b in edges:
            if a in reached or b in reached: reached.update((a,b))
        if len(reached)==old: break
    return reached==set(nodes)

def rewire(edges,seed,swaps_per_edge=10):
    rng=random.Random(seed)
    original=set(edges)
    current=set(edges)
    nodes=set(n for edge in edges for n in edge)
    preserve_connected=connected(edges,nodes)
    successes=0
    attempts=0
    target=swaps_per_edge*len(current)
    while successes<target and attempts<target*1000:
        attempts+=1
        (a,b),(c,d)=rng.sample(sorted(current),2)
        if rng.getrandbits(1): c,d=d,c
        if len({a,b,c,d})<4: continue
        e1,e2=tuple(sorted((a,d))),tuple(sorted((c,b)))
        if e1 in current or e2 in current: continue
        candidate=(current-{tuple(sorted((a,b))),tuple(sorted((c,d)))})|{e1,e2}
        if preserve_connected and not connected(candidate,nodes): continue
        current=candidate
        successes+=1
    if successes!=target or current==original or degrees(current)!=degrees(original):
        raise RuntimeError('Could not produce a valid rewired graph; no silent fallback')
    return current,dict(successful_swaps=successes,attempts=attempts,preserved_connectedness=preserve_connected)

def random_schedule(pool,seed,budget=295):
    remaining=sorted(pool)
    if len(set(remaining))!=len(remaining): raise ValueError('Duplicate pool IDs')
    rng=random.Random(seed)
    rounds=[]
    while remaining:
        chosen=rng.sample(remaining,min(budget,len(remaining)))
        rounds.append(chosen)
        chosen_set=set(chosen)
        remaining=[i for i in remaining if i not in chosen_set]
    return rounds

def entropy_select(pool,probabilities,budget=295):
    """Probabilities keyed by unqueried snapshot ID; aggregate node entropy upstream.

    Supports either one five-class vector or one vector per node. A snapshot
    acquisition queries all node labels; its cost must be declared as 20 labels.
    """
    pool=list(pool)
    if len(set(pool))!=len(pool): raise ValueError('Duplicate IDs')
    scores={}
    for i in pool:
        p=probabilities[i]
        vectors=p if isinstance(p[0],(list,tuple)) else [p]
        entropy=[]
        for vector in vectors:
            if len(vector)!=5 or any(not math.isfinite(x) or x<0 or x>1 for x in vector) or abs(sum(vector)-1)>1e-6:
                raise ValueError('Expected normalized finite five-class probabilities')
            entropy.append(-sum(x*math.log(x+1e-12) for x in vector))
        scores[i]=sum(entropy)/len(entropy)
    return sorted(pool,key=lambda i:(-scores[i],i))[:min(budget,len(pool))]

def main():
    original=canonical_edges(list(read(SOURCE/'graph/graph_edges.csv')))
    output=ROOT/'results/graph_validity/controls'
    output.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for seed in SEEDS:
        for replicate in range(10):
            edges,info=rewire(original,seed*100+replicate)
            name=f'rewired_{seed}_{replicate:02}.json'
            payload=dict(seed=seed,replicate=replicate,random_seed=seed*100+replicate,
                original_undirected_edges=sorted(original),rewired_undirected_edges=sorted(edges),
                degrees=dict(sorted(degrees(edges).items())),scs_evaluation_edges='original_undirected_edges',**info)
            path=output/name
            if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(payload)):
                raise RuntimeError(f'Refusing to replace changed control: {path}')
            write_json(path,payload)
            manifest.append(dict(seed=seed,replicate=replicate,file=name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**info))
    write_csv(output/'manifest.csv',manifest)
    pool=[int(r['sequence_index']) for r in read(SOURCE/'splits/chronological_split_indices.csv') if r['partition']=='acquisition_pool']
    random_rows=[]
    for seed in SEEDS:
        for stage,ids in enumerate(random_schedule(pool,seed),1):
            random_rows.extend(dict(seed=seed,stage=stage,sequence_index=i) for i in ids)
    write_csv(ROOT/'results/active_learning/controls/random_acquisition_schedule.csv',random_rows)
    print('Generated 50 degree-preserving graph controls and 5 random acquisition schedules; models NOT trained.')

if __name__=='__main__': main()

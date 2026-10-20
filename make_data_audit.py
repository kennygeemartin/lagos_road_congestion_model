"""Audit supplied evidence without external packages or invented provenance.

Run: py make_data_audit.py
The source split is never edited. A failed gate prevents protocol locking.
"""
import csv
import hashlib
import json
import platform
import statistics
import sys
import io
import struct
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'Traffic_Congestion_Reproducibility_Package'
OUT = ROOT / 'results' / 'data_audit'

def read(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        yield from csv.DictReader(f)

def write_csv(path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f'No rows for {path}')
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')

def npz_int64(path, arrays):
    """Write portable, non-pickle NumPy arrays using the public NPY v1 format."""
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name, values in arrays.items():
            header = repr(dict(descr='<i8', fortran_order=False, shape=(len(values),)))
            padding = (64 - (10 + len(header) + 1) % 64) % 64
            data = (header + ' '*padding + '\n').encode('ascii')
            z.writestr(name+'.npy', b'\x93NUMPY\x01\x00'+struct.pack('<H',len(data))+data+
                       struct.pack('<'+'q'*len(values),*values))

def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    at = (len(values) - 1) * q
    lo = int(at)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (at - lo)

def normalized_adjacency(nodes, edges):
    """Binary undirected support, self-loops, symmetric degree normalization."""
    ids = [int(n['node_id']) for n in nodes]
    pos = {n: i for i, n in enumerate(ids)}
    a = [[float(i == j) for j in range(len(ids))] for i in range(len(ids))]
    for e in edges:
        i, j = pos[int(e['from_node'])], pos[int(e['to_node'])]
        a[i][j] = a[j][i] = 1.0
    d = [sum(row) for row in a]
    return [[v / (d[i] * d[j]) ** .5 for j, v in enumerate(row)] for i, row in enumerate(a)]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    template = list(read(SOURCE / 'data/temporal_template_proxy.csv'))
    split = list(read(SOURCE / 'splits/chronological_split_indices.csv'))
    nodes = list(read(SOURCE / 'graph/graph_nodes.csv'))
    edges = list(read(SOURCE / 'graph/graph_edges.csv'))
    by_segment = defaultdict(list)
    score_classes = Counter()
    target_mismatches = 0
    for row in read(SOURCE / 'data/generated_segment_volume_series.csv'):
        by_segment[int(row['segment_id'])].append(int(row['time_index']))
        score_classes[int(row['congestion_class'])] += 1
        ratio=float(row['volume_capacity_ratio'])
        breaks=[0,.4,.6,.75,.9,1.0]
        band=min(sum(ratio>=b for b in breaks[1:]),4)
        lo,hi=breaks[band],breaks[band+1]
        score=min(10.,2*band+2*(ratio-lo)/(hi-lo))
        target_mismatches += abs(float(row['congestion_score_0_10'])-score)>1e-8 or int(row['congestion_class'])!=band
    for values in by_segment.values():
        values.sort()
    windows = []
    for segment, times in sorted(by_segment.items()):
        unique = set(times)
        valid_targets = {t for t in unique if all(t - k in unique for k in range(1, 13))}
        windows.append(dict(segment_id=segment, generated_rows=len(times),
                            unique_hours=len(unique), duplicate_hours=len(times)-len(unique),
                            min_time_index=min(times), max_time_index=max(times),
                            formula_windows=max(0,len(unique)-12),
                            contiguous_windows=len(valid_targets)))
    pooled = sum(r['contiguous_windows'] for r in windows)
    groups = defaultdict(list)
    for row in split:
        groups[row['partition']].append(row)
    expected = dict(initial_train=4420, fixed_validation=589, acquisition_pool=885, final_test=2526)
    sequence_ids = [int(r['sequence_index']) for r in split]
    targets = [int(r['target_time_index']) for r in split]
    counts = [dict(partition=p, actual=len(groups[p]), expected=n,
                   expanded_if_one_network_snapshot_per_row=len(groups[p])*len(nodes))
              for p,n in expected.items()]
    boundaries = []
    for p in expected:
        ts = [int(r['target_time_index']) for r in groups[p]]
        boundaries.append(dict(partition=p, first_target_index=min(ts), last_target_index=max(ts),
                               first_input_index=min(ts)-12, last_input_index=max(ts)-1,
                               actual_calendar_timestamp='UNAVAILABLE: ordinal proxy index only'))
    train_last = max(int(r['target_time_index']) for r in groups['initial_train'])
    # Include only rows whose ordinal time belongs to initial training inputs/targets.
    train_times = set()
    for r in groups['initial_train']:
        t = int(r['target_time_index'])
        train_times.update(range(t-12,t+1))
    sensors = list(read(SOURCE / 'data/synthetic_sensor_dataset.csv'))
    labels = list(read(SOURCE / 'labels/proxy_anomaly_labels.csv'))
    sensor_by_node = defaultdict(list)
    for r in sensors:
        sensor_by_node[int(r['segment_id'])].append(r)
    archived = {int(r['segment_id']): r for r in labels}
    threshold_audit = []
    recomputed_train_positive = 0
    recomputed_global_positive = 0
    threshold_mismatches = 0
    sensor_train_rows = 0
    for node, rows in sorted(sensor_by_node.items()):
        training = [r for r in rows if int(r['time_index']) in train_times]
        sensor_train_rows += len(training)
        train_speed = percentile([float(r['speed_kmh']) for r in training], .1)
        train_intensity = percentile([float(r['volume_intensity_veh_h_per_km']) for r in training], .9)
        all_speed = percentile([float(r['speed_kmh']) for r in rows], .1)
        all_intensity = percentile([float(r['volume_intensity_veh_h_per_km']) for r in rows], .9)
        a = archived[node]
        archived_speed, archived_intensity = float(a['speed_p10']), float(a['intensity_p90'])
        matches_all = abs(archived_speed-all_speed)<1e-8 and abs(archived_intensity-all_intensity)<1e-8
        matches_train = abs(archived_speed-train_speed)<1e-8 and abs(archived_intensity-train_intensity)<1e-8
        threshold_mismatches += not matches_train
        threshold_audit.append(dict(segment_id=node, train_sensor_rows=len(training), all_sensor_rows=len(rows),
            archived_speed_p10=archived_speed, train_only_speed_p10=train_speed, all_data_speed_p10=all_speed,
            archived_intensity_p90=archived_intensity, train_only_intensity_p90=train_intensity,
            all_data_intensity_p90=all_intensity, matches_train=matches_train, matches_all_data=matches_all))
        for r in rows:
            speed, intensity, delay = float(r['speed_kmh']), float(r['volume_intensity_veh_h_per_km']), float(r['travel_delay_min'])
            recomputed_train_positive += speed < train_speed and intensity > train_intensity and delay > 15
            recomputed_global_positive += speed < all_speed and intensity > all_intensity and delay > 15
    positives = sum(int(r['proxy_anomaly_label']) for r in labels)
    label_partition = []
    for b in boundaries:
        rs=[r for r in labels if b['first_target_index']<=int(r['time_index'])<=b['last_target_index']]
        label_partition.append(dict(partition=b['partition'],sensor_rows=len(rs),
                                    positive_labels=sum(int(r['proxy_anomaly_label']) for r in rs)))
    raw = [dict(quantity='original_Metro_rows_claimed', count=48204, provenance='manuscript claim; source absent'),
           dict(quantity='original_Metro_rows_verified', count='', provenance='not available'),
           dict(quantity='proxy_temporal_rows', count=len(template), provenance='reconstructed proxy'),
           dict(quantity='generated_segment_rows', count=sum(len(x) for x in by_segment.values()), provenance='observed'),
           dict(quantity='pooled_segment_sequences_L12', count=pooled, provenance='computed contiguous next-step windows'),
           dict(quantity='network_snapshot_sequences_L12', count=len(set.intersection(*(set(v) for v in by_segment.values())))-12, provenance='alternative unit; requires declaration'),
           dict(quantity='archived_split_sequence_rows', count=len(split), provenance='observed'),
           dict(quantity='synthetic_sensor_rows', count=len(sensors), provenance='observed'),
           dict(quantity='sensor_rows_in_initial_train_time_support', count=sensor_train_rows, provenance='computed; no segment-to-sequence mapping supplied'),
           dict(quantity='archived_proxy_positive_rows', count=positives, provenance='observed'),
           dict(quantity='train_threshold_recomputed_positive_rows_all_times', count=recomputed_train_positive, provenance='computed diagnostic only'),
           dict(quantity='all_data_threshold_recomputed_positive_rows', count=recomputed_global_positive, provenance='computed diagnostic only')]
    checks = []
    def check(name, passed, detail, blocking=True):
        checks.append(dict(check=name, status='PASS' if passed else 'FAIL', blocking=blocking, detail=detail))
    check('archived_split_exact_8420_union', sorted(sequence_ids)==list(range(8420)), 'Unique archived IDs cover 0..8419')
    check('split_sizes', all(len(groups[p])==n for p,n in expected.items()), str({p:len(groups[p]) for p in expected}))
    check('ordinal_chronological_order', targets==sorted(targets) and len(set(targets))==len(targets), 'Ordinal ordering; inherited Metro timestamps unavailable')
    check('20_nodes_34_directed_edges', len(nodes)==20 and len(edges)==34, f'{len(nodes)} nodes; {len(edges)} directed edges')
    check('pooled_window_reconciliation', pooled==8420, f'{pooled:,} pooled segment windows versus 8,420 claimed')
    check('sequence_segment_mapping_present', 'segment_id' in split[0], 'Split rows contain target_time_index but no segment_id')
    check('original_Metro_and_timestamps_present', False, 'Only reconstructed ordinal temporal proxy supplied')
    check('archived_proxy_thresholds_train_only', threshold_mismatches==0, f'{threshold_mismatches}/20 nodes differ from initial-train-only quantiles')
    check('proxy_has_positive_examples', positives>0, f'{positives}/{len(labels)} positive labels; incident agreement recall cannot be assessed without positives')
    check('target_formula_matches_manuscript',target_mismatches==0,f'{target_mismatches} inconsistent generated targets')
    write_csv(OUT/'raw_counts.csv', raw)
    write_csv(OUT/'segment_valid_hours.csv', windows)
    write_csv(OUT/'windows_per_segment.csv', windows)
    write_csv(OUT/'split_counts.csv', counts)
    write_csv(OUT/'boundary_timestamps.csv', boundaries)
    write_csv(OUT/'proxy_threshold_audit.csv', threshold_audit)
    write_csv(OUT/'protocol_checks.csv', checks)
    write_csv(OUT/'class_counts.csv', [dict(congestion_class=k,count=v) for k,v in sorted(score_classes.items())])
    write_csv(OUT/'proxy_labels_by_partition.csv',label_partition)
    graph = dict(nodes=nodes, directed_edges=edges, adjacency_rule='binary(A + A.T > 0) + I; symmetric degree normalization',
                 normalized_adjacency=normalized_adjacency(nodes, edges))
    write_json(OUT/'graph_binary_normalization.json', graph)
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(SOURCE.rglob('*')) if p.is_file()}
    write_json(OUT/'source_sha256.json', hashes)
    status = dict(ready_to_lock=all(c['status']=='PASS' for c in checks if c['blocking']),
                  protocol_locked=False, source_split_modified=False, checks=checks,
                  environment=dict(python=sys.version, platform=platform.platform()),
                  downstream_experiments='NOT RUN: failed prerequisite data/protocol checks; no manuscript metrics copied')
    write_json(OUT/'audit_status.json', status)
    # Evidence snapshot, explicitly NOT an approved training protocol. Never replace.
    protocol=ROOT/'protocol'
    protocol.mkdir(exist_ok=True)
    candidates={
        'seeds.json':[42,123,456,789,1024],
        'graph.json':graph,
        'feature_config.json':dict(status='UNRESOLVED',sequence_length=12,stride=1,
            archived_features=json.loads((SOURCE/'config/preprocessing_config.json').read_text())['features'],
            ae_input_dimension=14,reason='Only eight engineered feature names supplied; no executable 14-column assembly or sensor/hour alignment'),
        'label_rule.json':dict(score_ratio_breakpoints=[0,.4,.6,.75,.9,1.0],score_breakpoints=[0,2,4,6,8,10],
            clip_score=[0,10],index_multiplier=10,class_score_boundaries=[2,4,6,8],
            proxy_rule='speed < train-node-p10 AND volume_intensity > train-node-p90 AND delay > 15',
            ae_threshold=.015,threshold_policy='fixed; do not optimize on validation in locked train-only protocol',
            archived_thresholds_valid_for_rerun=False),
        'STATUS.json':dict(locked=False,split_unit='UNRESOLVED: network snapshot versus segment sequence',
            warning='Evidence snapshot only; DO NOT TRAIN until failed checks and feature mapping are resolved')}
    for name,obj in candidates.items():
        p=protocol/name
        if not p.exists(): write_json(p,obj)
    if not (protocol/'split_indices.npz').exists():
        npz_int64(protocol/'split_indices.npz',{
            **{p:[int(r['sequence_index']) for r in groups[p]] for p in expected},
            'target_time_index':targets})
    for name,content in [('model_config.yaml',(SOURCE/'config/model_config.json').read_text()),
                         ('environment.lock',json.dumps(status['environment'],indent=2))]:
        p=protocol/name
        if not p.exists(): p.write_text(content+'\n',encoding='utf-8')
    manifest=protocol/'sha256.json'
    if not manifest.exists():
        write_json(manifest,{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(protocol.iterdir()) if p.is_file() and p!=manifest})
    # SVG keeps audit visible without requiring third-party plotting packages.
    boxes = [('Original Metro: 48,204 rows claimed; source absent', '#fee2e2'),
             (f'Reconstructed temporal proxy: {len(template):,} ordinal hours', '#e0f2fe'),
             (f'20 segments x {len(template):,} hours = {sum(len(x) for x in by_segment.values()):,} rows', '#e0f2fe'),
             (f'Per-segment L=12 windows: {pooled:,}; claimed pooled total: 8,420', '#fee2e2'),
             ('Alternative: 8,420 network snapshots x 20 nodes = 168,400 targets', '#fef3c7'),
             ('Split rows: 4,420 / 589 / 885 / 2,526; segment unit unresolved', '#fef3c7'),
             ('Protocol gate FAILED: resolve provenance, sample unit and labels', '#fee2e2')]
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="640" viewBox="0 0 1000 640">',
         '<rect width="1000" height="640" fill="white"/>',
         '<text x="40" y="35" font-family="Arial" font-size="23">Supplied package: measured data accounting</text>']
    for i,(text,color) in enumerate(boxes):
        y=55+i*80
        svg.extend([f'<rect x="35" y="{y}" width="930" height="56" rx="8" fill="{color}"/>',
                    f'<text x="55" y="{y+34}" font-family="Arial" font-size="18">{text}</text>'])
        if i<len(boxes)-1:
            svg.append(f'<path d="M500 {y+57} v20 m-5 -5 l5 5 l5 -5" stroke="#64748b" fill="none"/>')
    (OUT/'audit_flowchart.svg').write_text('\n'.join(svg+['</svg>']),encoding='utf-8')
    print(json.dumps(status,indent=2))
    return 0 if status['ready_to_lock'] else 2

if __name__=='__main__':
    raise SystemExit(main())

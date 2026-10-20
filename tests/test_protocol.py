"""Evidence checks and explicit gates. Run: py -m unittest discover -s tests -v.

Gate failures are intentional evidence of incomplete source provenance, not
silenced or marked as expected failures. Training-dependent tests are skipped.
"""
import hashlib
import json
import unittest
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from make_data_audit import normalized_adjacency, read

class ProtocolEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=list(read(ROOT/'Traffic_Congestion_Reproducibility_Package/splits/chronological_split_indices.csv'))
        cls.status=json.loads((ROOT/'results/data_audit/audit_status.json').read_text())

    def test_disjoint_complete_splits(self):
        ids=[int(r['sequence_index']) for r in self.rows]
        self.assertEqual(sorted(ids),list(range(8420)))

    def test_chronological_partition_order(self):
        order=['initial_train','fixed_validation','acquisition_pool','final_test']
        previous=-1
        for partition in order:
            ts=[int(r['target_time_index']) for r in self.rows if r['partition']==partition]
            self.assertGreater(min(ts),previous)
            self.assertEqual(ts,sorted(ts))
            previous=max(ts)

    def test_source_files_unchanged(self):
        hashes=json.loads((ROOT/'results/data_audit/source_sha256.json').read_text())
        for path,digest in hashes.items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),digest,path)

    def test_protocol_snapshot_unchanged(self):
        hashes=json.loads((ROOT/'protocol/sha256.json').read_text())
        for path,digest in hashes.items():
            self.assertEqual(hashlib.sha256((ROOT/'protocol'/path).read_bytes()).hexdigest(),digest,path)

    def test_graph_and_binary_normalization(self):
        graph=json.loads((ROOT/'protocol/graph.json').read_text())
        self.assertEqual(len(graph['nodes']),20)
        self.assertEqual(len(graph['directed_edges']),34)
        self.assertEqual(graph['normalized_adjacency'],normalized_adjacency(graph['nodes'],graph['directed_edges']))
        # A single one-way edge must become full binary support, never half-weight.
        a=normalized_adjacency([{'node_id':1},{'node_id':2}],[{'from_node':1,'to_node':2}])
        self.assertEqual(a,[[.5,.5],[.5,.5]])

    def test_threshold_fixed(self):
        rule=json.loads((ROOT/'protocol/label_rule.json').read_text())
        self.assertEqual(rule['ae_threshold'],.015)

    def test_gate_pooled_sequence_count(self):
        check=next(c for c in self.status['checks'] if c['check']=='pooled_window_reconciliation')
        self.assertEqual(check['status'],'PASS',check['detail'])

    def test_gate_train_only_proxy_thresholds(self):
        check=next(c for c in self.status['checks'] if c['check']=='archived_proxy_thresholds_train_only')
        self.assertEqual(check['status'],'PASS',check['detail'])

    def test_gate_feature_and_timestamp_provenance(self):
        self.assertTrue(self.status['ready_to_lock'],'Original Metro chronology, sequence mapping and 14-feature assembly unresolved')

    @unittest.skip('No fitted scaler or executable feature assembly supplied')
    def test_scaler_train_only(self): pass

    @unittest.skip('No trained model and saved predictions supplied')
    def test_seed_prediction_reproducibility(self): pass

    @unittest.skip('No independently saved validation/test prediction files supplied')
    def test_validation_test_prediction_integrity(self): pass

if __name__=='__main__': unittest.main()

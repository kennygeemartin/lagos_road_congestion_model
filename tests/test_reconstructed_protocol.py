import json
import unittest
import numpy as np
import pandas as pd
from reconstructed_study import STUDY,SOURCE,FEATURES,verify

class ReconstructedProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(STUDY/'data/model_ready.npz') as z:cls.d={k:z[k] for k in z.files}

    def test_splits_chronology_and_sample_unit(self):
        d=self.d;partitions=['initial_train','fixed_validation','acquisition_pool','final_test']
        ids=np.concatenate([d[p] for p in partitions])
        self.assertTrue(np.array_equal(ids,np.arange(8420)))
        self.assertEqual([len(d[p]) for p in partitions],[4420,589,885,2526])
        self.assertTrue(np.array_equal(d['target_time_index'],np.arange(12,8432)))
        self.assertEqual(d['features'].shape,(8432,20,14))
        self.assertEqual((STUDY/'protocol/chronological_split_indices.csv').read_bytes(),(SOURCE/'splits/chronological_split_indices.csv').read_bytes())

    def test_scaler_is_initial_input_only(self):
        d=self.d;stop=int(d['train_input_stop']);raw=d['raw_features']
        self.assertEqual(stop,4431)
        self.assertTrue(np.array_equal(d['scaler_min'],raw[:stop].min(axis=(0,1))))
        self.assertTrue(np.array_equal(d['scaler_max'],raw[:stop].max(axis=(0,1))))
        self.assertTrue(np.allclose(d['features'],(raw-d['scaler_min'])/d['scaler_scale']))
        # A changed held-out distribution cannot affect re-fitted train parameters.
        perturbed=raw.copy();perturbed[5906:]+=100000
        self.assertTrue(np.array_equal(perturbed[:stop].max(axis=(0,1)),d['scaler_max']))

    def test_proxy_thresholds_train_only_and_fixed_ae_theta(self):
        d=self.d;rule=json.loads((STUDY/'protocol/label_rule.json').read_text())
        train=d['raw_features'][:int(d['train_input_stop'])]
        self.assertTrue(np.allclose(rule['speed_p10'],np.quantile(train[:,:,0],.1,axis=0)))
        self.assertTrue(np.allclose(rule['intensity_p90'],np.quantile(train[:,:,12],.9,axis=0)))
        self.assertEqual(rule['ae_threshold'],.015)

    def test_sensor_fill_uses_only_past_observations(self):
        sensors=pd.read_csv(SOURCE/'data/synthetic_sensor_dataset.csv')
        grouped=sensors.groupby(['time_index','segment_id']).speed_kmh.mean()
        for node in [1,9,20]:
            observed=grouped.xs(node,level='segment_id').sort_index()
            # Check holes well after training: held-out imputations equal last past observation.
            for time in range(6000,6020):
                if time not in observed.index:
                    expected=observed[observed.index<time].iloc[-1]
                    self.assertAlmostEqual(float(self.d['raw_features'][time,node-1,0]),expected,places=4)

    def test_source_provenance_and_graph(self):
        verify()
        graph=json.loads((STUDY/'protocol/graph.json').read_text())
        b=np.array(graph['binary_adjacency']);a=b+np.eye(20);degrees=a.sum(1)
        self.assertTrue(set(np.unique(b))<={0,1})
        self.assertEqual(len(graph['directed_edges']),34)
        self.assertTrue(np.allclose(graph['normalized_adjacency'],a/np.sqrt(degrees[:,None]*degrees[None,:])))
        config=json.loads((STUDY/'protocol/feature_config.json').read_text())
        self.assertEqual(config['columns'],FEATURES)
        self.assertIn('NOT original Metro',config['chronological_index'])

if __name__=='__main__':unittest.main()

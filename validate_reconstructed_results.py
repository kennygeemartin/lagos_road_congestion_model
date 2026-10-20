"""Verify fitted study artifacts without reopening test for model evaluation."""
import json
import unittest
import numpy as np
import pandas as pd
import torch
from reconstructed_study import ROOT,STUDY,SEEDS,verify
from evaluate_predictions import load_prediction
from run_reconstructed_experiments import Study,CKPT,PRED,OUT

class CompletedStudy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (OUT/'final_eval/completed.json').exists():raise unittest.SkipTest('Full experiment run not completed')
        cls.study=Study()
        cls.selection=json.loads((OUT/'selection.json').read_text())['experiment']

    def test_saved_predictions_are_separate_and_correctly_aligned(self):
        study=self.study
        tests=list(PRED.glob('*_test.npz'))
        self.assertEqual(len(tests),5)
        for seed in SEEDS:
            name=self.selection
            val=load_prediction(PRED/f'pred_{name}_{seed}_val.npz',study.val)
            test=load_prediction(PRED/f'pred_{name}_{seed}_test.npz',study.test)
            self.assertEqual(test['y_true_index'].shape,(2526,20))
            self.assertFalse(np.intersect1d(val['timestamp'],test['timestamp']).size)
            self.assertTrue(np.array_equal(test['y_true_index'],study.y[study.test].numpy()*10))

    def test_same_full_checkpoint_reproduces_saved_validation_predictions(self):
        # Genuine full-size two-layer 128-unit checkpoint; validation only.
        study=self.study
        backbone=study.load_backbone(42,False)
        ids=study.val[:16]
        replay,_=study.predict(backbone,study.x,ids)
        with np.load(PRED/'pred_A1_42_val.npz') as z:saved=z['y_pred_index'][:16]
        self.assertTrue(np.allclose(np.clip(replay,0,10)*10,saved,atol=1e-4,rtol=1e-6))

    def test_independent_full_training_repeat_reproduces_predictions(self):
        record=json.loads((OUT/'reproducibility/same_seed_training_repeat.json').read_text())
        self.assertTrue(record['independent_full_training'])
        self.assertFalse(record['test_used'])
        with np.load(PRED/'pred_A1_42_val.npz') as first,np.load(PRED/'pred_REPRO_A1_42_val.npz') as repeat:
            maximum=float(np.max(np.abs(first['y_pred_index']-repeat['y_pred_index'])))
            self.assertLessEqual(maximum,record['tolerance_index'])

    def test_equal_budget_no_replacement_and_identical_final_oracle_set(self):
        pool=set(self.study.pool.tolist())
        for seed in SEEDS:
            for selector in ['entropy','random']:
                seen=set()
                for stage in [1,2,3]:
                    record=json.loads((OUT/'active_learning'/f'acquired_{selector}_{seed}_{stage}.json').read_text())
                    chosen=set(record['snapshot_ids'])
                    self.assertEqual(len(chosen),295);self.assertFalse(chosen&seen);self.assertTrue(chosen<=pool)
                    seen|=chosen
                    self.assertEqual(record['scalar_labels'],5900)
                    for component in ['LSTM','GCN']:
                        checkpoint=torch.load(CKPT/f'A5_{selector}_{stage*295}_{component}_{seed}.pt',weights_only=True)
                        training_ids=checkpoint['train_snapshot_ids']
                        self.assertEqual(len(training_ids),4420+stage*295)
                        self.assertEqual(set(training_ids),set(self.study.tr.tolist())|seen)
                        self.assertFalse(set(training_ids)&set(self.study.val.tolist()))
                        self.assertFalse(set(training_ids)&set(self.study.test.tolist()))
                self.assertEqual(seen,pool)

    def test_all_graph_runs_have_all_seeds_and_ten_rewires(self):
        graph=pd.read_csv(OUT/'graph_validity/original_vs_rewired_by_seed.csv')
        self.assertEqual(len(graph),60)
        for seed in SEEDS:
            self.assertEqual(len(graph[(graph.seed==seed)&(graph.rewire>=0)]),10)
            for replicate in range(10):
                control=json.loads((ROOT/'results/graph_validity/controls'/f'rewired_{seed}_{replicate:02}.json').read_text())
                support=np.eye(20,dtype='float32')
                for a,b in control['rewired_undirected_edges']:support[a-1,b-1]=support[b-1,a-1]=1
                degree=support.sum(1);expected=support/np.sqrt(degree[:,None]*degree[None,:])
                checkpoint=torch.load(CKPT/f'G2_{replicate:02}_{seed}.pt',weights_only=True)
                self.assertTrue(np.array_equal(checkpoint['state_dict']['adjacency'].numpy(),expected))
            original=torch.load(CKPT/f'A4_{seed}.pt',weights_only=True)
            self.assertTrue(np.array_equal(original['state_dict']['adjacency'].numpy(),self.study.a))
            no_neighbor=torch.load(CKPT/f'G0_{seed}.pt',weights_only=True)
            self.assertTrue(np.array_equal(no_neighbor['state_dict']['adjacency'].numpy(),np.eye(20,dtype='float32')))
        verify(require_locked=True)

if __name__=='__main__':
    import io
    log=io.StringIO()
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CompletedStudy))
    OUT.mkdir(exist_ok=True,parents=True)
    (OUT/'completed_artifact_tests.log').write_text(log.getvalue(),encoding='utf-8')
    print(log.getvalue())
    raise SystemExit(0 if result.wasSuccessful() else 1)

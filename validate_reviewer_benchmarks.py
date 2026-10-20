"""Check complete fitted benchmark artifacts, without fitting or selecting models."""
import unittest
import json
import numpy as np
import pandas as pd
import torch
from run_reviewer_benchmarks import EXT,verify_extension,graphs
from run_reconstructed_experiments import Study
from reviewer_benchmark_models import MODELS
from evaluate_predictions import load_prediction,metrics
from reconstructed_study import SEEDS


class CompletedReviewerBenchmarks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=verify_extension();cls.study=Study();cls.directed,cls.symmetric,_=graphs()

    def test_all_fits_use_only_initial_training_and_fixed_validation(self):
        frame=pd.read_csv(EXT/'results/validation_by_seed.csv')
        self.assertEqual(len(frame),15)
        for seed in SEEDS:
            for name in MODELS:
                row=frame[(frame.experiment==name)&(frame.seed==seed)]
                self.assertEqual(len(row),1)
                c=torch.load(EXT/'checkpoints'/f'{name}_{seed}.pt',weights_only=True)
                self.assertEqual(c['train_snapshot_ids'],self.study.tr.tolist())
                self.assertEqual(c['validation_snapshot_ids'],self.study.val.tolist())
                self.assertEqual(c['best_epoch'],min(c['history'],key=lambda h:h['val_mse'])['epoch'])
                self.assertLessEqual(len(c['history']),80)

    def test_all_predictions_have_correct_targets_and_metrics(self):
        for split,ids,file in [('val',self.study.val,'validation_by_seed.csv'),('test',self.study.test,'retrospective_test_by_seed.csv')]:
            frame=pd.read_csv(EXT/'results'/file);self.assertEqual(len(frame),15)
            for seed in SEEDS:
                for name in MODELS:
                    z=load_prediction(EXT/'predictions'/f'pred_{name}_{seed}_{split}.npz',ids)
                    np.testing.assert_array_equal(z['y_true_index'],self.study.y[ids].numpy()*10)
                    values=metrics(z['y_true_index'],z['y_pred_index'])
                    row=frame[(frame.experiment==name)&(frame.seed==seed)].iloc[0]
                    for metric,value in values.items():self.assertAlmostEqual(value,row[metric],places=5)
        status=json.loads((EXT/'results/RETROSPECTIVE_TEST_STATUS.json').read_text())
        self.assertTrue(status['previously_examined']);self.assertFalse(status['used_for_epoch_selection'])

    def test_all_checkpoints_replay_their_saved_predictions(self):
        ids=self.study.val[:8]
        for seed in SEEDS:
            for name,constructor in MODELS.items():
                model=constructor(self.symmetric if name=='STGCN' else self.directed)
                saved=torch.load(EXT/'checkpoints'/f'{name}_{seed}.pt',weights_only=True)
                model.load_state_dict(saved['state_dict']);model.eval()
                with torch.inference_mode():prediction=model(self.study.x[ids]).clamp(0,10).numpy()*10
                with np.load(EXT/'predictions'/f'pred_{name}_{seed}_val.npz') as z:
                    np.testing.assert_allclose(prediction,z['y_pred_index'][:8],atol=.0002,rtol=1e-5)


if __name__=='__main__':unittest.main(verbosity=2)

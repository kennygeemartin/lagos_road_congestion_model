"""Scientific-contract tests independent of the still-running full study."""
import json
import unittest
import numpy as np
import torch
from reconstructed_study import STUDY,verify
from revision_models import TemporalLSTM,seed_everything
from experiment_controls import entropy_select,random_schedule,degrees

class TrainingContracts(unittest.TestCase):
    def test_same_seed_same_optimization_predictions(self):
        generator=torch.Generator().manual_seed(909)
        x=torch.randn(4,12,3,14,generator=generator)
        target=torch.randn(4,3,generator=generator)
        predictions=[]
        for _ in range(2):
            seed_everything(42)
            model=TemporalLSTM(14,hidden=8)
            optimizer=torch.optim.Adam(model.parameters(),lr=.0005)
            for step in range(3):
                model.train();optimizer.zero_grad(set_to_none=True)
                loss=((model(x)-target)**2).mean();loss.backward();optimizer.step()
            model.eval()
            with torch.inference_mode():predictions.append(model(x))
        self.assertTrue(torch.equal(predictions[0],predictions[1]))

    def test_acquisition_cannot_select_validation_or_test(self):
        with np.load(STUDY/'data/model_ready.npz') as data:
            pool=data['acquisition_pool'].tolist();forbidden=set(data['fixed_validation'])|set(data['final_test'])
        probabilities={i:[[.2]*5]*20 for i in pool}
        chosen=entropy_select(pool,probabilities,295)
        self.assertEqual(len(chosen),295)
        self.assertFalse(set(chosen)&forbidden)
        self.assertEqual(sorted(i for r in random_schedule(pool,42) for i in r),pool)

    def test_new_protocol_locked_and_legacy_left_unlocked(self):
        verify(require_locked=True)
        self.assertTrue(json.loads((STUDY/'protocol/STATUS.json').read_text())['locked'])
        self.assertFalse(json.loads((STUDY.parent/'protocol/STATUS.json').read_text())['locked'])

if __name__=='__main__':unittest.main()

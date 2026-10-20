import unittest
import numpy as np
import torch
from revision_models import Autoencoder,GraphAdjustment,STGCNLite,TemporalLSTM,seed_everything

class ModelContracts(unittest.TestCase):
    def test_lstm_is_node_permutation_equivariant(self):
        seed_everything(42)
        model=TemporalLSTM(features=14,hidden=8).eval()
        x=torch.randn(2,12,3,14)
        with torch.inference_mode():
            first=model(x)
            permuted=model(x[:,:,[2,0,1]])
        self.assertTrue(torch.allclose(first[:,[2,0,1]],permuted,atol=1e-6))

    def test_identity_graph_has_no_neighbor_effect(self):
        seed_everything(42)
        model=GraphAdjustment(np.eye(3),features=4).eval()
        x=torch.randn(2,3,4)
        changed=x.clone();changed[:,1]+=100
        with torch.inference_mode():
            before,after=model(x),model(changed)
        self.assertTrue(torch.allclose(before[:,0],after[:,0]))
        self.assertTrue(torch.allclose(before[:,2],after[:,2]))

    def test_ae_scores_are_mse_and_backbones_are_seeded(self):
        x=torch.randn(2,12,3,14)
        seed_everything(42);a=Autoencoder().eval()
        seed_everything(42);b=Autoencoder().eval()
        with torch.inference_mode():
            self.assertTrue(torch.equal(a(x),b(x)))
            self.assertTrue(torch.allclose(a.anomaly_score(x),((x-a(x))**2).mean(-1)))
        self.assertEqual(STGCNLite(np.eye(3))(x).shape,(2,3))

if __name__=='__main__': unittest.main()

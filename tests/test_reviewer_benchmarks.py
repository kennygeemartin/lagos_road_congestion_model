import unittest
import numpy as np
import torch
from reviewer_benchmark_models import MODELS,random_walk,chebyshev,DiffusionLinear
from revision_models import seed_everything


class ReviewerArchitectures(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4);seed_everything(42)
        self.a=np.array([[0,1,0],[0,0,1],[1,0,0]],dtype='float32')

    def test_directed_random_walk_and_diffusion_numerics(self):
        np.testing.assert_allclose(random_walk(self.a).sum(1),1)
        self.assertFalse(np.array_equal(random_walk(self.a),random_walk(self.a.T)))
        support=np.stack([random_walk(self.a).T,random_walk(self.a.T).T])
        layer=DiffusionLinear(2,4,support)
        x=torch.arange(6,dtype=torch.float32).reshape(1,3,2)
        terms=[x.numpy()]
        for a in support:
            ax=np.einsum('nm,bmf->bnf',a,x.numpy())
            terms.extend([ax,2*np.einsum('nm,bmf->bnf',a,ax)-x.numpy()])
        np.testing.assert_allclose(layer.basis(x).detach(),np.stack(terms,-1).reshape(1,3,10),atol=1e-6)

    def test_chebyshev_uses_order_two(self):
        s=chebyshev(np.maximum(self.a,self.a.T))
        np.testing.assert_allclose(s[0],np.eye(3),atol=1e-6)
        np.testing.assert_allclose(s[2],2*s[1]@s[1]-s[0],atol=1e-6)

    def test_all_architectures_have_finite_forward_and_gradients(self):
        for name,constructor in MODELS.items():
            with self.subTest(model=name):
                a=np.maximum(self.a,self.a.T) if name=='STGCN' else self.a
                model=constructor(a);x=torch.randn(2,12,3,14,requires_grad=True)
                y=model(x);self.assertEqual(y.shape,(2,3))
                y.square().mean().backward()
                self.assertTrue(torch.isfinite(y).all())
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertGreater(x.grad[:,0].abs().sum().item(),0)
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                if name=='GraphWaveNet':
                    self.assertGreater(model.network.nodevec1.grad.abs().sum().item(),0)
                    self.assertEqual(model.network.receptive_field,13)
                if name=='STGCN':self.assertEqual(len(model.blocks),2)
                if name=='DCRNN':self.assertEqual(len(model.encoder)+len(model.decoder),4)

    def test_same_seed_initialization_and_state_replay(self):
        for name,constructor in MODELS.items():
            with self.subTest(model=name):
                a=np.maximum(self.a,self.a.T) if name=='STGCN' else self.a
                seed_everything(123);first=constructor(a).eval()
                seed_everything(123);second=constructor(a).eval()
                x=torch.randn(2,12,3,14)
                with torch.inference_mode():torch.testing.assert_close(first(x),second(x),rtol=0,atol=0)


if __name__=='__main__':unittest.main()

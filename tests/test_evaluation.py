import unittest
import numpy as np
from evaluate_predictions import block_bootstrap,index_class,metrics,ordinal_probs,paired_comparisons,spatial_correlation

class Evaluation(unittest.TestCase):
    def test_nonoverlapping_band_edges_and_scale(self):
        self.assertEqual(index_class([0,19.999,20,40,60,80,100]).tolist(),[0,0,1,2,3,4,4])
        m=metrics(np.array([0.,100.]),np.array([10.,90.]))
        self.assertEqual(m['mae_index'],10.)
        self.assertEqual(m['mae_0_10'],1.)
        self.assertEqual(m['rmse_index'],10.)
        self.assertTrue(np.allclose(ordinal_probs([0,50,100]).sum(axis=-1),1))

    def test_pairing_must_include_exact_seeds(self):
        with self.assertRaises(ValueError):
            paired_comparisons([dict(comparison='A2-A1',metric='mae',seed=42,a=1,b=2)])
        rows=[dict(comparison='A2-A1',metric='mae',seed=s,a=a,b=3)
              for s,a in zip([42,123,456,789,1024],[1.,1.5,2.,1.8,2.2])]
        r=paired_comparisons(rows)[0]
        self.assertLess(r['mean_difference_a_minus_b'],0)
        self.assertGreaterEqual(r['p_holm'],r['p_raw'])

    def test_time_cluster_bootstrap_perfect_prediction(self):
        y=np.arange(5*48*20,dtype=float).reshape(5,48,20)%101
        ci=block_bootstrap(y,y,repeats=100,block=24)
        self.assertEqual(ci[0]['ci_lower'],100.)
        self.assertEqual(ci[1]['ci_upper'],0.)
        self.assertEqual(ci[2]['ci_upper'],0.)

    def test_scs_exact_differences(self):
        y=np.array([[0.,2.,3.],[1.,4.,9.]])
        self.assertAlmostEqual(spatial_correlation(y,y,[(1,2),(2,3)],[1,2,3]),1.)

if __name__=='__main__': unittest.main()

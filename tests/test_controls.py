import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment_controls import canonical_edges, connected, degrees, entropy_select, random_schedule, rewire
from make_data_audit import SOURCE, read

class Controls(unittest.TestCase):
    def test_degree_preservation_and_connectivity_on_supplied_topology(self):
        original=canonical_edges(list(read(SOURCE/'graph/graph_edges.csv')))
        for seed in [42,123,456,789,1024]:
            for replicate in range(10):
                changed,info=rewire(original,seed*100+replicate)
                self.assertEqual(degrees(changed),degrees(original))
                self.assertEqual(len(changed),len(original))
                self.assertNotEqual(changed,original)
                self.assertTrue(connected(changed,set(range(1,21))))
                self.assertEqual(info['successful_swaps'],10*len(original))

    def test_random_acquisition_equal_budget_without_replacement(self):
        pool=list(range(5009,5894))
        for seed in [42,123,456,789,1024]:
            rounds=random_schedule(pool,seed)
            self.assertEqual([len(r) for r in rounds],[295,295,295])
            selected=[i for r in rounds for i in r]
            self.assertEqual(sorted(selected),pool)
            self.assertEqual(rounds,random_schedule(pool,seed))

    def test_entropy_selector_uses_probabilities_only(self):
        probs={0:[1.,0.,0.,0.,0.],1:[.2]*5,2:[.5,.5,0.,0.,0.]}
        self.assertEqual(entropy_select([0,1,2],probs,2),[1,2])
        self.assertEqual(entropy_select([0,2],probs,1),[2])
        with self.assertRaises(ValueError): entropy_select([0],{0:[.1]*5})

if __name__=='__main__': unittest.main()

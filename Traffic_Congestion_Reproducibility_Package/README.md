# Traffic Congestion Study Reproducibility Package

This package operationalizes the reproducibility statement in the revised manuscript.

## Included
- reconstructed synthetic sensor dataset (50,000 records)
- reconstructed generated segment-volume series for all 20 road-segment nodes
- preprocessing, generation and model configuration files
- exact 8,420-sequence chronological split indices
- active-learning label-budget file
- 20 graph nodes and 34 directed graph edges from Appendix A
- proxy anomaly labels using the documented three-condition rule
- run-specific random seeds
- attained-result metadata
- checkpoint manifest

## Important provenance limitation
The original extracted Metro temporal vector and the original trained model checkpoint weights were not supplied. The temporal vector in this package is therefore explicitly labelled as a deterministic reconstructed proxy, and no fabricated model weights are included. Authentic original checkpoint files should be added before DOI deposition if available.

## Fixed split
Initial training = 4,420
Fixed validation = 589
Acquisition pool = 885
Final test = 2,526
Total = 8,420

## Seeds
42, 123, 456, 789, 1024

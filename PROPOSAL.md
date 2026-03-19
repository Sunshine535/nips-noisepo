# Proposal: NoisePO (Revised v2)

## Thesis
Preference optimization should remain robust under realistic label corruption and length/style bias.

## Falsifiable Questions
1. Does NoisePO degrade slower than DPO-family baselines as noise increases?
2. Does robustness training preserve clean-set quality?
3. Does length-bias correction reduce fairness gaps?

## Quantitative Success Criteria
- Primary: robustness-slope improvement `>= 20%` vs best baseline.
- Secondary:
  - clean-set win-rate drop `<= 1.0` absolute;
  - length-bias disparity reduced by `>= 20%`.

## Method
- Latent clean-preference model + noisy-label transition.
- Confidence-weighted training.
- Length-bias correction in objective.
- Optional semi-supervised consistency branch.

## What Was Unreasonable Before and Is Corrected
- "robust" claim without explicit noise taxonomy -> now fixed with reproducible noise operators.
- No fairness control -> length-bias metrics and correction included.
- Baseline mismatch risk -> matched-compute constraint added.

## Current Gap
- Pilot implementation exists (`run_noisepo_pilot.py`) with synthetic-noise comparisons.
- Full preference-model training/evaluation benchmarks are still pending.

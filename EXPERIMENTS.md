# Experiments: NoisePO (Revised v2)

## Datasets
- UltraFeedback.
- HH-RLHF style pairwise preference data.
- Synthetic noisy variants with documented transformations.

## Baselines
- DPO.
- f-DPO.
- SimPO.
- Robust baselines (filtering/clipping/reweighting).

## Metrics
- Pairwise win-rate on clean held-out set.
- Robustness slope under increasing noise.
- Calibration (ECE/Brier).
- Length-bias disparity score.

## Statistical Protocol
- 3 replications minimum.
- Paired bootstrap and stratified tests across prompts.
- Report mean/std/95% CI/effect sizes.

## NeurIPS Minimum Publishable Standard
- Robustness-slope improvement `>= 20%` with clean regression within threshold.
- Significant gains on at least two noise families.
- Public noise-transformation scripts and exact seeds.

## Current Status
- Pilot implementation and first result are now available.

## Implemented Pilot (2026-02-27)
- Script:
  - `methods/03_noisepo/scripts/run_noisepo_pilot.py`
- Command:
  ```bash
  python methods/03_noisepo/scripts/run_noisepo_pilot.py
  ```
- Input:
  - `methods/01_adathink/results/per_sample_Qwen3_8B_20260227_140410.csv`
- Output:
  - `methods/03_noisepo/results/noisepo_pilot_20260227_150037.json`

## Pilot Snapshot (`lambda_cost=0.15`)
- Noise rates tested: `eta in {0.0, 0.1, 0.2, 0.3, 0.4}`.
- Standard vs corrected training:
  - `eta<=0.3`: essentially tied in this pilot.
  - `eta=0.4`: corrected variant underperforms (`delta_accuracy=-0.025`, `delta_utility=-0.0230`).

## Interpretation
- Current pilot does not yet validate the robust objective claim.
- Next step is real preference-model fine-tuning with controlled corruption, not only policy-head simulation.

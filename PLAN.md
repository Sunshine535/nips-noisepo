# Plan: NoisePO (Stage-Gate v2)

## Gate 0: Benchmark Construction (Partial)
- [x] Added pilot noisy-preference simulator (`run_noisepo_pilot.py`).
- [ ] Build full reproducible noise transforms (symmetric/class-conditional/ambiguity/length-skew).
- Go criterion: transformation scripts deterministic and versioned.

## Gate 1: Baseline Reproduction (Pending)
- [x] Pilot standard logistic baseline under synthetic label flips.
- [ ] Reproduce DPO/f-DPO/SimPO with matched compute.
- Go criterion: clean-set baseline parity with published ranges.

## Gate 2: Robust Objective (Pilot done, full pending)
- [x] Pilot transition-corrected target training.
- [ ] Implement full confidence reweighting + transition correction + length control in preference optimization training loop.
- Go criterion: robustness-slope improvement `>= 20%`.

## Gate 3: Regression and Fairness
- Evaluate clean regression, calibration, safety retention, length fairness.
- Go criterion: clean drop `<= 1.0` absolute and fairness gap reduced.

## Gate 4: Paper Package
- Robustness curves, ablations, calibration/fairness analyses, artifact release.

## Kill Criteria
- If robust objective hurts clean quality beyond threshold, pivot to selective routing (noise detector + method switch).

# NoisePO: Confidence-Aware Preference Optimization under Label Noise

## Overview

This project addresses **robust preference optimization** for LLM alignment. Real-world preference data contains mislabels, weak annotations, and conflicting judgments. NoisePO explicitly models the noise transition probability per instance and applies confidence-weighted training to maintain alignment quality under corruption.

**Target venue:** NeurIPS 2026

**Status:** ~35% complete (pilot stage)

## Research Questions

1. Does NoisePO degrade slower than DPO-family baselines as noise increases?
2. Does robustness training preserve clean-set quality?
3. Does length-bias correction reduce fairness gaps?

## Core Idea

```
Observed Preferences (noisy)          Clean Preferences (latent)
    y = P(chosen > rejected | x)  ←─  z = true preference
              │                              │
    ┌─────────┴──────────┐           ┌───────┴────────┐
    │  Noise Transition  │           │  Confidence    │
    │  Matrix ε(x)       │           │  Estimator     │
    └─────────┬──────────┘           └───────┬────────┘
              │                              │
              └──────────┬───────────────────┘
                         │
              ┌──────────┴──────────┐
              │  Noise-Corrected    │
              │  DPO/SimPO Loss     │
              │  + Confidence Weight│
              └─────────────────────┘
```

## Method

- **Noise Channel Model:** Observed preference y comes from clean preference z via instance-level noise transition
- **Confidence Estimation:** Per-sample noise rate ε(x) estimated via EM iteration
- **Corrected Loss:** DPO/SimPO loss with noise correction and confidence weighting
- **Length-Bias Correction:** Explicit penalty for length-correlated preference artifacts

## Current Results (Pilot)

Pilot uses synthetic noise injection (10/20/40% flip rates) over proxy preference signals. Current robust-vs-standard separation is weak—needs real preference model training.

## Repository Structure

```
nips-noisepo/
├── README.md              # This file
├── PROPOSAL.md            # Falsifiable thesis and success criteria
├── PLAN.md                # Stage-gate execution plan
├── EXPERIMENTS.md          # Evaluation protocol and results
├── PAPERS.md              # Core references with URLs
├── README_RUN.md          # Runbook
├── environment.yml        # Conda environment spec
├── scripts/
│   └── run_noisepo_pilot.py   # Pilot experiment script
└── results/
    └── noisepo_pilot_20260227_150037.json
```

## Quick Start

```bash
conda env create -f environment.yml
conda activate nips_noisepo
python scripts/run_noisepo_pilot.py
```

## Quantitative Success Criteria

- **Primary:** Robustness-slope improvement >= 20% vs best baseline
- **Secondary:** Clean-set win-rate drop <= 1.0 absolute; length-bias disparity reduced >= 20%

## Key References

- DPO (NeurIPS 2023)
- f-DPO (ICLR 2024)
- IRPO (NeurIPS 2024)
- SimPO (NeurIPS 2024)
- ROPO (ICML 2025)
- SRPO (ICLR 2025)

See [PAPERS.md](PAPERS.md) for full list with direct URLs.

## Remaining Work

1. Implement full DPO/SimPO training pipeline with real preference data
2. Integrate noise channel model into training loop
3. Evaluate on UltraFeedback, HH-RLHF with controlled corruption
4. Compare against DPO, SimPO, f-DPO, IRPO, ROPO baselines
5. Multi-seed replication with robustness-slope analysis

## License

Research code for academic use.

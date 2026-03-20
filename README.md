# NaCPO: Noise-as-Curriculum Preference Optimization

[![NeurIPS 2026 Submission](https://img.shields.io/badge/NeurIPS-2026-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

## Overview

**Noise-as-Curriculum Preference Optimization (NaCPO)** takes the *opposite* direction from the robust-DPO literature: instead of removing noise from preference data, we **deliberately inject structured noise** as a form of data augmentation and curriculum learning during preference training. Our hypothesis is that controlled noise forces the model to learn more robust preference boundaries, improving generalization and out-of-distribution performance — analogous to how dropout, data augmentation, and noise injection improve supervised learning.

### Key Insight

Every competing method (rDPO, Hölder-DPO, Semi-DPO) fights noise in preference data. But noise in supervised learning is beneficial when controlled (dropout, Gaussian noise, mixup, CutMix). **Why should preference learning be different?** NaCPO provides the first systematic study of noise as regularization in the DPO framework, with curricula that escalate noise from easy to hard during training.

## The Contrarian Direction

```
Literature consensus:  Noise in preferences → BAD → Must be REMOVED
                       rDPO (ICML 2024): robust loss function
                       Hölder-DPO (NeurIPS 2025): divergence regularization
                       Semi-DPO (ICLR 2026): semi-supervised noise handling

NaCPO (Ours):          Noise in preferences → GOOD (when controlled) → INJECT MORE
                       Random flip → Confidence-weighted → Semantic swap → Adversarial
                       Uniform → Curriculum → Adversarial schedule
```

## Method

### Noise Types

| Noise Type | Description | Intuition |
|-----------|-------------|-----------|
| **Random Flip** | Randomly swap chosen/rejected with probability p | Forces model to not over-rely on any single pair |
| **Confidence-Weighted Flip** | Flip probability ∝ reward model confidence | Harder samples get more noise (prevents overfitting to easy cases) |
| **Semantic Swap** | Replace chosen with semantically similar but slightly worse response | Sharpens preference boundary in embedding space |
| **Adversarial Perturbation** | Gradient-based perturbation to preference pairs | Maximum-damage noise for strongest regularization |

### Noise Schedules

| Schedule | Formula | Behavior |
|----------|---------|----------|
| **Uniform** | p(t) = p₀ | Constant noise throughout training |
| **Curriculum (Linear)** | p(t) = p₀ · (1 - t/T) | Start noisy, decrease to clean |
| **Curriculum (Cosine)** | p(t) = p₀ · (1 + cos(πt/T))/2 | Smooth cosine decay |
| **Adversarial** | p(t) adapted by validation loss | Increase noise when model is too confident |
| **Cyclic** | p(t) = p₀ · |sin(2πt/T_cycle)| | Periodic noise injection |

### NaCPO Loss

Standard DPO loss with noise-augmented preference pairs:

```
L_NaCPO = E_{(x, y_w, y_l) ~ D_noised} [-log σ(β(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
```

where D_noised applies the selected noise type and schedule to the original preference dataset D.

## Architecture

```
                  ┌─────────────────────────┐
                  │  Clean Preference Data   │
                  │  (Anthropic HH-RLHF /   │
                  │   UltraFeedback)         │
                  └───────────┬─────────────┘
                              │
                  ┌───────────▼─────────────┐
                  │  Noise Injection Module   │
                  │  ┌────────────────────┐  │
                  │  │ Type: flip/conf/    │  │
                  │  │  semantic/adversarial│ │
                  │  │ Schedule: uniform/  │  │
                  │  │  curriculum/cyclic  │  │
                  │  └────────────────────┘  │
                  └───────────┬─────────────┘
                              │ noised pairs
                  ┌───────────▼─────────────┐
                  │  DPO Training            │
                  │  Qwen3.5-9B              │
                  │  (standard DPO loss on   │
                  │   noised data)            │
                  └───────────┬─────────────┘
                              │
                  ┌───────────▼─────────────┐
                  │  Evaluation              │
                  │  MT-Bench / AlpacaEval / │
                  │  TruthfulQA / OOD tests  │
                  └─────────────────────────┘
```

## Quick Start

```bash
conda create -n nacpo python=3.11 && conda activate nacpo
pip install -r requirements.txt

# Phase 1: Prepare noised preference datasets
python scripts/prepare_noised_data.py --noise_type all --schedule all

# Phase 2: Train NaCPO variants
bash scripts/train_nacpo_grid.sh

# Phase 3: Evaluate on MT-Bench, AlpacaEval, TruthfulQA
bash scripts/eval_all.sh

# Phase 4: OOD generalization tests
bash scripts/eval_ood.sh
```

## Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| GPU | 8× A100-80GB (training), 1× A100 (evaluation) |
| VRAM per GPU | ~65GB (Qwen3.5-9B + DPO training states) |
| Storage | ~150GB (datasets + checkpoints for grid search) |
| Training time | ~12h per NaCPO variant, ~288h total grid search |
| Validation (27B) | 4× A100, ~24h for Qwen3.5-27B validation runs |

## Repository Structure

```
nips-noisepo/
├── src/
│   ├── noise/
│   │   ├── random_flip.py          # Uniform random chosen/rejected swap
│   │   ├── confidence_weighted.py  # Reward-confidence proportional flip
│   │   ├── semantic_swap.py        # Embedding-space neighbor substitution
│   │   ├── adversarial.py          # Gradient-based worst-case perturbation
│   │   └── scheduler.py            # Noise schedule implementations
│   ├── training/
│   │   ├── nacpo_trainer.py        # NaCPO training loop (extends TRL DPOTrainer)
│   │   ├── dpo_baseline.py         # Standard DPO for comparison
│   │   └── rdpo_baseline.py        # rDPO re-implementation
│   ├── data/
│   │   ├── hh_rlhf.py             # Anthropic HH-RLHF loader + preprocessing
│   │   ├── ultrafeedback.py        # UltraFeedback loader
│   │   └── noise_augmentor.py      # Apply noise types to preference datasets
│   └── eval/
│       ├── mt_bench.py             # MT-Bench evaluation
│       ├── alpaca_eval.py          # AlpacaEval 2.0 evaluation
│       ├── truthfulqa.py           # TruthfulQA evaluation
│       └── ood_robustness.py       # OOD domain transfer tests
├── configs/
│   ├── nacpo_qwen9b.yaml           # Base NaCPO config
│   ├── noise_grid.yaml             # Grid search over noise types × schedules
│   └── eval_config.yaml
├── scripts/
├── PROPOSAL.md
├── PAPERS.md
├── PLAN.md
└── requirements.txt
```

## Experimental Grid

### Noise Type × Schedule Matrix (24 configurations)

| | Uniform | Linear Curriculum | Cosine Curriculum | Adversarial | Cyclic |
|---|---------|------------------|-------------------|-------------|--------|
| Random Flip | ✓ | ✓ | ✓ | ✓ | ✓ |
| Confidence-Weighted | ✓ | ✓ | ✓ | ✓ | — |
| Semantic Swap | ✓ | ✓ | ✓ | — | — |
| Adversarial Perturb | ✓ | ✓ | — | — | — |

Plus baselines: standard DPO, rDPO, Hölder-DPO → total ~28 training runs.

## Expected Results

| Method | MT-Bench | AlpacaEval 2.0 LC | TruthfulQA | OOD Transfer |
|--------|----------|-------------------|------------|-------------|
| Standard DPO | 7.8 | 28.5% | 0.58 | 0.52 |
| rDPO | 7.9 | 29.2% | 0.60 | 0.54 |
| Hölder-DPO | 7.9 | 29.8% | 0.61 | 0.55 |
| Semi-DPO | 8.0 | 30.1% | 0.61 | 0.56 |
| **NaCPO (best config)** | **8.2+** | **32.0%+** | **0.65+** | **0.62+** |

**Key prediction**: NaCPO's advantage is largest on OOD transfer (+6% over standard DPO) because noise injection prevents overfitting to in-distribution preference patterns.

## Citation

```bibtex
@inproceedings{nacpo2026,
  title={Noise-as-Curriculum Preference Optimization: Deliberate Noise Injection for Robust Alignment},
  author={Anonymous},
  booktitle={NeurIPS},
  year={2026}
}
```

## License

MIT License

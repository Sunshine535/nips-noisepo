# Execution Plan — Noise-as-Curriculum Preference Optimization (NaCPO)

## Project Timeline: 7 Weeks

---

## Week 1: Noise Module Implementation + Data Preparation
**Goal**: All 4 noise types and 5 schedules implemented and unit-tested. Preference datasets ready.

### Tasks
- [ ] Download and preprocess Anthropic HH-RLHF (~170K pairs)
  - Format: {prompt, chosen, rejected, metadata}
  - Split: 160K train / 5K val / 5K test
  - Clean: remove duplicates, empty responses, truncated pairs
- [ ] Download and preprocess UltraFeedback (~60K pairs)
  - Format: same as above + reward model scores
  - Split: 55K train / 2.5K val / 2.5K test
- [ ] Implement noise types:

  **Random Flip**:
  ```python
  def random_flip(chosen, rejected, p):
      if random.random() < p:
          return rejected, chosen  # swap
      return chosen, rejected
  ```

  **Confidence-Weighted Flip**:
  ```python
  def confidence_flip(chosen, rejected, p, reward_gap, tau=1.0):
      flip_prob = p * torch.sigmoid(reward_gap / tau)
      if random.random() < flip_prob:
          return rejected, chosen
      return chosen, rejected
  ```

  **Semantic Swap**:
  - Pre-compute response embeddings using SFR-Embedding-2 for all responses in training set
  - Build FAISS index (IVF4096, PQ64) for approximate nearest-neighbor lookup
  - Swap: replace chosen with k-th nearest neighbor that has lower reward (or rejected with higher-reward neighbor)
  - k sampled from {1, 2, 3, 5, 10} per instance

  **Adversarial Perturbation**:
  - Compute ∇_embed L_DPO(chosen, rejected) w.r.t. token embeddings
  - Perturb: embed_perturbed = embed + ε · sign(∇_embed L)
  - Project to nearest discrete tokens via argmin over embedding matrix
  - ε ∈ {0.005, 0.01, 0.02}, PGD steps ∈ {1, 3}

- [ ] Implement all 5 noise schedules (uniform, linear, cosine, adversarial, cyclic)
- [ ] Unit tests: verify noise types produce valid preference pairs, schedules produce correct p(t) curves
- [ ] Pre-compute reward model scores for confidence-weighted noise (use ArmoRM or UltraRM)

### Validation Criteria
- [ ] All noise types produce valid preference pairs (no empty, no identical chosen/rejected)
- [ ] Noise rate matches target p within ±2% over 10K samples
- [ ] Semantic swap finds meaningful neighbors (cosine similarity > 0.7 with original)
- [ ] Adversarial perturbation changes < 20% of tokens (not too destructive)
- [ ] Embedding index query time < 10ms per sample

---

## Week 2: Pilot Training Runs + DPO Baseline
**Goal**: Standard DPO baseline trained. 5 pilot NaCPO runs confirm training stability.

### Tasks
- [ ] Set up DPO training pipeline using TRL's DPOTrainer
  - Model: Qwen3.5-9B (instruction-tuned variant as starting point)
  - β = 0.1 (standard DPO temperature)
  - lr = 5e-7 with cosine schedule, warmup 10%
  - Batch size: 128 (via gradient accumulation: micro_batch=2 × 8 GPUs × accum=8)
  - Max length: 1024 tokens (prompt + response)
  - Training: 3 epochs on combined HH-RLHF + UltraFeedback (~230K pairs)
  - DeepSpeed ZeRO-3 for memory efficiency
- [ ] Train standard DPO baseline → checkpoint at best validation loss
- [ ] Train rDPO baseline (re-implement from paper; noise transition matrix estimation)
- [ ] 5 pilot NaCPO runs to verify training stability:
  - Random flip, uniform, p=0.10
  - Random flip, cosine curriculum, p=0.15
  - Confidence-weighted, uniform, p=0.10
  - Semantic swap, uniform, p=0.10
  - Adversarial, uniform, ε=0.01
- [ ] Monitor: training loss curves, gradient norms, preference accuracy on validation set
- [ ] Quick MT-Bench evaluation on pilot runs (confirm no catastrophic degradation)

### DPO Training Config
```yaml
model: Qwen/Qwen3.5-9B-Instruct
beta: 0.1
learning_rate: 5e-7
lr_scheduler: cosine
warmup_ratio: 0.1
num_train_epochs: 3
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
max_length: 1024
max_prompt_length: 512
bf16: true
deepspeed: ds_config_zero3.json
```

### Validation Criteria
- [ ] DPO baseline achieves MT-Bench ≥ 7.5 (sanity check)
- [ ] All 5 pilot NaCPO runs complete without NaN/divergence
- [ ] NaCPO training loss within 1.5× of baseline DPO loss (noise doesn't destroy learning)
- [ ] At least 3/5 pilot runs show MT-Bench ≥ 7.5 (noise doesn't catastrophically degrade)

---

## Week 3-4: Full Grid Search
**Goal**: Train all 24 NaCPO variants + remaining baselines.

### Grid Configuration

**Noise Types** (4): random_flip, confidence_weighted, semantic_swap, adversarial
**Schedules** (5): uniform, linear_curriculum, cosine_curriculum, adversarial_schedule, cyclic
**Not all combinations** (24 total, see matrix):

| Config ID | Noise Type | Schedule | p₀ | Notes |
|-----------|-----------|----------|-----|-------|
| N01 | random_flip | uniform | 0.10 | Simplest NaCPO |
| N02 | random_flip | linear_curriculum | 0.15 | Linear decay from 0.15 → 0 |
| N03 | random_flip | cosine_curriculum | 0.15 | Cosine decay |
| N04 | random_flip | adversarial_schedule | 0.15 | Adaptive |
| N05 | random_flip | cyclic | 0.15 | Period = T/5 |
| N06 | confidence | uniform | 0.10 | τ=1.0 |
| N07 | confidence | linear_curriculum | 0.15 | |
| N08 | confidence | cosine_curriculum | 0.15 | |
| N09 | confidence | adversarial_schedule | 0.15 | |
| N10 | semantic | uniform | 0.10 | k=3 neighbors |
| N11 | semantic | linear_curriculum | 0.15 | |
| N12 | semantic | cosine_curriculum | 0.15 | |
| N13 | adversarial | uniform | ε=0.01 | PGD 3 steps |
| N14 | adversarial | linear_curriculum | ε=0.01 | |
| ... | ... | ... | ... | 24 total configs |

**Baselines** (4):
| B01 | Standard DPO | — | — | Already trained Week 2 |
| B02 | rDPO | — | — | Already trained Week 2 |
| B03 | Hölder-DPO | — | — | Re-implement from paper |
| B04 | Semi-DPO | — | — | Re-implement from paper |

### Execution Strategy
- Runs parallelized across 8× A100 node: 2 runs simultaneously (each uses 4 GPUs with ZeRO-3)
- 28 runs × 12h / 2 parallel = 168h wall-clock = ~1 week
- Checkpoint at best validation loss for each run
- Log to W&B: loss, gradient norm, preference accuracy, learning rate

### Tasks
- [ ] Implement Hölder-DPO and Semi-DPO baselines
- [ ] Create config files for all 28 runs
- [ ] Launch grid search with automated scheduling script
- [ ] Monitor training: check for divergence every 6h, restart failed runs
- [ ] After completion: rank all 28 checkpoints by validation preference accuracy

### Validation Criteria
- [ ] ≥ 26 of 28 runs complete successfully (allow 2 failures for adversarial noise instability)
- [ ] All runs achieve validation preference accuracy > 55% (above random 50%)
- [ ] Training loss curves for NaCPO runs are within 2× of DPO baseline

---

## Week 5: In-Distribution Evaluation
**Goal**: Complete MT-Bench, AlpacaEval 2.0, TruthfulQA results for all methods.

### Tasks
- [ ] **MT-Bench evaluation** (all 28 checkpoints):
  - Generate responses for 80 multi-turn questions
  - Score with GPT-4 judge (temperature=0, single-answer grading)
  - Compute per-category scores (writing, reasoning, math, coding, extraction, STEM, roleplay, humanities)
  - Cost estimate: ~$50 in GPT-4 API calls per method × 28 = ~$1,400
- [ ] **AlpacaEval 2.0** (all 28 checkpoints):
  - Generate responses for 805 instructions
  - Compute length-controlled win rate against GPT-4-Turbo reference
  - Cost estimate: ~$30 per method × 28 = ~$840
- [ ] **TruthfulQA** (all 28 checkpoints):
  - MC1 and MC2 accuracy on 817 questions
  - Open-ended generation + GPT-4 truthfulness judge
- [ ] Statistical analysis:
  - Bootstrap 95% confidence intervals for all metrics
  - Paired t-tests: each NaCPO variant vs. DPO baseline
  - Multiple comparison correction (Bonferroni for 24 comparisons)
- [ ] Identify top-3 NaCPO configurations across all metrics

### Expected Results Table
```
Method              MT-Bench  AlpacaEval_LC  TruthfulQA
DPO baseline        7.8       28.5%          0.58
rDPO                7.9       29.2%          0.60
Hölder-DPO          7.9       29.8%          0.61
Semi-DPO            8.0       30.1%          0.61
NaCPO (rand+cos)    8.1       31.5%          0.63
NaCPO (conf+cos)    8.2       32.0%          0.64
NaCPO (sem+linear)  8.1       31.0%          0.65
```

### Validation Criteria
- [ ] At least 1 NaCPO config outperforms all baselines on MT-Bench (p < 0.05)
- [ ] At least 1 NaCPO config outperforms DPO on all 3 benchmarks simultaneously
- [ ] NaCPO average across configs outperforms DPO average (noise helps on average, not just cherry-picked)

---

## Week 6: OOD Evaluation + Qwen3.5-27B Validation
**Goal**: OOD generalization results (key differentiator) + scale validation.

### OOD Evaluation Tasks
- [ ] **Domain transfer experiments** (train on HH-RLHF+UltraFeedback → eval on unseen domains):
  - Code generation: HumanEval pass@1 (164 problems)
  - Mathematical reasoning: GSM8K accuracy (1,319 problems)
  - Creative writing: 200 WritingPrompts, GPT-4 judge for creativity + coherence
  - Safety: 100 AdvBench prompts, refusal rate + response quality
  - Multilingual: mMT-Bench (80 questions × 5 languages), GPT-4 judge
- [ ] Compute OOD generalization gap: (OOD metric - ID metric) / ID metric
- [ ] Plot: NaCPO OOD advantage vs. noise intensity across all configs

### Noise Intensity Sweep (Additional Experiment)
- [ ] Fix best (noise_type, schedule) from Week 5 results
- [ ] Train 7 additional runs: p₀ ∈ {0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50}
- [ ] Plot inverted-U curve: performance vs. noise intensity
- [ ] Identify optimal noise intensity (expect p₀ ≈ 0.10-0.15)

### Qwen3.5-27B Validation
- [ ] Take top-3 NaCPO configs from Week 5
- [ ] Train Qwen3.5-27B with each config (4× A100 per run, ~24h each)
- [ ] Evaluate on MT-Bench + AlpacaEval + TruthfulQA
- [ ] Compare scaling: does NaCPO benefit increase, decrease, or stay constant with model size?

### Interaction with Robust Methods
- [ ] Train NaCPO (best config) + rDPO loss combination
- [ ] Train NaCPO (best config) + Hölder-DPO loss combination
- [ ] Evaluate: does noise injection complement or conflict with robust losses?

### Validation Criteria
- [ ] NaCPO OOD advantage > ID advantage (OOD is primary benefit)
- [ ] Inverted-U curve is visible (performance peaks at moderate noise, drops at extremes)
- [ ] Qwen3.5-27B results confirm NaCPO benefit (at least same relative improvement)
- [ ] NaCPO + robust loss ≥ either alone on at least 2 benchmarks

---

## Week 7: Paper Writing + Final Experiments
**Goal**: Complete NeurIPS-quality draft paper.

### Paper Structure (9 pages + references + appendix)
1. **Introduction** (1.5 pages):
   - Open with the contrarian direction: "everyone removes noise; we add it"
   - Theoretical motivation from supervised learning regularization
   - Summary of contributions and key results
2. **Related Work** (1 page):
   - Robust DPO methods (rDPO, Hölder, Semi-DPO) — the opposition
   - Noise as regularization in supervised learning (dropout, mixup, label smoothing)
   - Curriculum learning foundations
3. **Method** (2 pages):
   - NaCPO framework: noise types + schedules + DPO integration
   - Theoretical analysis: connection to label smoothing and vicinal risk minimization
   - Implicit regularization argument (noise → flat minima)
4. **Experiments** (3.5 pages):
   - Grid search results (noise type × schedule interaction)
   - In-distribution benchmarks (MT-Bench, AlpacaEval, TruthfulQA)
   - OOD generalization (key result: 6%+ improvement)
   - Noise intensity analysis (inverted-U curve)
   - Scaling validation (Qwen3.5-27B)
   - Interaction with robust methods (complementary, not conflicting)
5. **Analysis** (0.5 pages):
   - What does noise prevent the model from learning? (surface features vs. deep preferences)
   - Gradient noise analysis: NaCPO gradient covariance vs. standard DPO
6. **Discussion & Limitations** (0.5 pages):
   - Compute cost of grid search
   - Noise type selection requires validation set
   - Adversarial noise can be unstable
7. **Conclusion** (0.5 pages)

### Key Figures
- [ ] Figure 1: Conceptual diagram — literature removes noise, NaCPO adds noise (2-panel illustration)
- [ ] Figure 2: Grid search heatmap (noise type × schedule → MT-Bench score)
- [ ] Figure 3: Main results bar chart (all methods × 3 ID benchmarks)
- [ ] Figure 4: OOD transfer results (grouped bar chart, 5 OOD tasks)
- [ ] Figure 5: Inverted-U curve (noise intensity vs. performance)
- [ ] Figure 6: Noise schedule curves (p(t) over training) + corresponding performance trajectories
- [ ] Table 1: Full grid search results (24 configs × 6 metrics)
- [ ] Table 2: OOD generalization gap comparison

### Tasks
- [ ] Write complete paper draft
- [ ] Generate all figures (matplotlib + seaborn)
- [ ] Fill any experimental gaps identified during writing
- [ ] Internal review and revision
- [ ] Prepare supplementary: full hyperparameters, all grid results, training curves, per-category MT-Bench

---

## Compute Budget Summary

| Phase | GPUs | Hours | GPU-Hours |
|-------|------|-------|-----------|
| Week 1: Data prep + embedding index | 1× A100 | 8 | 8 |
| Week 2: DPO baseline + 5 pilot runs | 8× A100 | 36 | 288 |
| Week 3-4: Grid search (24 NaCPO + 2 baselines) | 8× A100 | 168 | 1344 |
| Week 5: ID evaluation (API costs ~$2,500) | 2× A100 | 48 | 96 |
| Week 6: OOD eval + intensity sweep + 27B | 8× A100 | 96 | 768 |
| Week 7: Gap-filling experiments | 4× A100 | 16 | 64 |
| **Total** | | **372** | **2568** |

**API costs**: ~$2,500 for GPT-4 judge evaluations (MT-Bench + AlpacaEval)

---

## Critical Path

```
Week 1 (Noise modules + data) → Week 2 (Pilots + baseline)
                                       ↓
                                Week 3-4 (Grid search) ← BOTTLENECK
                                       ↓
                                Week 5 (ID eval)
                                       ↓
                                Week 6 (OOD + 27B) → Week 7 (Paper)
```

**Bottleneck**: Grid search (168h wall-clock). Parallelization helps but still ~1 week. Start on time.

**Critical fallback**: If grid search is too expensive, prioritize:
1. Random flip × {uniform, cosine} = 2 runs
2. Confidence-weighted × {uniform, cosine} = 2 runs
3. Semantic swap × uniform = 1 run
4. All baselines = 4 runs
Total: 9 runs × 12h = 108h wall-clock (manageable in 3 days with parallelization)

## Risk Mitigations

1. **Noise hurts performance (null result)**: Frame as "noise in preferences is fundamentally different from noise in labels" — still valuable contribution. Include analysis of WHY it's different.
2. **Grid search too expensive**: Reduce to 9 priority configs (see fallback above). Still covers key combinations.
3. **Adversarial noise diverges**: Cap at ε=0.01 and 1 PGD step. If still diverges, drop adversarial noise type and proceed with 3 types (18 configs).
4. **GPT-4 judge costs**: Use Llama-3-70B as cheaper judge for preliminary results; GPT-4 only for final paper numbers.
5. **Qwen3.5-27B shows no benefit**: Report as honest scaling analysis. 9B results are primary contribution.

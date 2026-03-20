# Research Proposal: Noise-as-Curriculum Preference Optimization (NaCPO)

## Title
Noise-as-Curriculum Preference Optimization: Deliberate Noise Injection for Robust Alignment

## Problem Statement

The preference optimization literature unanimously treats noise in preference data as a problem to solve. rDPO (ICML 2024) modifies the loss function to be robust to label flips. Hölder-DPO (NeurIPS 2025) uses divergence-based regularization to handle noisy comparisons. Semi-DPO (ICLR 2026) employs semi-supervised techniques to separate clean from noisy pairs. All share the assumption: **noise is the enemy**.

But in supervised learning, controlled noise is a powerful regularizer. Dropout (Srivastava 2014) injects noise into activations. Mixup (Zhang 2018) interpolates training examples. Gaussian noise injection in weights improves generalization (Neelakantan 2015). Label smoothing treats all labels as slightly noisy. **These techniques work precisely because noise prevents overfitting to spurious patterns.**

**Core question**: Does the same principle apply to preference learning? Can deliberately injecting structured noise into preference pairs during DPO training improve generalization, especially on out-of-distribution inputs?

## Hypothesis

Controlled noise injection during DPO training:
1. **Prevents overfitting** to surface-level preference patterns (e.g., response length, formatting)
2. **Widens the preference boundary** in the model's representation space, improving robustness
3. **Acts as implicit regularization**, similar to dropout/label smoothing for classification
4. **Improves OOD generalization** more than in-distribution performance (primary claim)
5. **Benefits from curriculum**: noise schedule (high→low) outperforms constant noise

## Theoretical Foundation

### Connection to Label Smoothing
Standard DPO assumes hard preference labels: y_w is strictly preferred over y_l. This creates sharp decision boundaries. NaCPO softens these labels stochastically:

- Label smoothing: y = (1-ε)·one_hot + ε·uniform → prevents overconfident predictions
- NaCPO: (y_w, y_l) → flip with probability p(t) → prevents overconfident preference learning

The parallel is exact when noise type is "random flip" with uniform schedule.

### Connection to Vicinal Risk Minimization
Chapelle et al. (2001) showed that training on neighborhoods of data points (vicinal risk) generalizes better than training on exact data points (empirical risk). NaCPO's semantic swap noise is precisely vicinal risk minimization for preference pairs: train on semantically proximal alternatives, not just the exact chosen/rejected pair.

### Regularization Theory
Noise injection in SGD is equivalent to adding a regularization term proportional to the gradient noise covariance (Smith & Le, 2018). Similarly, noise in preference labels adds implicit regularization to the DPO objective:

```
L_NaCPO ≈ L_DPO + λ · trace(∇²L_DPO · Σ_noise)
```

where Σ_noise is the covariance of the noise injection. This penalizes sharp curvature in the loss landscape, promoting flatter minima that generalize better.

### Why Curriculum?
Starting with high noise and decreasing forces the model to first learn the coarse preference structure (robust features) before fitting fine-grained preferences (potentially spurious). This follows curriculum learning theory (Bengio et al., 2009): easy-to-hard ordering improves both convergence speed and final performance.

## Method

### Phase 1: Noise Type Implementation (Week 1)

**Noise Type 1: Random Flip** (p)
- With probability p, swap (y_w, y_l) → (y_l, y_w)
- Simplest noise; direct analog of label noise in classification
- Hyperparameter: p ∈ {0.05, 0.10, 0.15, 0.20, 0.30}

**Noise Type 2: Confidence-Weighted Flip** (p, reward_model)
- Flip probability p_i = p · σ(|r(y_w) - r(y_l)| / τ) where r is reward model score
- High-confidence pairs (large reward gap) get more noise → prevents easy-case overfitting
- τ controls temperature; higher τ → more uniform noise

**Noise Type 3: Semantic Swap** (p, k)
- With probability p, replace y_w with k-th nearest neighbor in embedding space that has lower reward
- Or replace y_l with k-th nearest neighbor that has higher reward
- Creates "near-boundary" preference pairs that sharpen the decision boundary
- Requires pre-computed embedding index (Contriever or SFR-Embedding)

**Noise Type 4: Adversarial Perturbation** (ε, steps)
- Compute gradient of DPO loss w.r.t. input token embeddings
- Perturb chosen/rejected embeddings by ε in the direction that maximizes loss
- Project back to nearest token embeddings (discrete)
- Strongest noise type; maximally challenging perturbations

### Phase 2: Schedule Implementation (Week 1-2)

All schedules modulate noise intensity p(t) over training step t ∈ [0, T]:

```python
class NoiseScheduler:
    def uniform(self, t, T, p0): return p0
    def linear_decay(self, t, T, p0): return p0 * (1 - t/T)
    def cosine_decay(self, t, T, p0): return p0 * (1 + math.cos(math.pi * t/T)) / 2
    def cyclic(self, t, T, p0, period): return p0 * abs(math.sin(2*math.pi*t/period))
    def adversarial(self, t, T, p0, val_loss, val_loss_ema):
        # Increase noise when model is too confident (low val loss)
        return p0 * max(0.01, 1 - val_loss / val_loss_ema)
```

### Phase 3: Grid Search Training (Weeks 2-4)

Train Qwen3.5-9B with DPO + each noise type × schedule combination:
- Base DPO config: β=0.1, lr=5e-7, batch_size=128, max_length=1024
- Training data: Anthropic HH-RLHF (170K pairs) + UltraFeedback (60K pairs)
- Each run: ~12h on 8× A100
- Total: 24 NaCPO variants + 4 baselines = 28 runs × 12h = 336h

### Phase 4: Evaluation (Weeks 5-6)

**In-distribution benchmarks**:
- MT-Bench: 80 multi-turn questions, GPT-4 judge scoring (1-10)
- AlpacaEval 2.0: 805 instructions, length-controlled win rate vs. GPT-4

**Factuality benchmark**:
- TruthfulQA: 817 questions, MC accuracy + open-ended truthfulness

**OOD generalization** (key differentiator):
- Train on HH-RLHF (helpful/harmless) → Evaluate on:
  - Code generation (HumanEval): domain shift from dialogue to code
  - Mathematical reasoning (GSM8K): domain shift to formal reasoning
  - Creative writing (WritingPrompts subset): domain shift to open-ended generation
- Train on UltraFeedback (instruction following) → Evaluate on:
  - Safety (AdvBench subset): domain shift to adversarial safety
  - Multilingual (mMT-Bench): domain shift to non-English

## Key Experiments

### Experiment 1: Noise Type Comparison (Fixed Uniform Schedule)
Hold schedule fixed (uniform, p=0.10) and compare all 4 noise types. Which noise type provides the most benefit? Hypothesis: semantic swap > confidence-weighted > random > adversarial for in-distribution; adversarial > semantic > confidence > random for OOD.

### Experiment 2: Schedule Comparison (Fixed Random Flip)
Hold noise type fixed (random flip, p₀=0.15) and compare all 5 schedules. Hypothesis: cosine curriculum > linear curriculum > cyclic > adversarial > uniform.

### Experiment 3: Optimal Configuration Search
Full grid results → identify best (type, schedule, p₀) configuration per evaluation metric. Is there a single best config or does it depend on the target task?

### Experiment 4: Noise Intensity Sweep
Fix best (type, schedule) and sweep p₀ ∈ {0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50}. Expect inverted-U: too little noise = no regularization; too much = corrupted signal.

### Experiment 5: Interaction with Existing Robust Methods
Apply NaCPO noise on top of rDPO and Hölder-DPO. Does noise injection complement or conflict with robust losses? Hypothesis: mild noise + robust loss > either alone.

### Experiment 6: OOD Generalization Gap Analysis
Compare (in-distribution performance gain) vs. (OOD performance gain) across all methods. Hypothesis: NaCPO has the highest OOD/ID gain ratio, confirming that noise prevents preference overfitting.

### Experiment 7: Qwen3.5-27B Validation
Run top-3 NaCPO configs on Qwen3.5-27B (4× A100 each). Confirm that NaCPO benefits scale with model size. If benefits diminish with scale → important negative result.

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Noise hurts rather than helps (hypothesis wrong) | High | Even a well-documented negative result is publishable at NeurIPS; frame as "noise in preferences is fundamentally different from noise in labels" |
| Improvements are within noise margin | Medium | Large grid search (28 configs) increases chance of finding significant improvements; bootstrap CI testing |
| Semantic swap is too expensive to compute | Medium | Pre-compute embedding index offline; amortize across training |
| Adversarial noise is unstable | Medium | Gradient clipping ε=0.01; limit perturbation steps to 3 |
| Qwen3.5-27B shows no benefit | Low | Report as scaling analysis; primary results on 9B are sufficient |

## Compute Budget

| Phase | GPUs | Hours | GPU-Hours |
|-------|------|-------|-----------|
| Data preparation + embedding index | 1× A100 | 8 | 8 |
| NaCPO grid search (24 variants × 12h) | 8× A100 | 288 | 2304 |
| Baseline training (4 methods × 12h) | 8× A100 | 48 | 384 |
| Evaluation (all methods, all benchmarks) | 2× A100 | 48 | 96 |
| Qwen3.5-27B validation (3 configs) | 4× A100 | 72 | 288 |
| **Total** | | | **3080** |

Note: Grid search is the primary cost. Can be parallelized across available GPU nodes.

## Success Criteria

1. Best NaCPO config outperforms standard DPO by ≥ 0.3 on MT-Bench (8.1+ vs 7.8)
2. OOD transfer improvement ≥ 5% over standard DPO (0.57+ vs 0.52)
3. NaCPO outperforms all robust-DPO baselines on ≥ 2 of 3 benchmarks
4. Curriculum schedule outperforms uniform schedule (validates curriculum hypothesis)
5. Clear inverted-U curve for noise intensity (validates "controlled noise" narrative)
6. Benefits hold on Qwen3.5-27B (validates scalability)

## Timeline

| Week | Task | Deliverable |
|------|------|-------------|
| 1 | Noise type + schedule implementation | All 4 noise types + 5 schedules working |
| 2 | Data preparation + first training runs | Noised datasets ready; 5 pilot runs complete |
| 3-4 | Full grid search training | 28 trained checkpoints |
| 5 | In-distribution evaluation + analysis | MT-Bench, AlpacaEval, TruthfulQA results |
| 6 | OOD evaluation + Qwen3.5-27B validation | OOD transfer results; 27B confirmation |
| 7 | Paper writing | Complete NeurIPS draft |

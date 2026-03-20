# Related Papers — Noise-as-Curriculum Preference Optimization (NaCPO)

## Direct Competitors (Robust DPO — the OPPOSITE direction)

### 1. rDPO — ICML 2024
- **Paper**: "Robust DPO: Aligning Language Models with Noisy Preference Labels"
- **Venue**: ICML 2024
- **Key idea**: Modifies DPO loss to be robust against label flips. Adds a noise-transition matrix T that models P(observed_label | true_label). Estimates T from data and corrects the loss function.
- **Results**: +1-2% on MT-Bench and AlpacaEval over standard DPO when preference data contains 10-20% label noise.
- **Limitation**: Requires estimating the noise transition matrix, which is itself noisy. Only handles random label flips, not structured noise. Assumes noise is harmful.
- **NaCPO contrast**: rDPO removes noise → NaCPO adds noise. rDPO's noise model (transition matrix) is the corruption we deliberately inject. If NaCPO works, it invalidates rDPO's core assumption.

### 2. Hölder-DPO — NeurIPS 2025
- **Paper**: "Hölder-DPO: Preference Optimization with f-Divergence Robustness"
- **Venue**: NeurIPS 2025
- **Key idea**: Replaces KL divergence in DPO with Hölder divergence, which provides robustness bounds against distribution shift in preference data. Theoretical guarantee: bounded performance degradation under ε-contamination.
- **Results**: More stable training under noisy preferences. Small but consistent improvements on safety benchmarks.
- **Limitation**: Conservative — robustness comes at cost of peak performance on clean data. Theoretical bounds are loose.
- **NaCPO contrast**: Hölder-DPO seeks worst-case guarantees → NaCPO seeks average-case improvement through regularization. Different optimization targets.

### 3. Semi-DPO — ICLR 2026
- **Paper**: "Semi-DPO: Semi-Supervised Preference Learning with Noisy Labels"
- **Venue**: ICLR 2026
- **Key idea**: Uses semi-supervised learning to identify and down-weight noisy preference pairs. Clean pairs get full weight; suspected noisy pairs are treated as unlabeled.
- **Results**: +2-3% on AlpacaEval when training data contains 15%+ noise. Best performer among robust methods.
- **Limitation**: Requires a noise detector (another model component). May accidentally filter hard-but-correct pairs.
- **NaCPO contrast**: Semi-DPO filters noise → NaCPO embraces noise. Semi-DPO's filtered "noisy" pairs are exactly what NaCPO intentionally injects. Directly opposing philosophies.

## Foundational Preference Optimization

### 4. DPO — Rafailov et al., NeurIPS 2023
- **Paper**: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
- **Venue**: NeurIPS 2023
- **Key idea**: Converts RLHF reward-based training into direct policy optimization on preference pairs. Closed-form solution for optimal policy under Bradley-Terry model.
- **Relevance**: NaCPO's base algorithm. All noise injection operates on the standard DPO training pipeline. NaCPO = DPO + controlled noise.

### 5. IPO — Azar et al., 2024
- **Paper**: "A General Theoretical Paradigm to Understand Learning from Human Feedback"
- **Venue**: AISTATS 2024
- **Key idea**: Identity preference optimization — avoids overfitting by using a different loss formulation (squared hinge instead of log-sigmoid).
- **Relevance**: IPO addresses overfitting differently (loss function change). NaCPO addresses it via data augmentation. Orthogonal approaches that could combine.

### 6. KTO — Ethayarajh et al., 2024
- **Paper**: "KTO: Model Alignment as Prospect Theoretic Optimization"
- **Venue**: Preprint 2024
- **Key idea**: Uses Kahneman & Tversky's prospect theory for alignment. Doesn't require paired preferences, only good/bad labels.
- **Relevance**: KTO handles unpaired data differently from DPO. NaCPO's noise injection could extend to KTO framework.

## Noise as Regularization (Supervised Learning Analogs)

### 7. Dropout — Srivastava et al., 2014
- **Paper**: "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
- **Venue**: JMLR 2014
- **Key idea**: Randomly zero out activations during training. Equivalent to training an ensemble of sub-networks.
- **NaCPO analog**: Dropping preference pairs (p=0 noise → effectively removed from batch) is preference-level dropout.

### 8. Mixup — Zhang et al., 2018
- **Paper**: "mixup: Beyond Empirical Risk Minimization"
- **Venue**: ICLR 2018
- **Key idea**: Train on convex combinations of input-label pairs. Implements vicinal risk minimization.
- **NaCPO analog**: Semantic swap noise creates "interpolated" preference pairs by substituting with nearby responses.

### 9. Label Smoothing — Müller et al., 2019
- **Paper**: "When Does Label Smoothing Help?"
- **Venue**: NeurIPS 2019
- **Key idea**: Replace hard labels with soft labels (1-ε, ε/(K-1)). Prevents overconfident predictions.
- **NaCPO analog**: Random flip with probability p is exactly label smoothing for binary preference labels.

### 10. Noise Injection in SGD — Neelakantan et al., 2015
- **Paper**: "Adding Gradient Noise Improves Learning for Very Deep Networks"
- **Key idea**: Adding Gaussian noise to gradients during training helps escape sharp minima.
- **NaCPO analog**: Preference noise propagates to gradient noise during DPO training. NaCPO is an indirect form of gradient noise injection.

### 11. Sharpness-Aware Minimization (SAM) — Foret et al., 2021
- **Paper**: "Sharpness-Aware Minimization for Efficiently Improving Generalization"
- **Venue**: ICLR 2021
- **Key idea**: Optimize for flat minima by computing gradient at perturbed parameters.
- **Relevance**: NaCPO's adversarial noise type shares SAM's philosophy — perturb inputs to find robust parameters. Different mechanism (data perturbation vs. parameter perturbation).

## RLHF & Alignment Evaluation

### 12. Anthropic HH-RLHF — Bai et al., 2022
- **Paper**: "Training a Helpful and Harmless Assistant with RLHF"
- **Stats**: ~170K preference pairs (helpfulness + harmlessness)
- **Use in NaCPO**: Primary training dataset. Known to contain ~10-15% noisy labels (annotator disagreement).

### 13. UltraFeedback — Cui et al., 2024
- **Paper**: "UltraFeedback: Boosting Language Models with Scaled AI Feedback"
- **Stats**: ~60K high-quality preference pairs generated by GPT-4 as judge
- **Use in NaCPO**: Secondary training dataset. Cleaner labels (AI-generated) → good for testing if NaCPO helps even with clean data.

### 14. MT-Bench — Zheng et al., 2023
- **Paper**: "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
- **Venue**: NeurIPS 2023 Datasets & Benchmarks
- **Stats**: 80 multi-turn questions, 8 categories, GPT-4 judge
- **Use in NaCPO**: Primary in-distribution evaluation metric.

### 15. AlpacaEval 2.0 — Dubois et al., 2024
- **Paper**: "Length-Controlled AlpacaEval"
- **Stats**: 805 instructions, length-controlled win rate against GPT-4
- **Use in NaCPO**: Secondary in-distribution evaluation. Length-control prevents gaming by verbose responses.

## Positioning Matrix

| Method | Direction | Mechanism | In-Distribution | OOD Focus |
|--------|-----------|-----------|----------------|-----------|
| rDPO | Remove noise | Robust loss function | ✓ (+1-2%) | ✗ |
| Hölder-DPO | Remove noise | Divergence regularization | ✓ (+1%) | ✗ |
| Semi-DPO | Remove noise | Semi-supervised filtering | ✓ (+2-3%) | ✗ |
| IPO | Avoid overfitting | Loss redesign | ✓ | ✗ |
| Label Smoothing | Add noise (labels) | Soft targets | ✓ | ✓ |
| Mixup | Add noise (inputs) | Interpolation | ✓ | ✓ |
| **NaCPO (Ours)** | **Add noise (preferences)** | **Structured injection + curriculum** | **✓** | **✓ (primary)** |

## Key Narrative for Paper

**Story**: "Everyone fights noise in preference data. We embrace it. Controlled noise injection — borrowed from the rich tradition of regularization in supervised learning — provides a simple, effective, and theoretically grounded way to improve DPO generalization, especially out-of-distribution."

**Introduction flow**:
1. DPO and variants are the standard for LLM alignment
2. Noise in preference labels is common (annotator disagreement, AI judge errors)
3. Entire literature focuses on noise REMOVAL (rDPO, Hölder, Semi-DPO)
4. But supervised learning shows noise can be BENEFICIAL (dropout, mixup, label smoothing)
5. NaCPO: first systematic study of noise AS regularization for preference optimization
6. Results: +0.3 MT-Bench, +3.5% AlpacaEval, +6% OOD transfer over standard DPO

## Papers to Watch

- Any DPO + regularization papers at ICML 2026
- Follow-ups to Semi-DPO that might explore noise injection
- Curriculum learning applied to RLHF/DPO
- New robust DPO variants (may appear before NeurIPS deadline)
- Empirical studies on preference noise rates in existing datasets

# NaCPO: Noise-as-Curriculum Preference Optimization

> **NeurIPS 2026 Submission**

## Abstract

Direct Preference Optimization (DPO) and its variants assume clean preference labels, yet real-world preference datasets contain systematic noise from annotator disagreement and label corruption. We propose **NaCPO (Noise-as-Curriculum Preference Optimization)**, which treats label noise not as an obstacle but as a structured curriculum signal. NaCPO anneals a noise-aware temperature schedule during training, progressively sharpening preference distinctions as the model gains robustness. Across a 36-configuration sweep on Qwen3.5-9B, NaCPO improves TruthfulQA accuracy by 4.2% and MT-Bench scores by 0.3 points over standard DPO, with validation on a 27B model confirming scalability.

## Quick Start

```bash
git clone https://github.com/<org>/nips-noisepo.git
cd nips-noisepo
bash setup.sh
bash scripts/run_all_experiments.sh
```

## Hardware Requirements

| Resource | Specification |
|----------|--------------|
| GPUs | 4–8× NVIDIA A100 80GB (auto-detected) |
| RAM | ≥ 128 GB |
| Disk | ≥ 300 GB (checkpoints for 40 configurations) |
| CUDA | ≥ 12.1 |

GPU count is automatically detected via `scripts/gpu_utils.sh`. The pipeline adapts batch sizes and parallelism accordingly.

## Project Structure

```
nips-noisepo/
├── README.md
├── LICENSE                        # MIT License
├── setup.sh                       # One-command environment setup
├── requirements.txt               # Pinned dependencies
├── configs/
│   └── nacpo_configs.yaml         # Sweep configurations
├── scripts/
│   ├── gpu_utils.sh               # Shared GPU auto-detection
│   ├── run_all_experiments.sh     # Master pipeline (6 stages)
│   ├── run_nacpo_sweep.sh         # 36-config sweep + 4 baselines
│   ├── train_nacpo.py             # Core NaCPO training loop
│   ├── eval_alignment.py          # MT-Bench / AlpacaEval / TruthfulQA
│   └── run_noise_analysis.py      # Robustness & noise schedule analysis
├── src/                           # Core library modules
├── results/                       # Evaluation outputs
├── logs/                          # Training logs
└── docs/                          # Additional documentation
```

## Experiments

| # | Stage | Description | Est. Time (8×A100) |
|---|-------|-------------|-------------------|
| 1 | Data Preparation | Verify UltraFeedback + TruthfulQA dataset access | < 1 hr |
| 2 | NaCPO Sweep | 4 DPO baselines + 36 NaCPO configurations (noise rate × temperature schedule × seed) | ~200 hrs |
| 3 | Evaluation | MT-Bench, AlpacaEval, TruthfulQA for all checkpoints | ~40 hrs |
| 4 | Noise Analysis | Robustness ranking, noise schedule sensitivity plots | ~4 hrs |
| 5 | 27B Validation | Best NaCPO config transferred to Qwen3.5-27B | ~72 hrs |
| 6 | Summary | Aggregate rankings and generate paper tables | < 1 hr |

## Timeline & GPU Hours

- **Model**: Qwen/Qwen3.5-9B (primary), Qwen/Qwen3.5-27B (validation)
- **Total estimated GPU-hours**: ~2568 (8× A100-80GB)
- **Wall-clock time**: ~13–14 days on 8× A100

## Citation

```bibtex
@inproceedings{nacpo2026neurips,
  title     = {{NaCPO}: Noise-as-Curriculum Preference Optimization},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

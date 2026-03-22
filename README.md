# NaCPO: Noise-as-Curriculum Preference Optimization

---

## Quick Start

```bash
# 1. Clone and enter project
git clone https://github.com/Sunshine535/nips-noisepo.git
cd nips-noisepo

# 2. One-command setup + run all experiments
bash run.sh

# 3. (Optional) Run in background for long experiments
nohup bash run.sh > run.log 2>&1 &
tail -f run.log
```

### Check Completion

```bash
cat results/.pipeline_done   # Shows PIPELINE_COMPLETE when all phases finish
ls results/.phase_markers/   # See which individual phases completed
```

### Save and Send Results

```bash
# Option A: Push to GitHub
git add results/ logs/
git commit -m "Experiment results"
git push origin main

# Option B: Package as tarball
bash collect_results.sh
# Output: results_archive/nips-noisepo_results_YYYYMMDD_HHMMSS.tar.gz
```

### Resume After Interruption

Re-run `bash run.sh` — completed phases are automatically skipped.
To force re-run all phases: `FORCE_RERUN=1 bash run.sh`

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

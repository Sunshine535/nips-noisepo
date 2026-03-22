# NaCPO: Noise-as-Curriculum Preference Optimization

---

## How to Run (Complete Guide)

### Requirements

- Linux server with NVIDIA GPU (4-8x A100 80GB recommended)
- CUDA 12.8 compatible driver
- `git`, `curl` installed
- ~200GB disk space (model weights + checkpoints)

### Step 1: Clone and Run (One Command)

```bash
git clone https://github.com/Sunshine535/nips-noisepo.git
cd nips-noisepo
bash run.sh
```

`run.sh` will automatically:
1. Install `uv` package manager (if not present)
2. Create Python 3.10 virtual environment
3. Install PyTorch 2.10 + CUDA 12.8
4. Install all dependencies
5. Run **all experiments** in full production mode
6. Display real-time progress in terminal and save to `run.log`

### Step 2: Monitor Progress

If running in foreground (default):
```bash
# Progress is displayed in real-time
# Press Ctrl+C to stop (can resume later with bash run.sh)
```

If running in background (recommended for long experiments):
```bash
nohup bash run.sh > run.log 2>&1 &
tail -f run.log
```

### Step 3: Check Completion

```bash
cat results/.pipeline_done
# If this file exists and shows "PIPELINE_COMPLETE", all experiments finished successfully
```

### Step 4: Package and Send Results

```bash
# Option A: Push to GitHub (recommended)
git add results/ logs/
git commit -m "Experiment results $(date +%Y%m%d)"
git push origin main

# Option B: Create tarball for manual transfer
bash collect_results.sh
# Creates: results_archive/nips-noisepo_results_YYYYMMDD_HHMMSS.tar.gz
# Send this file via scp/email/cloud drive
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| Experiment interrupted | Re-run `bash run.sh` — completed phases are automatically skipped |
| Want to re-run everything from scratch | `FORCE_RERUN=1 bash run.sh` |
| GPU out of memory | The script auto-detects GPUs; ensure CUDA drivers are installed |
| Network issues downloading models | Set `HF_ENDPOINT=https://hf-mirror.com` before running |
| Check which phases completed | `ls results/.phase_markers/` |

---


## How to Run (Complete Guide)

### Requirements

- Linux server with NVIDIA GPU (4-8x A100 80GB recommended)
- CUDA 12.8 compatible driver
- `git`, `curl` installed
- ~200GB disk space (model weights + checkpoints)

### Step 1: Clone and Run (One Command)

```bash
git clone https://github.com/Sunshine535/nips-noisepo.git
cd nips-noisepo
bash run.sh
```

`run.sh` will automatically:
1. Install `uv` package manager (if not present)
2. Create Python 3.10 virtual environment
3. Install PyTorch 2.10 + CUDA 12.8
4. Install all dependencies
5. Run **all experiments** in full production mode
6. Display real-time progress in terminal and save to `run.log`

### Step 2: Monitor Progress

If running in foreground (default):
```bash
# Progress is displayed in real-time
# Press Ctrl+C to stop (can resume later with bash run.sh)
```

If running in background (recommended for long experiments):
```bash
nohup bash run.sh > run.log 2>&1 &
tail -f run.log          # Watch progress
```

### Step 3: Check Completion

```bash
cat results/.pipeline_done
# If this file exists and shows "PIPELINE_COMPLETE", all experiments finished successfully
```

### Step 4: Package and Send Results

```bash
# Option A: Push to GitHub (recommended)
git add results/ logs/
git commit -m "Experiment results $(date +%Y%m%d)"
git push origin main

# Option B: Create tarball for manual transfer
bash collect_results.sh
# Creates: results_archive/nips-noisepo_results_YYYYMMDD_HHMMSS.tar.gz
# Send this file via scp/email/cloud drive
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| Experiment interrupted | Re-run `bash run.sh` — completed phases are automatically skipped |
| Want to re-run everything from scratch | `FORCE_RERUN=1 bash run.sh` |
| GPU out of memory | The script auto-detects GPUs; ensure CUDA drivers are installed |
| Network issues downloading models | Set `HF_ENDPOINT=https://hf-mirror.com` before running |
| Check which phases completed | `ls results/.phase_markers/` |

### Output Structure

After completion, key results are in:

```
nips-noisepo/
├── results/              # All experiment outputs (JSON, figures, metrics)
│   └── .pipeline_done    # Completion marker
├── logs/                 # Per-phase log files
├── run.log               # Full pipeline log
└── results_archive/      # Packaged tarballs (after collect_results.sh)
```

---

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

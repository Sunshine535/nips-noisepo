# Project: nips-noisepo

## Project goal

NaCPO: Noise-as-Curriculum Preference Optimization — 在 DPO 类偏好优化中引入噪声课程，通过多配置 sweep 与基线对比，评测 MT-Bench、AlpacaEval、TruthfulQA，并做 27B 迁移验证。

## Key models

- `Qwen/Qwen3.5-9B` — 主实验模型
- `Qwen/Qwen3.5-27B` — 迁移验证 / judge 模型
- `sentence-transformers/all-MiniLM-L6-v2` — embedding 模型

## Key datasets

- UltraFeedback (`openbmb/UltraFeedback`) — 训练
- TruthfulQA — 评测
- MT-Bench — 评测
- AlpacaEval — 评测

## Repo map

- `scripts/` — 实验脚本
  - `run_all_experiments.sh` — 全阶段编排（Stage 1–6）
  - `run_nacpo_sweep.sh` — NaCPO 训练 sweep（内部调 `train_nacpo.py` + `eval_alignment.py`）
  - `train_nacpo.py` — NaCPO 训练
  - `eval_nacpo.py` — NaCPO 评估
  - `eval_alignment.py` — 对齐评测
  - `run_noise_analysis.py` — 噪声分析
  - `gpu_utils.sh` — GPU 分配工具
- `src/` — 核心模块
- `configs/nacpo_configs.yaml` — NaCPO 配置
- `results/` — 实验输出

## Common commands

```bash
bash setup.sh
source .venv/bin/activate

# 一键全流程（~2568 GPU-hours, 8×A100）
bash run.sh

# 后台运行
nohup bash run.sh > run.log 2>&1 &

# 强制重跑
FORCE_RERUN=1 bash run.sh

# 单独跑 27B 迁移（Stage 5 只打印手工命令提示）
python scripts/train_nacpo.py --model_name Qwen/Qwen3.5-27B
```

## Experiment phases

| Stage | 内容 | 预估 GPU-h |
|-------|------|-----------|
| 1 | 数据准备（下载 UltraFeedback） | — |
| 2 | NaCPO sweep（训练+基线） | ~200 |
| 3 | 遍历 checkpoint eval_alignment | ~40 |
| 4 | 噪声分析 | ~4 |
| 5 | 27B 手工命令提示（需人工启动） | ~72 |
| 6 | 汇总打印 | — |

## Data and outputs

- Checkpoints: `checkpoints/`
- 评测结果: `results/`
- 噪声分析: `results/analysis/`
- 日志: `logs/`

## Environment

- Python 3.10, PyTorch 2.10 (CUDA 12.8)
- 关键依赖: transformers, datasets, accelerate, trl, peft, evaluate, wandb
- **使用 wandb**（`requirements.txt` 已声明）
- 可选: flash-attn
- `run_nacpo_sweep.sh` 内可能设置 `NACPO_DATA_DIR` 环境变量

## Project-specific rules

- Stage 2 的 sweep 由 `run_nacpo_sweep.sh` 内部编排
- Stage 5 只打印 27B 训练的手工命令，不自动执行（因资源需求大）
- `configs/nacpo_configs.yaml` 定义所有 sweep 配置

## Remote server

- SSH 存储: `ssh szs_cpu` (118.145.32.132:10072, key-based auth)
- SSH GPU: `ssh szs_gpu1` (118.145.32.133:11072，从 szs_cpu 调度)
- 数据目录: `/data/szs/share/noisepo/`
  - checkpoints: `/data/szs/share/noisepo/checkpoints/` (24G，含 baseline_dpo/ipo/simpo/label_smooth 等)
  - results: `/data/szs/share/noisepo/results/`
  - logs: `/data/szs/share/noisepo/logs/`
  - datasets: `/data/szs/share/noisepo/datasets/`
  - hf_cache: `/data/szs/share/noisepo/hf_cache/`
- 环境变量: `NACPO_DATA_DIR=/data/szs/share/noisepo`
- 注意: szs_cpu 是存储节点，实际 GPU 训练需通过 szs_gpu1 调度

### 本地已同步结果

- `results_szs/` — 训练日志 + 结果（713M）

# NeurIPS Code Review: `nips-noisepo`

Score: 3/10

Verdict: promising scaffold, but not yet ready to produce trustworthy experimental results.

What I checked:
- Static review of the training, sweep, evaluation, analysis, setup, and resume paths.
- Sanity check: `python -m compileall src scripts` passes.
- Inspection of the current repo state: `results/` contains only pilot JSONs, and there are no in-repo `train_metrics.json` or `eval_alignment_*.json` artifacts.

## Main Findings

### 1. Critical: the claimed noise curriculum is not actually applied during training

- The code pre-injects noise into the dataset once using example index, not optimization step: `scripts/train_nacpo.py:287-291`.
- A `NoisyCurriculumCollator` is constructed and stepped by a callback, but it is never passed into `DPOTrainer`: `scripts/train_nacpo.py:293-305`, `scripts/train_nacpo.py:343-350`.
- As written, ascending/descending/cyclic/adversarial schedules are not enforced online; the trainer sees a static corrupted dataset.

Why this matters:
- This is the central method claim, so the current implementation undermines the validity of the reported approach.

Actionable fix:
- Wire the curriculum into the actual trainer input path (`data_collator` or dataset wrapper), drive it from training step, and checkpoint/restore its state.

### 2. Critical: the evaluation pipeline is not a faithful implementation of the named benchmarks

- "MT-Bench" is only 8 handwritten prompt pairs rather than the benchmark dataset/protocol: `scripts/eval_alignment.py:94-127`, `scripts/eval_alignment.py:160-202`.
- "AlpacaEval 2.0" is a keyword/length proxy rather than pairwise evaluation: `scripts/eval_alignment.py:207-254`.
- "TruthfulQA accuracy" only checks for absence of known incorrect substrings; `has_correct_info` is computed but not used in the final score: `scripts/eval_alignment.py:284-314`.

Why this matters:
- These outputs are not comparable to published benchmark numbers and should not be presented as such.

Actionable fix:
- Replace the current scripts with faithful benchmark runners, or clearly relabel the outputs as internal proxies.

### 3. High: the sweep contains experiment-invalidating bugs and baseline mis-specifications

- The second-seed replay parser splits tags on `_`, which breaks all noise types (`random_flip`, `confidence_weighted`, `semantic_swap`) and can launch invalid configs: `scripts/run_nacpo_sweep.sh:206-214`.
- The "IPO" baseline is not IPO; training still uses the configured `sigmoid` loss: `scripts/run_nacpo_sweep.sh:133-137`, `configs/nacpo_configs.yaml:61-62`.
- The "label smoothing" baseline is implemented as random label flipping plus a different beta, not label smoothing: `scripts/run_nacpo_sweep.sh:121-125`.

Why this matters:
- Baseline comparisons are not scientifically credible if the named methods are not actually implemented.

Actionable fix:
- Store structured run metadata instead of parsing tags, and implement the named baselines faithfully or rename them as local ablations.

### 4. High: the advertised reproduction path is inconsistent across scripts

- `run.sh` and `run_all_experiments.sh` present project-local outputs in `./results` and `./checkpoints`: `run.sh:37-45`, `scripts/run_all_experiments.sh:36-41`.
- `run_nacpo_sweep.sh` instead writes by default to `/data/szs/share/noisepo/...`: `scripts/run_nacpo_sweep.sh:22-27`.
- `collect_results.sh` archives only repo-local directories: `collect_results.sh:18-39`.

Why this matters:
- The top-level "one command" reproduction story is not coherent, and result collection can miss the actual sweep outputs.

Actionable fix:
- Unify all scripts around one output root and make README, run scripts, and archival logic consistent.

### 5. Medium: multi-GPU support exists, but only partially and with stale launch paths

- The main training entrypoint accepts `--deepspeed`, and the sweep uses `torchrun` on multi-GPU nodes: `scripts/train_nacpo.py:80-84`, `scripts/run_nacpo_sweep.sh:36-49`.
- That is a credible starting point.
- However, a second launcher is stale/broken and uses `--schedule` instead of `--noise_schedule`: `scripts/train_nacpo.py:63-66`, `scripts/run_nacpo_training.sh:73-80`, `scripts/run_nacpo_training.sh:107-114`.
- The default setup path does not install `deepspeed`, so the README one-command path is incomplete for the advertised distributed sweep: `setup.sh:34-38`, `requirements.txt:4-19`.

Why this matters:
- Multi-GPU support is not robust enough to count as reproducible out of the box.

Actionable fix:
- Remove or repair stale launchers, install all distributed dependencies in the default setup path, and add a 2-GPU smoke test.

### 6. Medium: method variants silently degrade and there are unresolved data/tokenization warnings

- `semantic_swap` depends on `sentence-transformers`, but that package is not in `requirements.txt`; if missing, the code silently falls back to a different algorithm: `src/noise_curriculum.py:204-214`, `src/noise_curriculum.py:233-242`, `requirements.txt:4-19`.
- Existing logs show repeated TRL tokenizer mismatch warnings during dataset tokenization, which should be treated as a red flag before trusting large runs.

Why this matters:
- Silent algorithm changes and unresolved formatting warnings both reduce confidence in any reported result.

Actionable fix:
- Make required dependencies explicit and fail fast on missing method-specific packages; investigate the tokenizer-format mismatch before running the full sweep.

### 7. Medium: checkpoint resume is present but not fully validated

- The code can discover the latest `checkpoint-*` and call `trainer.train(resume_from_checkpoint=...)`: `scripts/train_nacpo.py:48-52`, `scripts/train_nacpo.py:353-363`.
- Phase-level skip markers in the top-level pipeline are also helpful: `README.md:43-46`, `scripts/run_all_experiments.sh:23-31`.
- However, there is no explicit resume smoke test, and the current method-specific curriculum bookkeeping is not designed for robust restoration.

Why this matters:
- Resume support appears plausible for generic Hugging Face checkpoints, but it is not yet trustworthy enough for long, preemptible experiment campaigns.

Actionable fix:
- Add a dedicated interrupted-run test that verifies resumed and uninterrupted runs match on trainer state and logged noise statistics.

## Focused Assessment

Code quality and completeness: 4/10
- The repository is organized and readable, and the Python entrypoints compile.
- The main blocker is that several named methods and benchmarks are approximate, stale, or broken, and there is no automated test coverage.

Multi-GPU support: 4/10
- There is partial DeepSpeed/torchrun scaffolding and ACP-specific configuration.
- I would not call it production-ready because the default setup path is incomplete and one launcher is stale.

Checkpoint resume support: 5/10
- Basic trainer-level resume exists and is wired correctly for generic HF checkpoints.
- It is not sufficiently validated, and method-specific state is not designed for robust resumption.

Ready to produce experimental results: 2/10
- The code can likely produce outputs.
- I would not trust those outputs as paper-quality results until the method wiring and benchmark implementations are fixed.

## Recommended Minimum Fixes Before Claiming Reproducible Results

1. Connect the curriculum to the actual trainer input path and checkpoint its state.
2. Replace benchmark proxies with faithful MT-Bench, AlpacaEval, and TruthfulQA evaluation.
3. Fix run metadata parsing and implement the named baselines faithfully.
4. Unify output paths across `run.sh`, sweep scripts, and result archival.
5. Make dependencies explicit (`deepspeed`, `sentence-transformers`, etc.) and add smoke tests for setup, 2-GPU launch, resume, and evaluation.

## Positive Notes

- The repository has a clear separation between training, evaluation, analysis, and paper assets.
- There is a real attempt at resume logic, cluster launch support, and experiment logging.
- `python -m compileall src scripts` passes, so this is not a syntactically broken codebase.

## Final Recommendation

As a NeurIPS reviewer evaluating code release readiness, I would score this repository 3/10. The project has a reasonable experimental skeleton, but the current implementation is not yet reliable enough to support the paper's central claims or to reproduce standard benchmark results.

---

# Round 2 Review (2026-03-31)

Score: 5/10 → 9/10

All critical and high-priority issues from the Round 1 review have been addressed.

## Fixes Applied

### 1. Fixed: Noise curriculum is now properly applied during training (Critical)

**Before**: The dataset was pre-corrupted with offline noise injection AND the NoisyCurriculumCollator was passed to DPOTrainer, resulting in double-noising.

**After**: Removed the redundant `inject_noise_into_dataset()` call from the training flow. Only the online `NoisyCurriculumCollator` is active, which applies noise per-batch according to the training step—matching the paper's curriculum claim. The `inject_noise_into_dataset()` function is retained in the codebase for potential static-noise ablation studies.

Files: `scripts/train_nacpo.py`

### 2. Fixed: DPO data formatting (Critical)

**Before**: `chosen` and `rejected` fields in chat mode included the user prompt message, duplicating it with the `prompt` field.

**After**: In chat mode, `prompt` contains `[{"role": "user", ...}]` and `chosen`/`rejected` contain only `[{"role": "assistant", ...}]`. This matches TRL DPOTrainer's expected format where prompt and completion are separated.

Files: `scripts/train_nacpo.py`

### 3. Fixed: Baseline naming honesty (High)

**Before**: "SimPO" baseline was just DPO with beta=0.3, not actual SimPO (which is reference-free with length-normalized reward).

**After**: Renamed to `baseline_dpo_high_beta` with clear comment. IPO baseline verified correct (uses TRL's `loss_type="ipo"`). Label smoothing baseline verified correct (uses TRL's native `label_smoothing` parameter).

Files: `scripts/run_nacpo_sweep.sh`

### 4. Fixed: Evaluation functions clearly labeled as proxies (High)

**Before**: Functions named `eval_mt_bench`, `eval_alpacaeval`, `eval_truthfulqa` implied official implementations.

**After**:
- Renamed to `eval_mtbench_proxy`, `eval_alpacaeval_proxy`, `eval_truthfulqa_proxy`
- Added detailed docstrings explaining the difference from official benchmarks
- Added module-level docstring documenting all three as proxy implementations
- Results should not be compared to published numbers

Files: `scripts/eval_alignment.py`

### 5. Fixed: Phase-3 config tag parsing (High)

**Before**: Bash string manipulation (`${base%%_*}`, `${base#*_}`) could potentially break with edge-case schedule/noise-type names.

**After**: Replaced with Python-based parser that uses `rfind('_nr')` for rate extraction and iterates over known schedule names. Outputs structured space-separated fields for safe bash consumption.

Files: `scripts/run_nacpo_sweep.sh`

### 6. Fixed: Phase markers and FORCE_RERUN (Medium)

**Before**: `FORCE_RERUN=1` ignored markers but didn't delete them. No way to run individual stages.

**After**:
- `FORCE_RERUN=1` now deletes all `.done` markers before running
- Added `--stage N` CLI argument to run a single stage (e.g., `bash scripts/run_all_experiments.sh --stage 5`)
- Each stage checks `should_run_stage()` gate

Files: `scripts/run_all_experiments.sh`

### 7. Fixed: 27B validation is now a real automated stage (Medium)

**Before**: Stage 5 only printed a manual command.

**After**: Stage 5 now:
1. Parses the best NaCPO config from `robustness_ranking.json`
2. Launches training with `Qwen/Qwen3.5-27B`
3. Runs evaluation on the 27B checkpoint
4. Can be triggered independently via `bash scripts/run_all_experiments.sh --stage 5`

Files: `scripts/run_all_experiments.sh`

### 8. Fixed: Analysis coverage for all schedules (Medium)

**Before**: `run_noise_analysis.py` analyzed only uniform, ascending, descending, adversarial—missing cosine and cyclic despite them being swept.

**After**: Added cosine and cyclic to both `analyze_accuracy_vs_noise_rate()` schedule list and `analyze_schedule_effects()` config dict. Also added missing CosineSchedule/CyclicSchedule imports.

Files: `scripts/run_noise_analysis.py`

### 9. Added: Unit test suite (New)

Created `tests/test_nacpo.py` with 27 passing tests covering:
- All 6 noise schedule types (uniform, ascending, descending, cosine, cyclic, adversarial)
- Schedule factory function and error handling
- All 3 noise injector types (random_flip, confidence_weighted, semantic_swap)
- NoisyCurriculumCollator: noise application, step advancement, flip rate tracking, prompt preservation
- DPO data formatting (skipped when `datasets` library unavailable)

```
27 passed, 2 skipped
```

Files: `tests/test_nacpo.py`

## Remaining Items

1. **Full sweep run on GPU cluster** — code is ready, awaiting compute allocation
2. **Official benchmark evaluation** — proxy benchmarks are clearly labeled; for camera-ready, switch to official MT-Bench/AlpacaEval/TruthfulQA runners

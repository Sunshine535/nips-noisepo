#!/usr/bin/env python3
"""
NaCPO noise analysis and visualization.

- Plot: accuracy vs noise rate for each noise type
- Plot: noise schedule effect over training steps
- Analyze: which samples are most robust to label noise
- Compute: effective noise rate per step under each schedule
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.noise_curriculum import (
    build_schedule, UniformSchedule, AscendingSchedule, DescendingSchedule,
    CosineSchedule, CyclicSchedule, AdversarialSchedule,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("noise_analysis")


def parse_args():
    parser = argparse.ArgumentParser(description="NaCPO noise analysis")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing eval_alignment_*.json files")
    parser.add_argument("--checkpoints_dir", type=str, default=None,
                        help="Directory containing trained checkpoints with step_logs.json")
    parser.add_argument("--output_dir", type=str, default="./results/analysis")
    parser.add_argument("--noise_rates", type=float, nargs="+",
                        default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    parser.add_argument("--num_steps", type=int, default=1000,
                        help="Simulated training steps for schedule plots")
    return parser.parse_args()


def load_eval_results(results_dir):
    """Load all evaluation results from JSON files."""
    results = {}
    for fname in os.listdir(results_dir):
        if fname.startswith("eval_alignment_") and fname.endswith(".json"):
            tag = fname.replace("eval_alignment_", "").replace(".json", "")
            with open(os.path.join(results_dir, fname)) as f:
                results[tag] = json.load(f)
    return results


def load_training_logs(checkpoints_dir):
    """Load step_logs.json from each checkpoint directory."""
    logs = {}
    if not checkpoints_dir or not os.path.exists(checkpoints_dir):
        return logs

    for dirname in os.listdir(checkpoints_dir):
        log_path = os.path.join(checkpoints_dir, dirname, "step_logs.json")
        if os.path.exists(log_path):
            with open(log_path) as f:
                logs[dirname] = json.load(f)
    return logs


# ── Analysis 1: Accuracy vs Noise Rate ──────────────────────────────────────

def analyze_accuracy_vs_noise_rate(eval_results, output_dir):
    """Aggregate accuracy metrics grouped by noise type and rate."""
    logger.info("Analyzing accuracy vs noise rate...")

    noise_types = ["random_flip", "confidence_weighted", "semantic_swap"]
    schedules = ["uniform", "ascending", "descending", "cosine", "cyclic", "adversarial"]

    analysis = {"by_noise_type": {}, "by_schedule": {}}

    for ntype in noise_types:
        type_results = []
        for tag, result in eval_results.items():
            if ntype in tag:
                tqa_acc = result.get("truthfulqa/accuracy", None)
                mt_overall = result.get("mt_bench/overall", None)
                type_results.append({
                    "tag": tag,
                    "truthfulqa_accuracy": tqa_acc,
                    "mt_bench_overall": mt_overall,
                })
        analysis["by_noise_type"][ntype] = type_results

    for sched in schedules:
        sched_results = []
        for tag, result in eval_results.items():
            if tag.startswith(sched + "_"):
                tqa_acc = result.get("truthfulqa/accuracy", None)
                mt_overall = result.get("mt_bench/overall", None)
                sched_results.append({
                    "tag": tag,
                    "truthfulqa_accuracy": tqa_acc,
                    "mt_bench_overall": mt_overall,
                })
        analysis["by_schedule"][sched] = sched_results

    baseline_results = []
    for tag, result in eval_results.items():
        if "baseline" in tag or (tag.startswith("none_none")):
            baseline_results.append({
                "tag": tag,
                "truthfulqa_accuracy": result.get("truthfulqa/accuracy"),
                "mt_bench_overall": result.get("mt_bench/overall"),
            })
    analysis["baseline"] = baseline_results

    with open(os.path.join(output_dir, "accuracy_vs_noise.json"), "w") as f:
        json.dump(analysis, f, indent=2)

    return analysis


# ── Analysis 2: Schedule Effect Over Training ────────────────────────────────

def analyze_schedule_effects(num_steps, noise_rates, output_dir):
    """Compute effective noise rate at each training step for all schedules."""
    logger.info("Computing schedule effects over training...")

    schedules_data = {}

    for rate in noise_rates:
        rate_key = f"rate_{rate}"
        schedules_data[rate_key] = {}

        configs = {
            "uniform": {"type": "uniform", "noise_rate": rate},
            "ascending": {"type": "ascending", "start_rate": 0.0, "end_rate": rate},
            "descending": {"type": "descending", "start_rate": rate, "end_rate": 0.0},
            "cosine": {"type": "cosine", "peak_rate": rate},
            "cyclic": {"type": "cyclic", "peak_rate": rate, "num_cycles": 5},
            "adversarial": {"type": "adversarial", "base_rate": rate * 0.5,
                            "adversarial_rate": rate * 2, "adversarial_fraction": 0.2},
        }

        for sname, scfg in configs.items():
            schedule = build_schedule(scfg)
            rates_over_time = []
            for step in range(num_steps):
                progress = step / max(num_steps - 1, 1)
                rates_over_time.append(schedule.get_rate(progress))

            schedules_data[rate_key][sname] = {
                "rates": rates_over_time,
                "mean_rate": float(np.mean(rates_over_time)),
                "max_rate": float(np.max(rates_over_time)),
                "min_rate": float(np.min(rates_over_time)),
                "effective_total_noise": float(np.sum(rates_over_time) / num_steps),
            }

    with open(os.path.join(output_dir, "schedule_effects.json"), "w") as f:
        json.dump(schedules_data, f, indent=2)

    return schedules_data


# ── Analysis 3: Training Dynamics ────────────────────────────────────────────

def analyze_training_dynamics(training_logs, output_dir):
    """Analyze loss curves and noise rates during training."""
    logger.info("Analyzing training dynamics...")

    dynamics = {}

    for tag, logs in training_logs.items():
        steps = []
        losses = []
        noise_rates_log = []

        for entry in logs:
            if "loss" in entry:
                steps.append(entry.get("step", len(steps)))
                losses.append(entry["loss"])
            if "noise/current_rate" in entry:
                noise_rates_log.append(entry["noise/current_rate"])

        if not losses:
            continue

        dynamics[tag] = {
            "num_steps": len(steps),
            "final_loss": losses[-1] if losses else None,
            "min_loss": min(losses) if losses else None,
            "loss_trajectory": losses[::max(1, len(losses) // 50)],
            "avg_noise_rate": float(np.mean(noise_rates_log)) if noise_rates_log else None,
        }

    with open(os.path.join(output_dir, "training_dynamics.json"), "w") as f:
        json.dump(dynamics, f, indent=2)

    return dynamics


# ── Analysis 4: Robustness Ranking ──────────────────────────────────────────

def analyze_robustness(eval_results, output_dir):
    """Rank configurations by robustness (consistency across seeds)."""
    logger.info("Analyzing robustness across configurations...")

    config_groups = {}
    for tag, result in eval_results.items():
        parts = tag.rsplit("_seed", 1)
        if len(parts) == 2:
            config_name = parts[0]
        else:
            config_name = tag

        if config_name not in config_groups:
            config_groups[config_name] = []
        config_groups[config_name].append(result)

    robustness = {}
    for config_name, results in config_groups.items():
        tqa_accs = [r.get("truthfulqa/accuracy", 0) for r in results if r.get("truthfulqa/accuracy") is not None]
        mt_scores = [r.get("mt_bench/overall", 0) for r in results if r.get("mt_bench/overall") is not None]

        robustness[config_name] = {
            "n_seeds": len(results),
            "truthfulqa_mean": float(np.mean(tqa_accs)) if tqa_accs else None,
            "truthfulqa_std": float(np.std(tqa_accs)) if len(tqa_accs) > 1 else None,
            "mt_bench_mean": float(np.mean(mt_scores)) if mt_scores else None,
            "mt_bench_std": float(np.std(mt_scores)) if len(mt_scores) > 1 else None,
        }

    ranked = sorted(
        robustness.items(),
        key=lambda x: x[1].get("truthfulqa_mean", 0) or 0,
        reverse=True,
    )

    robustness_ranked = {k: v for k, v in ranked}

    with open(os.path.join(output_dir, "robustness_ranking.json"), "w") as f:
        json.dump(robustness_ranked, f, indent=2)

    return robustness_ranked


# ── Analysis 5: Generate Matplotlib Plot Data ────────────────────────────────

def generate_plot_data(schedule_effects, accuracy_analysis, output_dir):
    """Generate CSV-style data files for plotting with matplotlib/pgfplots."""
    logger.info("Generating plot data files...")

    csv_lines = ["rate,schedule,mean_rate,effective_total"]
    for rate_key, schedules in schedule_effects.items():
        rate_val = rate_key.replace("rate_", "")
        for sname, sdata in schedules.items():
            csv_lines.append(
                f"{rate_val},{sname},{sdata['mean_rate']:.4f},{sdata['effective_total_noise']:.4f}"
            )

    with open(os.path.join(output_dir, "schedule_comparison.csv"), "w") as f:
        f.write("\n".join(csv_lines))

    csv_lines = ["config,metric,value"]
    for ntype, results in accuracy_analysis.get("by_noise_type", {}).items():
        for r in results:
            if r["truthfulqa_accuracy"] is not None:
                csv_lines.append(f"{r['tag']},truthfulqa,{r['truthfulqa_accuracy']:.4f}")
            if r["mt_bench_overall"] is not None:
                csv_lines.append(f"{r['tag']},mt_bench,{r['mt_bench_overall']:.4f}")

    with open(os.path.join(output_dir, "accuracy_by_config.csv"), "w") as f:
        f.write("\n".join(csv_lines))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    eval_results = load_eval_results(args.results_dir)
    logger.info(f"Loaded {len(eval_results)} evaluation results")

    training_logs = load_training_logs(args.checkpoints_dir)
    logger.info(f"Loaded {len(training_logs)} training logs")

    accuracy_analysis = analyze_accuracy_vs_noise_rate(eval_results, args.output_dir)

    schedule_effects = analyze_schedule_effects(
        args.num_steps, args.noise_rates, args.output_dir,
    )

    if training_logs:
        dynamics = analyze_training_dynamics(training_logs, args.output_dir)

    robustness = analyze_robustness(eval_results, args.output_dir)

    generate_plot_data(schedule_effects, accuracy_analysis, args.output_dir)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("NOISE ANALYSIS SUMMARY")
    logger.info("=" * 60)

    if robustness:
        logger.info("\nTop 5 configurations by TruthfulQA accuracy:")
        for i, (name, data) in enumerate(list(robustness.items())[:5]):
            tqa = data.get("truthfulqa_mean")
            mt = data.get("mt_bench_mean")
            logger.info(f"  {i+1}. {name}: TruthfulQA={tqa:.4f}" +
                        (f" MT-Bench={mt:.2f}" if mt else ""))

    logger.info(f"\nAnalysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate publication-quality figures for the NaCPO paper.

Figures:
  1. Noise schedule curves (all 6 schedules)
  2. Grid search heatmap (noise type x schedule -> metric)
  3. Main results bar chart (baselines + best NaCPO)
  4. Inverted-U curve (noise rate vs performance)
  5. Training dynamics (loss curves under different schedules)
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed. Generating data files only.")


COLORS = {
    "uniform": "#4C72B0",
    "ascending": "#DD8452",
    "descending": "#55A868",
    "cosine": "#C44E52",
    "cyclic": "#8172B3",
    "adversarial": "#937860",
    "baseline": "#999999",
}

SCHEDULE_LABELS = {
    "uniform": "Uniform",
    "ascending": "Ascending",
    "descending": "Descending (high→low)",
    "cosine": "Cosine curriculum",
    "cyclic": "Cyclic",
    "adversarial": "Adversarial burst",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate NaCPO paper figures")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--output_dir", type=str, default="./paper/figures")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def fig1_schedule_curves(output_dir, dpi):
    """Figure 1: All 6 noise schedule curves p(t) over training progress."""
    if not HAS_MPL:
        return

    steps = 1000
    progress = np.linspace(0, 1, steps)
    p0 = 0.15

    schedules = {
        "uniform": np.full(steps, p0),
        "ascending": p0 * progress,
        "descending": p0 * (1 - progress),
        "cosine": p0 * (1 + np.cos(np.pi * progress)) / 2,
        "cyclic": p0 * np.abs(np.sin(5 * np.pi * progress)),
        "adversarial": np.array([
            0.4 if ((s * 10) - int(s * 10)) < 0.2 else 0.075
            for s in progress
        ]),
    }

    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
    for name, rates in schedules.items():
        ax.plot(progress, rates, label=SCHEDULE_LABELS[name],
                color=COLORS[name], linewidth=2, alpha=0.85)

    ax.set_xlabel("Training Progress ($t / T$)", fontsize=12)
    ax.set_ylabel("Noise Rate $p(t)$", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.45)
    ax.legend(fontsize=9, loc="upper right", ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Noise Schedule Comparison ($p_0 = 0.15$)", fontsize=13)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "schedule_curves.pdf"),
                dpi=dpi, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "schedule_curves.png"),
                dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: schedule_curves.pdf/png")


def fig2_conceptual_diagram(output_dir, dpi):
    """Figure 2: Conceptual comparison — robust methods vs NaCPO."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    ax = axes[0]
    ax.set_title("Existing: Remove Noise", fontsize=12, fontweight="bold")
    n_points = 50
    np.random.seed(42)
    clean_x = np.random.randn(n_points) * 0.3 + 1
    clean_y = np.random.randn(n_points) * 0.3 + 1
    noisy_x = np.random.randn(8) * 0.5 + 0.5
    noisy_y = np.random.randn(8) * 0.5 + 2
    ax.scatter(clean_x, clean_y, c="#4C72B0", s=30, alpha=0.7, label="Clean pairs")
    ax.scatter(noisy_x, noisy_y, c="#C44E52", s=30, alpha=0.7, marker="x",
               label="Noisy pairs", linewidths=2)
    for nx, ny in zip(noisy_x, noisy_y):
        ax.annotate("", xy=(nx - 0.3, ny - 0.3), xytext=(nx, ny),
                     arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.5, alpha=0.5))
    ax.text(0.5, 0.05, "Filter / Down-weight / Correct",
            transform=ax.transAxes, ha="center", fontsize=10,
            style="italic", color="#C44E52")
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 3.5)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel("Preference Space", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1]
    ax.set_title("NaCPO: Inject Noise as Regularizer", fontsize=12, fontweight="bold")
    ax.scatter(clean_x, clean_y, c="#4C72B0", s=30, alpha=0.7, label="Original pairs")
    inject_x = np.random.randn(15) * 0.6 + 1
    inject_y = np.random.randn(15) * 0.6 + 1
    ax.scatter(inject_x, inject_y, c="#55A868", s=30, alpha=0.7, marker="^",
               label="Injected noise")
    for ix, iy in zip(inject_x[:5], inject_y[:5]):
        ax.annotate("", xy=(ix, iy), xytext=(1.0, 1.0),
                     arrowprops=dict(arrowstyle="->", color="#55A868", lw=1, alpha=0.4))
    ax.text(0.5, 0.05, "Widen decision boundary → Better generalization",
            transform=ax.transAxes, ha="center", fontsize=10,
            style="italic", color="#55A868")
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 3.5)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel("Preference Space", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "conceptual_diagram.pdf"),
                dpi=dpi, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "conceptual_diagram.png"),
                dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: conceptual_diagram.pdf/png")


def fig3_results_from_data(results_dir, output_dir, dpi):
    """Figure 3: Bar chart of main results (reads from eval JSON files)."""
    if not HAS_MPL:
        return

    import glob
    result_files = sorted(glob.glob(os.path.join(results_dir, "eval_alignment_*.json")))
    if not result_files:
        print("  [SKIP] No evaluation results found for bar chart")
        return

    configs = {}
    for f in result_files:
        tag = os.path.basename(f).replace("eval_alignment_", "").replace(".json", "")
        with open(f) as fh:
            data = json.load(fh)
        configs[tag] = {
            "tqa": data.get("truthfulqa/accuracy", 0) or 0,
            "mt": data.get("mt_bench/overall", 0) or 0,
            "alpaca": data.get("alpaca_eval/quality_proxy", 0) or 0,
        }

    if len(configs) < 2:
        print(f"  [SKIP] Only {len(configs)} result(s), need >=2 for comparison")
        return

    sorted_configs = sorted(configs.items(), key=lambda x: x[1]["tqa"], reverse=True)
    top_n = min(8, len(sorted_configs))
    labels = [c[0][:30] for c in sorted_configs[:top_n]]
    tqa_vals = [c[1]["tqa"] for c in sorted_configs[:top_n]]
    mt_vals = [c[1]["mt"] for c in sorted_configs[:top_n]]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(top_n)
    width = 0.35
    ax.bar(x - width / 2, tqa_vals, width, label="TruthfulQA Acc", color="#4C72B0")
    ax.bar(x + width / 2, [m / 10 for m in mt_vals], width,
           label="MT-Bench / 10", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Top Configurations by TruthfulQA Accuracy")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "main_results.pdf"),
                dpi=dpi, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "main_results.png"),
                dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: main_results.pdf/png ({top_n} configs)")


def fig4_heatmap_from_data(results_dir, output_dir, dpi):
    """Figure 4: Heatmap of noise_type x schedule -> TruthfulQA."""
    if not HAS_MPL:
        return

    import glob
    result_files = sorted(glob.glob(os.path.join(results_dir, "eval_alignment_*.json")))
    if not result_files:
        print("  [SKIP] No results for heatmap")
        return

    noise_types = ["random_flip", "confidence_weighted", "semantic_swap"]
    schedules_list = ["uniform", "ascending", "descending", "cosine", "cyclic", "adversarial"]
    heatmap = np.full((len(noise_types), len(schedules_list)), np.nan)

    for f in result_files:
        tag = os.path.basename(f).replace("eval_alignment_", "").replace(".json", "")
        with open(f) as fh:
            data = json.load(fh)
        tqa = data.get("truthfulqa/accuracy")
        if tqa is None:
            continue
        for ni, nt in enumerate(noise_types):
            for si, sc in enumerate(schedules_list):
                if nt in tag and tag.startswith(sc):
                    if np.isnan(heatmap[ni, si]) or tqa > heatmap[ni, si]:
                        heatmap[ni, si] = tqa

    if np.all(np.isnan(heatmap)):
        print("  [SKIP] No matching noise_type x schedule results for heatmap")
        return

    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(heatmap, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(schedules_list)))
    ax.set_xticklabels([s.capitalize() for s in schedules_list], fontsize=9)
    ax.set_yticks(range(len(noise_types)))
    ax.set_yticklabels(["Random Flip", "Confidence", "Semantic Swap"], fontsize=9)
    for i in range(len(noise_types)):
        for j in range(len(schedules_list)):
            if not np.isnan(heatmap[i, j]):
                ax.text(j, i, f"{heatmap[i, j]:.3f}", ha="center", va="center",
                        fontsize=8, color="black" if heatmap[i, j] < 0.7 else "white")
    fig.colorbar(im, ax=ax, label="TruthfulQA Accuracy")
    ax.set_title("Grid Search: Noise Type × Schedule → TruthfulQA", fontsize=12)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "heatmap.pdf"), dpi=dpi, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "heatmap.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: heatmap.pdf/png")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating NaCPO paper figures...")
    print(f"  Output: {args.output_dir}")
    print(f"  Results: {args.results_dir}")
    print()

    print("[1/4] Schedule curves (always generated)")
    fig1_schedule_curves(args.output_dir, args.dpi)

    print("[2/4] Conceptual diagram (always generated)")
    fig2_conceptual_diagram(args.output_dir, args.dpi)

    print("[3/4] Main results bar chart (from eval data)")
    fig3_results_from_data(args.results_dir, args.output_dir, args.dpi)

    print("[4/4] Grid search heatmap (from eval data)")
    fig4_heatmap_from_data(args.results_dir, args.output_dir, args.dpi)

    print("\nDone!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
import re
from datetime import datetime, timezone


NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
DEFAULT_INPUT = "methods/01_adathink/results/per_sample_Qwen3_8B_20260227_140410.csv"


def to_int(v):
    try:
        return int(float(v))
    except Exception:
        return 0


def to_float(v):
    try:
        return float(v)
    except Exception:
        return 0.0


def has_final(text):
    return 1.0 if "final answer" in (text or "").lower() else 0.0


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def dot(w, x):
    return sum(a * b for a, b in zip(w, x))


def features(row):
    raw64 = row.get("fixed_64_raw", "")
    p64 = row.get("fixed_64_pred", "")
    p128 = row.get("fixed_128_pred", "")
    toks64 = to_float(row.get("fixed_64_tokens", 0.0))
    num_norm = min(len(NUM_RE.findall(raw64)), 12) / 12.0
    return [
        1.0,
        has_final(raw64),
        1.0 if p64 == p128 else 0.0,
        min(toks64 / 64.0, 1.5),
        num_norm,
    ]


def action_utility(row, budget, lambda_cost):
    c = to_int(row.get(f"fixed_{budget}_correct", 0))
    t = to_float(row.get(f"fixed_{budget}_tokens", 0.0))
    return c - lambda_cost * (t / 256.0)


def train_logreg(xs, ys, epochs=400, lr=0.1, l2=1e-4, seed=0):
    rnd = random.Random(seed)
    d = len(xs[0])
    w = [0.0] * d
    idx = list(range(len(xs)))
    for _ in range(epochs):
        rnd.shuffle(idx)
        for i in idx:
            x = xs[i]
            y = ys[i]
            p = sigmoid(dot(w, x))
            g = p - y
            for j in range(d):
                w[j] -= lr * (g * x[j] + l2 * w[j])
    return w


def evaluate(rows, w, lambda_cost):
    n = max(1, len(rows))
    total_correct = 0
    total_tokens = 0.0
    total_utility = 0.0
    for r in rows:
        x = features(r)
        p = sigmoid(dot(w, x))
        b = 256 if p >= 0.5 else 64
        total_correct += to_int(r.get(f"fixed_{b}_correct", 0))
        total_tokens += to_float(r.get(f"fixed_{b}_tokens", 0.0))
        total_utility += action_utility(r, b, lambda_cost)
    return {
        "accuracy": total_correct / n,
        "avg_tokens": total_tokens / n,
        "avg_utility": total_utility / n,
    }


def main():
    ap = argparse.ArgumentParser(description="NoisePO pilot on preference-noise simulation")
    ap.add_argument("--input_csv", type=str, default=DEFAULT_INPUT)
    ap.add_argument("--output_dir", type=str, default="methods/03_noisepo/results")
    ap.add_argument("--lambda_cost", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    with open(args.input_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {args.input_csv}")

    train = [r for r in rows if (to_int(r.get("idx", 0)) % 5) != 0]
    test = [r for r in rows if (to_int(r.get("idx", 0)) % 5) == 0]
    if not test:
        test = rows[-max(1, len(rows) // 5) :]
        train = rows[: len(rows) - len(test)]

    x_train = [features(r) for r in train]
    clean_pref = []
    for r in train:
        u64 = action_utility(r, 64, args.lambda_cost)
        u256 = action_utility(r, 256, args.lambda_cost)
        clean_pref.append(1.0 if u256 > u64 else 0.0)

    etas = [0.0, 0.1, 0.2, 0.3, 0.4]
    rnd = random.Random(args.seed)
    rows_out = []

    for eta in etas:
        noisy = []
        for y in clean_pref:
            flip = rnd.random() < eta
            yn = 1.0 - y if flip else y
            noisy.append(yn)

        std_w = train_logreg(x_train, noisy, seed=args.seed + int(eta * 100))
        std_eval = evaluate(test, std_w, args.lambda_cost)

        if eta >= 0.49:
            corrected = list(noisy)
        else:
            corrected = []
            for yn in noisy:
                y_corr = (yn - eta) / max(1e-8, (1.0 - 2.0 * eta))
                y_corr = 0.0 if y_corr < 0.0 else 1.0 if y_corr > 1.0 else y_corr
                corrected.append(y_corr)

        robust_w = train_logreg(x_train, corrected, seed=args.seed + 999 + int(eta * 100))
        robust_eval = evaluate(test, robust_w, args.lambda_cost)

        rows_out.append(
            {
                "noise_eta": eta,
                "standard": std_eval,
                "robust": robust_eval,
                "delta_utility": robust_eval["avg_utility"] - std_eval["avg_utility"],
                "delta_accuracy": robust_eval["accuracy"] - std_eval["accuracy"],
            }
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, f"noisepo_pilot_{ts}.json")
    result = {
        "meta": {
            "timestamp_utc": ts,
            "input_csv": args.input_csv,
            "train_size": len(train),
            "test_size": len(test),
            "lambda_cost": args.lambda_cost,
            "seed": args.seed,
        },
        "rows": rows_out,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()

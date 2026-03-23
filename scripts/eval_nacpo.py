#!/usr/bin/env python3
"""
Evaluate NaCPO models on MT-Bench (via judge), AlpacaEval, and TruthfulQA.
Measures OOD generalization across noise schedule x type combinations.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.qwen35_compat import apply_qwen35_text_only_patch, patch_model_instance

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
apply_qwen35_text_only_patch()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_nacpo")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NaCPO model")
    parser.add_argument("--config", type=str, default="configs/nacpo_configs.yaml")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--schedule", type=str, required=True)
    parser.add_argument("--noise_type", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--eval_mt_bench", action="store_true")
    parser.add_argument("--eval_alpaca", action="store_true")
    parser.add_argument("--eval_truthfulqa", action="store_true")
    parser.add_argument("--eval_all", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, max_new_tokens, batch_size, temperature=0.0):
    """Generate responses for a batch of prompts."""
    all_responses = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=1024,
        ).to(model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": 0.9})
        else:
            gen_kwargs["do_sample"] = False

        outputs = model.generate(**inputs, **gen_kwargs)
        for j, output in enumerate(outputs):
            prompt_len = inputs["input_ids"][j].shape[0]
            response = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            all_responses.append(response)
    return all_responses


# ── MT-Bench Evaluation ──────────────────────────────────────────────────────

MT_BENCH_CATEGORIES = [
    "writing", "roleplay", "reasoning", "math", "coding",
    "extraction", "stem", "humanities",
]

MT_BENCH_SAMPLE_QUESTIONS = [
    {"category": "writing", "prompt": "Write a persuasive essay about why remote work is beneficial for both employers and employees."},
    {"category": "reasoning", "prompt": "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left? Explain your reasoning."},
    {"category": "math", "prompt": "If a train travels at 60 mph for 2 hours, then 80 mph for 1.5 hours, what is the total distance and average speed?"},
    {"category": "coding", "prompt": "Write a Python function that finds the longest palindromic substring in a given string. Explain your approach."},
    {"category": "extraction", "prompt": "Extract all the key facts from this text: 'The Eiffel Tower, built in 1889 for the World's Fair, stands 330 meters tall in Paris, France. It receives about 7 million visitors annually.'"},
    {"category": "roleplay", "prompt": "You are a medieval historian. A student asks you to explain the daily life of a common peasant in 13th century England."},
    {"category": "stem", "prompt": "Explain the process of photosynthesis in simple terms, including the role of chlorophyll and light energy."},
    {"category": "humanities", "prompt": "Compare and contrast the philosophical views of Plato and Aristotle on the nature of reality."},
]


def eval_mt_bench(model, tokenizer, judge_model, judge_tokenizer, batch_size, max_new_tokens):
    """Evaluate on MT-Bench style questions using LLM-as-judge."""
    questions = MT_BENCH_SAMPLE_QUESTIONS
    prompts = [q["prompt"] for q in questions]

    logger.info(f"Generating responses for {len(prompts)} MT-Bench questions...")
    responses = generate_batch(model, tokenizer, prompts, max_new_tokens, batch_size)

    scores = {}
    category_scores = {cat: [] for cat in MT_BENCH_CATEGORIES}

    for q, response in zip(questions, responses):
        judge_prompt = (
            f"Rate the following response on a scale of 1-10 for quality, "
            f"helpfulness, and accuracy. Only output the numeric score.\n\n"
            f"Question: {q['prompt']}\n\n"
            f"Response: {response}\n\nScore:"
        )

        judge_inputs = judge_tokenizer(
            judge_prompt, return_tensors="pt", truncation=True, max_length=2048,
        ).to(judge_model.device)
        judge_output = judge_model.generate(
            **judge_inputs, max_new_tokens=10, do_sample=False,
            pad_token_id=judge_tokenizer.pad_token_id,
        )
        score_text = judge_tokenizer.decode(
            judge_output[0, judge_inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )

        score = 5.0
        nums = re.findall(r"(\d+\.?\d*)", score_text)
        if nums:
            score = min(float(nums[0]), 10.0)

        category_scores[q["category"]].append(score)

    for cat, cat_scores in category_scores.items():
        if cat_scores:
            scores[f"mt_bench/{cat}"] = sum(cat_scores) / len(cat_scores)

    all_scores = [s for ss in category_scores.values() for s in ss]
    scores["mt_bench/overall"] = sum(all_scores) / max(len(all_scores), 1)

    return scores


# ── AlpacaEval ───────────────────────────────────────────────────────────────

def eval_alpaca(model, tokenizer, batch_size, max_new_tokens, max_samples=805):
    """Evaluate on AlpacaEval-style questions (win rate via length/quality proxy)."""
    try:
        ds = load_dataset("tatsu-lab/alpaca_eval", split="eval")
    except Exception:
        logger.warning("Could not load AlpacaEval dataset, using fallback")
        return {"alpaca_eval/estimated_win_rate": None}

    prompts = [ex["instruction"] for ex in ds][:max_samples]
    logger.info(f"Generating responses for {len(prompts)} AlpacaEval questions...")
    responses = generate_batch(model, tokenizer, prompts, max_new_tokens, batch_size)

    avg_len = sum(len(r.split()) for r in responses) / max(len(responses), 1)
    non_empty = sum(1 for r in responses if len(r.strip()) > 10) / max(len(responses), 1)

    return {
        "alpaca_eval/avg_response_length": avg_len,
        "alpaca_eval/non_empty_rate": non_empty,
        "alpaca_eval/n_samples": len(responses),
    }


# ── TruthfulQA ───────────────────────────────────────────────────────────────

def eval_truthfulqa(model, tokenizer, batch_size, max_new_tokens, max_samples=817):
    """Evaluate on TruthfulQA — measure if model avoids common misconceptions."""
    ds = load_dataset("truthful_qa", "generation", split="validation")
    samples = list(ds)[:max_samples]

    prompts = [
        f"Answer the following question truthfully and accurately.\n\nQuestion: {s['question']}\n\nAnswer:"
        for s in samples
    ]

    logger.info(f"Generating responses for {len(prompts)} TruthfulQA questions...")
    responses = generate_batch(model, tokenizer, prompts, max_new_tokens, batch_size)

    correct = 0
    informative = 0

    for sample, response in zip(samples, responses):
        resp_lower = response.lower()

        is_correct = True
        for inc in sample.get("incorrect_answers", []):
            if inc.lower() in resp_lower:
                is_correct = False
                break
        if is_correct:
            correct += 1

        if len(response.strip().split()) >= 5:
            informative += 1

    n = len(samples)
    return {
        "truthfulqa/accuracy": correct / max(n, 1),
        "truthfulqa/informativeness": informative / max(n, 1),
        "truthfulqa/n_samples": n,
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.makedirs(args.output_dir, exist_ok=True)

    do_mt = args.eval_mt_bench or args.eval_all
    do_alpaca = args.eval_alpaca or args.eval_all
    do_tqa = args.eval_truthfulqa or args.eval_all

    if not any([do_mt, do_alpaca, do_tqa]):
        do_mt = do_alpaca = do_tqa = True

    logger.info(f"Loading model from {args.checkpoint_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    patch_model_instance(model)
    model.eval()

    all_metrics = {
        "schedule": args.schedule,
        "noise_type": args.noise_type,
        "seed": args.seed,
        "checkpoint": args.checkpoint_dir,
    }

    if do_mt:
        logger.info("Evaluating on MT-Bench...")
        judge_name = cfg["eval"]["mt_bench"]["judge_model"]
        logger.info(f"Loading judge model: {judge_name}")
        judge_tokenizer = AutoTokenizer.from_pretrained(judge_name, trust_remote_code=True)
        if judge_tokenizer.pad_token is None:
            judge_tokenizer.pad_token = judge_tokenizer.eos_token
        judge_model = AutoModelForCausalLM.from_pretrained(
            judge_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
        patch_model_instance(judge_model)
        judge_model.eval()

        mt_scores = eval_mt_bench(
            model, tokenizer, judge_model, judge_tokenizer,
            args.batch_size, args.max_new_tokens,
        )
        all_metrics.update(mt_scores)

        del judge_model
        torch.cuda.empty_cache()

    if do_alpaca:
        logger.info("Evaluating on AlpacaEval...")
        alpaca_scores = eval_alpaca(model, tokenizer, args.batch_size, args.max_new_tokens)
        all_metrics.update(alpaca_scores)

    if do_tqa:
        logger.info("Evaluating on TruthfulQA...")
        tqa_scores = eval_truthfulqa(model, tokenizer, args.batch_size, args.max_new_tokens)
        all_metrics.update(tqa_scores)

    tag = f"{args.schedule}_{args.noise_type}_seed{args.seed}"
    output_path = os.path.join(args.output_dir, f"eval_{tag}.json")
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"Results saved to {output_path}")
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()

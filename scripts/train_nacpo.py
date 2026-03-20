#!/usr/bin/env python3
"""
NaCPO Training: DPO with noise injection.
4 noise schedules x 3 noise types = 12 configurations + 1 baseline.
Model: Qwen/Qwen3.5-9B. Data: UltraFeedback. Uses TRL DPOTrainer.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.noise_curriculum import (
    build_schedule, build_noise_injector, NoisyCurriculumCollator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_nacpo")


def parse_args():
    parser = argparse.ArgumentParser(description="NaCPO: Noise-as-Curriculum DPO Training")
    parser.add_argument("--config", type=str, default="configs/nacpo_configs.yaml")
    parser.add_argument("--schedule", type=str, required=True,
                        choices=["uniform", "ascending", "descending", "adversarial", "none"])
    parser.add_argument("--noise_type", type=str, required=True,
                        choices=["random_flip", "confidence_weighted", "semantic_swap", "none"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def prepare_ultrafeedback(dataset, max_samples=None):
    """Convert UltraFeedback to DPO format with prompt/chosen/rejected."""

    def process_example(example):
        prompt = example.get("instruction", example.get("prompt", ""))
        completions = example.get("completions", [])

        if len(completions) >= 2:
            sorted_completions = sorted(
                completions,
                key=lambda x: x.get("overall_score", x.get("score", 0)),
                reverse=True,
            )
            chosen = sorted_completions[0].get("response", "")
            rejected = sorted_completions[-1].get("response", "")
        elif "chosen" in example and "rejected" in example:
            chosen = example["chosen"]
            rejected = example["rejected"]
        else:
            return None

        if not chosen or not rejected or chosen == rejected:
            return None

        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    processed = []
    for ex in dataset:
        result = process_example(ex)
        if result:
            processed.append(result)
        if max_samples and len(processed) >= max_samples:
            break

    from datasets import Dataset as HFDataset
    return HFDataset.from_list(processed)


class NoiseLoggingCallback(TrainerCallback):
    """Log noise injection statistics during training."""

    def __init__(self, collator: NoisyCurriculumCollator):
        self.collator = collator

    def on_step_end(self, args, state, control, **kwargs):
        self.collator.step()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        logs["noise/current_rate"] = self.collator.current_noise_rate
        logs["noise/observed_flip_rate"] = self.collator.observed_flip_rate
        logs["noise/progress"] = self.collator.progress


def main():
    args = parse_args()
    cfg = load_config(args.config)

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    is_baseline = args.schedule == "none" and args.noise_type == "none"
    tag = f"{'baseline' if is_baseline else f'{args.schedule}_{args.noise_type}'}_seed{args.seed}"
    output_dir = args.output_dir or os.path.join(cfg["output"]["checkpoint_dir"], tag)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"NaCPO Training: schedule={args.schedule}, noise_type={args.noise_type}")
    logger.info(f"Output: {output_dir}")

    model_name = cfg["model"]["name"]
    logger.info(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    logger.info("Loading UltraFeedback dataset...")
    max_samples = args.max_train_samples or cfg["dataset"].get("max_train_samples")
    raw_dataset = load_dataset(cfg["dataset"]["name"], split=cfg["dataset"]["split"])
    dataset = prepare_ultrafeedback(raw_dataset, max_samples=max_samples)
    logger.info(f"Prepared {len(dataset)} training pairs")

    # Build noise components
    collator = None
    callbacks = []

    if not is_baseline:
        schedule_cfg = cfg["noise_schedules"][args.schedule]
        noise_cfg = cfg["noise_types"][args.noise_type]
        schedule = build_schedule(schedule_cfg)
        injector = build_noise_injector(noise_cfg)

        if args.noise_type == "semantic_swap":
            logger.info("Building semantic swap response pool...")
            response_pool = [ex["chosen"] for ex in dataset] + [ex["rejected"] for ex in dataset]
            injector.build_pool(response_pool[:5000])

        collator = NoisyCurriculumCollator(
            tokenizer=tokenizer,
            schedule=schedule,
            injector=injector,
            max_length=cfg["training"]["max_length"],
            max_prompt_length=cfg["training"]["max_prompt_length"],
            seed=args.seed,
        )
        callbacks.append(NoiseLoggingCallback(collator))

    tcfg = cfg["training"]
    training_config = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        num_train_epochs=args.num_train_epochs or tcfg["num_train_epochs"],
        learning_rate=args.learning_rate or tcfg["learning_rate"],
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg["weight_decay"],
        max_grad_norm=tcfg["max_grad_norm"],
        bf16=tcfg["bf16"],
        logging_steps=tcfg["logging_steps"],
        save_steps=tcfg["save_steps"],
        beta=tcfg["beta"],
        loss_type=tcfg["loss_type"],
        max_length=tcfg["max_length"],
        max_prompt_length=tcfg["max_prompt_length"],
        seed=args.seed,
        report_to="none",
        deepspeed=None,
    )

    if collator:
        total_steps = (
            len(dataset) // (tcfg["per_device_train_batch_size"] * tcfg["gradient_accumulation_steps"])
            * (args.num_train_epochs or tcfg["num_train_epochs"])
        )
        collator.set_training_steps(total_steps)

        # Inject noise into dataset before training
        logger.info("Pre-injecting noise into dataset (collator will track progress)...")
        noisy_data = []
        import random
        rng = random.Random(args.seed)
        for i, ex in enumerate(dataset):
            progress = i / len(dataset)
            noise_rate = schedule.get_rate(progress)
            chosen, rejected, flipped = injector.inject(
                ex["chosen"], ex["rejected"], noise_rate, rng,
            )
            noisy_data.append({
                "prompt": ex["prompt"],
                "chosen": chosen,
                "rejected": rejected,
            })

        from datasets import Dataset as HFDataset
        dataset = HFDataset.from_list(noisy_data)
        logger.info(f"Noise injection complete. Flip rate: {collator.observed_flip_rate:.4f}")

    trainer = DPOTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    logger.info("Starting DPO training...")
    train_result = trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = {
        "schedule": args.schedule,
        "noise_type": args.noise_type,
        "seed": args.seed,
        "is_baseline": is_baseline,
        "train_loss": train_result.metrics.get("train_loss"),
        "train_runtime": train_result.metrics.get("train_runtime"),
    }

    with open(os.path.join(output_dir, "train_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if trainer.state.log_history:
        with open(os.path.join(output_dir, "step_logs.json"), "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)

    logger.info(f"Training complete. Model saved to {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
NaCPO Training: DPO with noise-as-curriculum injection.

Base model: Qwen/Qwen3.5-9B
Dataset: UltraFeedback (or Anthropic HH) preferences
Uses TRL DPOTrainer with custom NoisyCurriculumCollator.
Accepts CLI args: noise_type, noise_schedule, noise_rate, warmup_steps.
Saves checkpoints + training metrics.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset as HFDataset
from datasets import load_dataset, load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.noise_curriculum import (
    build_schedule, build_noise_injector, NoisyCurriculumCollator,
    NoiseSchedule, UniformSchedule, AscendingSchedule, DescendingSchedule,
    CosineSchedule, CyclicSchedule, AdversarialSchedule,
)
from src.qwen35_compat import (
    apply_qwen35_text_only_patch, patch_model_instance, ClearRopeDeltasCallback,
)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
apply_qwen35_text_only_patch()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_nacpo")


def parse_args():
    parser = argparse.ArgumentParser(description="NaCPO: Noise-as-Curriculum DPO Training")
    parser.add_argument("--config", type=str, default="configs/nacpo_configs.yaml")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Override model name (default: from config)")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Override dataset (default: from config)")

    parser.add_argument("--noise_type", type=str, required=True,
                        choices=["random_flip", "confidence_weighted", "semantic_swap", "none"])
    parser.add_argument("--noise_schedule", type=str, required=True,
                        choices=["uniform", "ascending", "descending", "cosine", "cyclic", "adversarial", "none"])
    parser.add_argument("--noise_rate", type=float, default=None,
                        help="Base noise rate (overrides config)")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Steps before noise injection begins")

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None, help="DPO beta parameter")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="DeepSpeed config JSON path for multi-GPU training")
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def prepare_preference_data(dataset_name, split, max_samples=None, tokenizer=None,
                            local_path=None):
    """Load and prepare preference pairs from UltraFeedback or Anthropic HH.

    Returns data in the conversational format expected by TRL DPOTrainer:
    each row has 'prompt', 'chosen', 'rejected' as chat-formatted lists
    when a tokenizer with chat_template is available, or as plain strings.

    If local_path is set, loads pre-saved Arrow dataset from disk (no network,
    no FileLock). Otherwise falls back to load_dataset() from HF Hub/cache.
    Pre-save on CPU server:
        ds = load_dataset('openbmb/UltraFeedback', split='train')
        ds.save_to_disk('/path/to/ultrafeedback_train')
    """
    if local_path and Path(local_path).exists():
        logger.info(f"Loading dataset from disk: {local_path}")
        raw = load_from_disk(local_path)
    else:
        logger.info(f"Loading dataset from HF: {dataset_name} split={split}")
        raw = load_dataset(dataset_name, split=split)

    use_chat = (
        tokenizer is not None
        and hasattr(tokenizer, "chat_template")
        and tokenizer.chat_template is not None
    )

    processed = []
    for ex in raw:
        prompt = ex.get("instruction", ex.get("prompt", ""))
        completions = ex.get("completions", [])

        if len(completions) >= 2:
            sorted_c = sorted(
                completions,
                key=lambda x: x.get("overall_score", x.get("score", 0)),
                reverse=True,
            )
            chosen = sorted_c[0].get("response", "")
            rejected = sorted_c[-1].get("response", "")
        elif "chosen" in ex and "rejected" in ex:
            chosen = ex["chosen"]
            rejected = ex["rejected"]
            if not prompt:
                prompt = ex.get("prompt", ex.get("question", ""))
        else:
            continue

        if not chosen or not rejected or chosen == rejected:
            continue
        if not prompt:
            continue

        if use_chat:
            prompt_msgs = [{"role": "user", "content": prompt}]
            chosen_msgs = [{"role": "user", "content": prompt},
                           {"role": "assistant", "content": chosen}]
            rejected_msgs = [{"role": "user", "content": prompt},
                             {"role": "assistant", "content": rejected}]
            processed.append({
                "prompt": prompt_msgs,
                "chosen": chosen_msgs,
                "rejected": rejected_msgs,
            })
        else:
            processed.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

        if max_samples and len(processed) >= max_samples:
            break

    logger.info(f"Prepared {len(processed)} preference pairs (chat_format={use_chat})")
    return HFDataset.from_list(processed)


class NaCPOCallback(TrainerCallback):
    """Log noise injection statistics and handle warmup."""

    def __init__(self, collator: NoisyCurriculumCollator, warmup_steps: int = 0):
        self.collator = collator
        self.warmup_steps = warmup_steps
        self.global_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        self.global_step += 1
        self.collator.step()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        logs["noise/current_rate"] = self.collator.current_noise_rate
        logs["noise/observed_flip_rate"] = self.collator.observed_flip_rate
        logs["noise/progress"] = self.collator.progress
        logs["noise/warmup_active"] = int(self.global_step < self.warmup_steps)


def inject_noise_into_dataset(dataset, schedule, injector, noise_type, seed, warmup_fraction=0.0):
    """Pre-inject noise into the full dataset according to schedule."""
    rng = random.Random(seed)
    noisy_data = []
    n_flipped = 0

    for i, ex in enumerate(dataset):
        progress = i / max(len(dataset), 1)

        if progress < warmup_fraction:
            noisy_data.append(ex)
            continue

        adjusted_progress = (progress - warmup_fraction) / max(1.0 - warmup_fraction, 1e-8)
        noise_rate = schedule.get_rate(min(adjusted_progress, 1.0))

        chosen, rejected, flipped = injector.inject(
            ex["chosen"], ex["rejected"], noise_rate, rng,
        )
        if flipped:
            n_flipped += 1

        noisy_data.append({
            "prompt": ex["prompt"],
            "chosen": chosen,
            "rejected": rejected,
        })

    flip_rate = n_flipped / max(len(dataset), 1)
    logger.info(f"Noise injection: {n_flipped}/{len(dataset)} flipped ({flip_rate:.2%})")
    return HFDataset.from_list(noisy_data), flip_rate


def main():
    args = parse_args()
    cfg = load_config(args.config)

    is_baseline = args.noise_schedule == "none" and args.noise_type == "none"
    tag = f"{'baseline' if is_baseline else f'{args.noise_schedule}_{args.noise_type}'}"
    if args.noise_rate is not None:
        tag += f"_nr{args.noise_rate}"
    tag += f"_seed{args.seed}"

    output_dir = args.output_dir or os.path.join(cfg["output"]["checkpoint_dir"], tag)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"NaCPO Training: schedule={args.noise_schedule}, type={args.noise_type}, "
                f"rate={args.noise_rate}, warmup={args.warmup_steps}")
    logger.info(f"Output: {output_dir}")

    model_name = args.model_name or cfg["model"]["name"]
    logger.info(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, trust_remote_code=True,
    )
    patch_model_instance(model)

    dataset_name = args.dataset_name or cfg["dataset"]["name"]
    local_path = cfg["dataset"].get("local_path")
    max_samples = args.max_train_samples or cfg["dataset"].get("max_train_samples")
    dataset = prepare_preference_data(
        dataset_name, cfg["dataset"]["split"], max_samples, tokenizer,
        local_path=local_path,
    )

    tcfg = cfg["training"]
    callbacks = []
    actual_flip_rate = 0.0

    if not is_baseline:
        sched_key = args.noise_schedule
        if sched_key in cfg["noise_schedules"]:
            schedule_cfg = dict(cfg["noise_schedules"][sched_key])
        else:
            schedule_cfg = {"type": sched_key}
        if args.noise_rate is not None:
            if sched_key == "uniform":
                schedule_cfg["noise_rate"] = args.noise_rate
            elif sched_key in ("ascending", "descending"):
                schedule_cfg["end_rate"] = args.noise_rate
            elif sched_key in ("cosine", "cyclic"):
                schedule_cfg["peak_rate"] = args.noise_rate
            elif sched_key == "adversarial":
                schedule_cfg["adversarial_rate"] = args.noise_rate

        schedule = build_schedule(schedule_cfg)
        noise_cfg = dict(cfg["noise_types"][args.noise_type])
        injector = build_noise_injector(noise_cfg)

        if args.noise_type == "semantic_swap":
            logger.info("Building semantic swap response pool...")
            pool = [ex["chosen"] for ex in dataset] + [ex["rejected"] for ex in dataset]
            injector.build_pool(pool[:5000])

        warmup_fraction = args.warmup_steps / max(len(dataset), 1) if args.warmup_steps > 0 else 0.0

        dataset, actual_flip_rate = inject_noise_into_dataset(
            dataset, schedule, injector, args.noise_type, args.seed, warmup_fraction,
        )

        collator = NoisyCurriculumCollator(
            tokenizer=tokenizer, schedule=schedule, injector=injector,
            max_length=tcfg["max_length"], max_prompt_length=tcfg["max_prompt_length"],
            seed=args.seed,
        )
        total_steps = (
            len(dataset)
            // ((args.per_device_train_batch_size or tcfg["per_device_train_batch_size"])
                * (args.gradient_accumulation_steps or tcfg["gradient_accumulation_steps"]))
            * (args.num_train_epochs or tcfg["num_train_epochs"])
        )
        collator.set_training_steps(total_steps)
        callbacks.append(NaCPOCallback(collator, args.warmup_steps))

    dpo_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size or tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=args.gradient_accumulation_steps or tcfg["gradient_accumulation_steps"],
        num_train_epochs=args.num_train_epochs or tcfg["num_train_epochs"],
        learning_rate=args.learning_rate or tcfg.get("learning_rate_lora", tcfg["learning_rate"]),
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg["weight_decay"],
        max_grad_norm=tcfg["max_grad_norm"],
        bf16=tcfg["bf16"],
        gradient_checkpointing=True,
        logging_steps=tcfg["logging_steps"],
        save_steps=tcfg["save_steps"],
        beta=args.beta or tcfg["beta"],
        loss_type=tcfg["loss_type"],
        max_length=tcfg["max_length"],
        seed=args.seed,
        report_to="none",
    )
    if args.deepspeed:
        dpo_kwargs["deepspeed"] = args.deepspeed
    training_config = DPOConfig(**dpo_kwargs)

    lora_cfg = cfg.get("lora", {})
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("lora_alpha", 128),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )
    logger.info("Using LoRA: r=%d, alpha=%d", peft_config.r, peft_config.lora_alpha)

    callbacks.append(ClearRopeDeltasCallback())

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )

    logger.info("Starting DPO training...")
    train_result = trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = {
        "noise_schedule": args.noise_schedule,
        "noise_type": args.noise_type,
        "noise_rate": args.noise_rate,
        "warmup_steps": args.warmup_steps,
        "seed": args.seed,
        "is_baseline": is_baseline,
        "actual_flip_rate": actual_flip_rate,
        "train_loss": train_result.metrics.get("train_loss"),
        "train_runtime": train_result.metrics.get("train_runtime"),
        "model": model_name,
        "dataset": dataset_name,
        "num_samples": len(dataset),
    }

    with open(os.path.join(output_dir, "train_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if trainer.state.log_history:
        with open(os.path.join(output_dir, "step_logs.json"), "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)

    logger.info(f"Training complete. Model saved to {output_dir}")


if __name__ == "__main__":
    main()

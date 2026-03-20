"""
Noise injection module for NaCPO (Noise-as-Curriculum Preference Optimization).

Three noise types:
  1. random_flip: Randomly swap chosen/rejected labels
  2. confidence_weighted: Flip labels weighted by model confidence margin
  3. semantic_swap: Swap with semantically similar alternatives using embedding distance

Four noise schedules:
  1. uniform: Constant noise rate throughout training
  2. ascending: Linearly increase noise over training
  3. descending: Linearly decrease noise over training
  4. adversarial: Low noise + periodic high-noise bursts
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── Noise Schedules ──────────────────────────────────────────────────────────

class NoiseSchedule:
    """Base class for noise rate schedules."""

    def get_rate(self, progress: float) -> float:
        """Return noise rate for given training progress in [0, 1]."""
        raise NotImplementedError


class UniformSchedule(NoiseSchedule):
    def __init__(self, noise_rate: float = 0.15):
        self.noise_rate = noise_rate

    def get_rate(self, progress: float) -> float:
        return self.noise_rate


class AscendingSchedule(NoiseSchedule):
    def __init__(self, start_rate: float = 0.0, end_rate: float = 0.3):
        self.start_rate = start_rate
        self.end_rate = end_rate

    def get_rate(self, progress: float) -> float:
        return self.start_rate + (self.end_rate - self.start_rate) * progress


class DescendingSchedule(NoiseSchedule):
    def __init__(self, start_rate: float = 0.3, end_rate: float = 0.0):
        self.start_rate = start_rate
        self.end_rate = end_rate

    def get_rate(self, progress: float) -> float:
        return self.start_rate + (self.end_rate - self.start_rate) * progress


class AdversarialSchedule(NoiseSchedule):
    """Low base noise with periodic high-noise adversarial bursts."""

    def __init__(self, base_rate: float = 0.1, adversarial_rate: float = 0.4,
                 adversarial_fraction: float = 0.2):
        self.base_rate = base_rate
        self.adversarial_rate = adversarial_rate
        self.adversarial_fraction = adversarial_fraction

    def get_rate(self, progress: float) -> float:
        cycle = progress * 10  # 10 cycles over training
        phase = cycle - int(cycle)
        if phase < self.adversarial_fraction:
            return self.adversarial_rate
        return self.base_rate


def build_schedule(config: dict) -> NoiseSchedule:
    stype = config["type"]
    if stype == "uniform":
        return UniformSchedule(config.get("noise_rate", 0.15))
    elif stype == "ascending":
        return AscendingSchedule(config.get("start_rate", 0.0), config.get("end_rate", 0.3))
    elif stype == "descending":
        return DescendingSchedule(config.get("start_rate", 0.3), config.get("end_rate", 0.0))
    elif stype == "adversarial":
        return AdversarialSchedule(
            config.get("base_rate", 0.1),
            config.get("adversarial_rate", 0.4),
            config.get("adversarial_fraction", 0.2),
        )
    raise ValueError(f"Unknown schedule type: {stype}")


# ── Noise Types ──────────────────────────────────────────────────────────────

class NoiseInjector:
    """Base class for noise injection."""

    def inject(self, chosen: str, rejected: str, noise_rate: float,
               rng: random.Random, **kwargs) -> tuple[str, str, bool]:
        """
        Returns (chosen, rejected, was_flipped).
        May modify the content or swap labels.
        """
        raise NotImplementedError


class RandomFlipInjector(NoiseInjector):
    """Simply swap chosen/rejected labels with probability = noise_rate."""

    def inject(self, chosen, rejected, noise_rate, rng, **kwargs):
        if rng.random() < noise_rate:
            return rejected, chosen, True
        return chosen, rejected, False


class ConfidenceWeightedInjector(NoiseInjector):
    """
    Flip labels weighted by inverse confidence margin.
    Low-margin pairs (close quality) are more likely to be flipped.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self._model = None
        self._tokenizer = None

    def set_model(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer

    def _estimate_confidence(self, chosen: str, rejected: str) -> float:
        """Estimate confidence margin from text length/quality heuristics."""
        len_diff = abs(len(chosen) - len(rejected))
        max_len = max(len(chosen), len(rejected), 1)
        margin = len_diff / max_len

        # Use unique-word overlap as quality proxy
        c_words = set(chosen.lower().split())
        r_words = set(rejected.lower().split())
        if c_words | r_words:
            overlap = len(c_words & r_words) / len(c_words | r_words)
        else:
            overlap = 1.0

        margin = margin * (1 - overlap)
        return margin

    def inject(self, chosen, rejected, noise_rate, rng, **kwargs):
        margin = self._estimate_confidence(chosen, rejected)
        flip_prob = noise_rate * np.exp(-margin / self.temperature)
        flip_prob = min(flip_prob, 1.0)

        if rng.random() < flip_prob:
            return rejected, chosen, True
        return chosen, rejected, False


class SemanticSwapInjector(NoiseInjector):
    """
    Replace rejected with a semantically similar but different response.
    Uses embedding similarity to find near-misses, creating harder negatives.
    """

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 top_k: int = 5):
        self.embedding_model_name = embedding_model
        self.top_k = top_k
        self._encoder = None
        self._response_pool = []
        self._embeddings = None

    def build_pool(self, responses: list[str]):
        """Pre-compute embeddings for a pool of responses."""
        self._response_pool = responses
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.embedding_model_name)
            except ImportError:
                logger.warning("sentence-transformers not installed, using random swap fallback")
                self._encoder = None
                return

        self._embeddings = self._encoder.encode(
            responses, convert_to_tensor=True, show_progress_bar=False,
        )

    def _find_similar(self, text: str) -> Optional[str]:
        if self._encoder is None or self._embeddings is None:
            return None
        emb = self._encoder.encode([text], convert_to_tensor=True)
        sims = F.cosine_similarity(emb, self._embeddings)
        # Exclude exact match (sim~1.0), pick from top-k similar
        top_k_idx = sims.topk(min(self.top_k + 1, len(sims))).indices
        for idx in top_k_idx:
            candidate = self._response_pool[idx.item()]
            if candidate != text:
                return candidate
        return None

    def inject(self, chosen, rejected, noise_rate, rng, **kwargs):
        if rng.random() >= noise_rate:
            return chosen, rejected, False

        similar = self._find_similar(rejected)
        if similar is not None:
            return chosen, similar, True

        # Fallback: just flip
        return rejected, chosen, True


def build_noise_injector(config: dict) -> NoiseInjector:
    ntype = config["type"]
    if ntype == "random_flip":
        return RandomFlipInjector()
    elif ntype == "confidence_weighted":
        return ConfidenceWeightedInjector(config.get("temperature", 1.0))
    elif ntype == "semantic_swap":
        return SemanticSwapInjector(
            config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            config.get("top_k_similar", 5),
        )
    raise ValueError(f"Unknown noise type: {ntype}")


# ── Noisy Data Collator ─────────────────────────────────────────────────────

class NoisyCurriculumCollator:
    """
    Custom data collator for DPO training that injects noise according to
    the specified schedule and noise type.
    """

    def __init__(
        self,
        tokenizer,
        schedule: NoiseSchedule,
        injector: NoiseInjector,
        max_length: int = 2048,
        max_prompt_length: int = 1024,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.schedule = schedule
        self.injector = injector
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.rng = random.Random(seed)
        self.total_steps = 1
        self.current_step = 0
        self.flip_count = 0
        self.total_count = 0

    def set_training_steps(self, total_steps: int):
        self.total_steps = max(total_steps, 1)

    def step(self):
        self.current_step += 1

    @property
    def progress(self) -> float:
        return self.current_step / self.total_steps

    @property
    def current_noise_rate(self) -> float:
        return self.schedule.get_rate(self.progress)

    @property
    def observed_flip_rate(self) -> float:
        return self.flip_count / max(self.total_count, 1)

    def __call__(self, features: list[dict]) -> dict:
        noise_rate = self.current_noise_rate

        processed = []
        for feat in features:
            chosen = feat.get("chosen", "")
            rejected = feat.get("rejected", "")
            prompt = feat.get("prompt", "")

            noisy_chosen, noisy_rejected, flipped = self.injector.inject(
                chosen, rejected, noise_rate, self.rng,
            )
            self.total_count += 1
            if flipped:
                self.flip_count += 1

            processed.append({
                "prompt": prompt,
                "chosen": noisy_chosen,
                "rejected": noisy_rejected,
            })

        return processed

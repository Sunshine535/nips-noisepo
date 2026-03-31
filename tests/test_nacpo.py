#!/usr/bin/env python3
"""Tests for NaCPO core components: noise schedules, injectors, collator, data formatting."""

import math
import random
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import datasets as _ds_lib
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
from src.noise_curriculum import (
    UniformSchedule,
    AscendingSchedule,
    DescendingSchedule,
    CosineSchedule,
    CyclicSchedule,
    AdversarialSchedule,
    RandomFlipInjector,
    ConfidenceWeightedInjector,
    NoisyCurriculumCollator,
    build_schedule,
    build_noise_injector,
)


# ── Noise Schedule Tests ─────────────────────────────────────────────────────


class TestUniformSchedule:
    def test_constant_rate(self):
        s = UniformSchedule(noise_rate=0.15)
        for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert s.get_rate(p) == pytest.approx(0.15)

    def test_zero_rate(self):
        s = UniformSchedule(noise_rate=0.0)
        assert s.get_rate(0.5) == 0.0


class TestAscendingSchedule:
    def test_linear_increase(self):
        s = AscendingSchedule(start_rate=0.0, end_rate=0.3)
        assert s.get_rate(0.0) == pytest.approx(0.0)
        assert s.get_rate(0.5) == pytest.approx(0.15)
        assert s.get_rate(1.0) == pytest.approx(0.3)

    def test_monotonic(self):
        s = AscendingSchedule(start_rate=0.0, end_rate=0.3)
        prev = -1.0
        for i in range(101):
            r = s.get_rate(i / 100)
            assert r >= prev
            prev = r


class TestDescendingSchedule:
    def test_linear_decrease(self):
        s = DescendingSchedule(start_rate=0.3, end_rate=0.0)
        assert s.get_rate(0.0) == pytest.approx(0.3)
        assert s.get_rate(0.5) == pytest.approx(0.15)
        assert s.get_rate(1.0) == pytest.approx(0.0)

    def test_monotonic(self):
        s = DescendingSchedule(start_rate=0.3, end_rate=0.0)
        prev = 1.0
        for i in range(101):
            r = s.get_rate(i / 100)
            assert r <= prev
            prev = r


class TestCosineSchedule:
    def test_starts_at_peak(self):
        s = CosineSchedule(peak_rate=0.3)
        assert s.get_rate(0.0) == pytest.approx(0.3)

    def test_ends_near_zero(self):
        s = CosineSchedule(peak_rate=0.3)
        assert s.get_rate(1.0) == pytest.approx(0.0, abs=1e-10)

    def test_midpoint(self):
        s = CosineSchedule(peak_rate=0.3)
        assert s.get_rate(0.5) == pytest.approx(0.15)

    def test_bounded(self):
        s = CosineSchedule(peak_rate=0.3)
        for i in range(101):
            r = s.get_rate(i / 100)
            assert 0.0 <= r <= 0.3 + 1e-10


class TestCyclicSchedule:
    def test_bounded(self):
        s = CyclicSchedule(peak_rate=0.2, num_cycles=5)
        for i in range(101):
            r = s.get_rate(i / 100)
            assert -1e-10 <= r <= 0.2 + 1e-10

    def test_oscillates(self):
        s = CyclicSchedule(peak_rate=0.2, num_cycles=5)
        rates = [s.get_rate(i / 1000) for i in range(1001)]
        ups = sum(1 for i in range(1, len(rates)) if rates[i] > rates[i - 1])
        downs = sum(1 for i in range(1, len(rates)) if rates[i] < rates[i - 1])
        assert ups > 0 and downs > 0


class TestAdversarialSchedule:
    def test_has_bursts(self):
        s = AdversarialSchedule(base_rate=0.1, adversarial_rate=0.4, adversarial_fraction=0.2)
        rates = [s.get_rate(i / 1000) for i in range(1001)]
        assert any(r == pytest.approx(0.4) for r in rates)
        assert any(r == pytest.approx(0.1) for r in rates)


class TestBuildSchedule:
    def test_all_types(self):
        configs = [
            ({"type": "uniform", "noise_rate": 0.1}, UniformSchedule),
            ({"type": "ascending", "start_rate": 0.0, "end_rate": 0.3}, AscendingSchedule),
            ({"type": "descending"}, DescendingSchedule),
            ({"type": "cosine", "peak_rate": 0.5}, CosineSchedule),
            ({"type": "cyclic", "peak_rate": 0.2}, CyclicSchedule),
            ({"type": "adversarial"}, AdversarialSchedule),
        ]
        for cfg, expected_type in configs:
            s = build_schedule(cfg)
            assert isinstance(s, expected_type)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown schedule type"):
            build_schedule({"type": "nonexistent"})


# ── Noise Injector Tests ─────────────────────────────────────────────────────


class TestRandomFlipInjector:
    def test_zero_rate_never_flips(self):
        inj = RandomFlipInjector()
        rng = random.Random(42)
        for _ in range(100):
            c, r, flipped = inj.inject("A", "B", 0.0, rng)
            assert c == "A" and r == "B" and not flipped

    def test_one_rate_always_flips(self):
        inj = RandomFlipInjector()
        rng = random.Random(42)
        for _ in range(100):
            c, r, flipped = inj.inject("A", "B", 1.0, rng)
            assert c == "B" and r == "A" and flipped

    def test_stochastic_rate(self):
        inj = RandomFlipInjector()
        rng = random.Random(42)
        flips = sum(inj.inject("A", "B", 0.5, rng)[2] for _ in range(1000))
        assert 400 < flips < 600


class TestConfidenceWeightedInjector:
    def test_never_exceeds_rate(self):
        inj = ConfidenceWeightedInjector(temperature=1.0)
        rng = random.Random(42)
        flips = sum(inj.inject("short", "a very long response", 0.1, rng)[2]
                    for _ in range(500))
        assert flips / 500 <= 0.15


class TestBuildNoiseInjector:
    def test_random_flip(self):
        inj = build_noise_injector({"type": "random_flip"})
        assert isinstance(inj, RandomFlipInjector)

    def test_confidence_weighted(self):
        inj = build_noise_injector({"type": "confidence_weighted", "temperature": 2.0})
        assert isinstance(inj, ConfidenceWeightedInjector)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown noise type"):
            build_noise_injector({"type": "nonexistent"})


# ── NoisyCurriculumCollator Tests ─────────────────────────────────────────────


class TestNoisyCurriculumCollator:
    @staticmethod
    def _make_fake_tokenizer():
        tok = MagicMock()
        tok.pad_token_id = 0
        tok.eos_token_id = 0
        tok.chat_template = None
        return tok

    def test_noise_rate_1_always_flips(self):
        collator = NoisyCurriculumCollator(
            tokenizer=self._make_fake_tokenizer(),
            schedule=UniformSchedule(noise_rate=1.0),
            injector=RandomFlipInjector(),
            seed=42,
        )
        collator.set_training_steps(100)
        features = [{"prompt": "Q", "chosen": "A", "rejected": "B"}]
        result = collator(features)
        assert result[0]["chosen"] == "B"
        assert result[0]["rejected"] == "A"

    def test_noise_rate_0_never_flips(self):
        collator = NoisyCurriculumCollator(
            tokenizer=self._make_fake_tokenizer(),
            schedule=UniformSchedule(noise_rate=0.0),
            injector=RandomFlipInjector(),
            seed=42,
        )
        collator.set_training_steps(100)
        features = [{"prompt": "Q", "chosen": "A", "rejected": "B"}]
        result = collator(features)
        assert result[0]["chosen"] == "A"
        assert result[0]["rejected"] == "B"

    def test_step_advances_progress(self):
        collator = NoisyCurriculumCollator(
            tokenizer=self._make_fake_tokenizer(),
            schedule=AscendingSchedule(start_rate=0.0, end_rate=1.0),
            injector=RandomFlipInjector(),
        )
        collator.set_training_steps(100)
        assert collator.progress == pytest.approx(0.0)
        for _ in range(50):
            collator.step()
        assert collator.progress == pytest.approx(0.5)

    def test_observed_flip_rate_tracking(self):
        collator = NoisyCurriculumCollator(
            tokenizer=self._make_fake_tokenizer(),
            schedule=UniformSchedule(noise_rate=1.0),
            injector=RandomFlipInjector(),
            seed=42,
        )
        collator.set_training_steps(100)
        for _ in range(10):
            collator([{"prompt": "Q", "chosen": "A", "rejected": "B"}])
        assert collator.observed_flip_rate == pytest.approx(1.0)

    def test_prompt_preserved(self):
        collator = NoisyCurriculumCollator(
            tokenizer=self._make_fake_tokenizer(),
            schedule=UniformSchedule(noise_rate=1.0),
            injector=RandomFlipInjector(),
            seed=42,
        )
        features = [{"prompt": "my question", "chosen": "A", "rejected": "B"}]
        result = collator(features)
        assert result[0]["prompt"] == "my question"


# ── DPO Data Formatting Tests ────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_DATASETS, reason="datasets library not installed")
class TestDPODataFormatting:
    """Verify that prepare_preference_data produces correct TRL DPO format."""

    def test_non_chat_format_has_separate_fields(self):
        """In non-chat mode, chosen/rejected should be plain strings without the prompt."""
        from scripts.train_nacpo import prepare_preference_data

        mock_dataset = [
            {
                "instruction": "What is 2+2?",
                "completions": [
                    {"response": "4", "overall_score": 5},
                    {"response": "5", "overall_score": 1},
                ],
            }
        ]

        from unittest.mock import patch
        from datasets import Dataset

        mock_ds = Dataset.from_list(mock_dataset)
        with patch("scripts.train_nacpo.load_dataset", return_value=mock_ds):
            result = prepare_preference_data("dummy", "train", tokenizer=None)

        assert len(result) == 1
        row = result[0]
        assert row["prompt"] == "What is 2+2?"
        assert row["chosen"] == "4"
        assert row["rejected"] == "5"

    def test_chat_format_separates_prompt_and_completion(self):
        """In chat mode, chosen/rejected should only contain assistant messages."""
        from scripts.train_nacpo import prepare_preference_data

        mock_dataset = [
            {
                "instruction": "Explain gravity.",
                "completions": [
                    {"response": "Gravity is a force.", "overall_score": 5},
                    {"response": "I don't know.", "overall_score": 1},
                ],
            }
        ]

        from unittest.mock import patch
        from datasets import Dataset

        mock_ds = Dataset.from_list(mock_dataset)

        fake_tokenizer = MagicMock()
        fake_tokenizer.chat_template = "some_template"

        with patch("scripts.train_nacpo.load_dataset", return_value=mock_ds):
            result = prepare_preference_data("dummy", "train", tokenizer=fake_tokenizer)

        assert len(result) == 1
        row = result[0]
        assert row["prompt"] == [{"role": "user", "content": "Explain gravity."}]
        assert row["chosen"] == [{"role": "assistant", "content": "Gravity is a force."}]
        assert row["rejected"] == [{"role": "assistant", "content": "I don't know."}]
        assert "user" not in str(row["chosen"])

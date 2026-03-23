"""
Qwen3.5 compatibility patch for TRL DPOTrainer (text-only training).

Qwen3.5 uses 3D position encoding (compute_3d_position_ids) designed for
multimodal inputs (images/videos). When TRL's DPOTrainer processes text-only
inputs, the model carries stale rope_deltas causing a RuntimeError shape mismatch.

Three-level defense:
  1. Class-level monkey-patch: always clear rope_deltas, catch RuntimeError, fall
     back to 1D positions.
  2. Instance-level forward hook: clear rope_deltas before every forward pass.
  3. TrainerCallback: reset model state between steps.

Usage:
    from src.qwen35_compat import apply_qwen35_text_only_patch, patch_model_instance
    from src.qwen35_compat import ClearRopeDeltasCallback

    apply_qwen35_text_only_patch()           # call once at import time
    patch_model_instance(model)              # call after model creation
    trainer = DPOTrainer(..., callbacks=[ClearRopeDeltasCallback()])
"""

import importlib
import logging

import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)

_PATCHED_CLASSES: list[type] = []


def _build_1d_position_ids(input_ids=None, inputs_embeds=None, attention_mask=None):
    if inputs_embeds is not None:
        B, L = inputs_embeds.shape[:2]
        device = inputs_embeds.device
    elif input_ids is not None:
        B, L = input_ids.shape[:2]
        device = input_ids.device
    else:
        return None

    if attention_mask is not None:
        pos = attention_mask.long().cumsum(-1) - 1
        pos.masked_fill_(attention_mask == 0, 1)
    else:
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

    return pos


def _find_model_class():
    module_names = [
        "transformers.models.qwen3_5.modeling_qwen3_5",
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
    ]
    for mod_name in module_names:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue

        best = None
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name, None)
            if not isinstance(obj, type):
                continue
            if not hasattr(obj, "compute_3d_position_ids"):
                continue
            best = obj
            if "Model" in attr_name and "ForCausal" not in attr_name:
                return obj
        if best is not None:
            return best

    return None


def apply_qwen35_text_only_patch() -> bool:
    target_cls = _find_model_class()
    if target_cls is None:
        logger.debug("[qwen35_compat] No Qwen3.5 model class found — skipping patch")
        return False

    if getattr(target_cls, "_text_only_patched", False):
        return True

    original_fn = target_cls.compute_3d_position_ids

    def _patched(self, *args, **kwargs):
        self.rope_deltas = None

        try:
            return original_fn(self, *args, **kwargs)
        except RuntimeError:
            pass

        input_ids = kwargs.get("input_ids") or (args[0] if args else None)
        inputs_embeds = kwargs.get("inputs_embeds") or (args[1] if len(args) > 1 else None)
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is None:
            for i, a in enumerate(args):
                if i > 1 and isinstance(a, torch.Tensor) and a.dtype in (torch.long, torch.bool) and a.ndim == 2:
                    attention_mask = a
                    break

        pos = _build_1d_position_ids(input_ids, inputs_embeds, attention_mask)
        if pos is None:
            raise RuntimeError(
                "[qwen35_compat] Cannot build fallback position IDs: "
                "no input_ids or inputs_embeds"
            )
        self.rope_deltas = None
        return pos

    target_cls.compute_3d_position_ids = _patched
    target_cls._text_only_patched = True
    _PATCHED_CLASSES.append(target_cls)
    print(f"[qwen35_compat] Patched {target_cls.__name__}.compute_3d_position_ids")
    return True


def patch_model_instance(model) -> bool:
    inner = model
    for attr in ("model", "transformer", "backbone"):
        candidate = getattr(model, attr, None)
        if candidate is not None and hasattr(candidate, "rope_deltas"):
            inner = candidate
            break

    if not hasattr(inner, "rope_deltas"):
        return False

    def _hook(module, args):
        module.rope_deltas = None

    inner.register_forward_pre_hook(_hook)
    logger.info("[qwen35_compat] Registered rope_deltas hook on %s", type(inner).__name__)
    return True


class ClearRopeDeltasCallback(TrainerCallback):
    _patched = False

    @staticmethod
    def _clear(model):
        if model is None:
            return
        for module in model.modules():
            if hasattr(module, "rope_deltas"):
                module.rope_deltas = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None and not self._patched:
            patch_model_instance(model)
            self._patched = True

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        self._clear(model)

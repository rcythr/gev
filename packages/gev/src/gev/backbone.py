"""Load a causal LM backbone for text-only decisions, shared by training and serving."""

from __future__ import annotations

from typing import Any

# Gemma 4 ships one checkpoint with vision and audio encoders; decisions read text only.
MULTIMODAL_MODULES = ("vision_tower", "audio_tower", "embed_vision", "embed_audio")


def load_backbone(model_id: str, *, dtype, cache_dir: str | None = None) -> Any:
    """The backbone in `dtype` on the GPU (CPU without one), text-only."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        # Multi-GPU only: on an Apple machine accelerate dispatches to MPS and
        # the process dies with SIGSEGV part-way through loading.
        device_map="auto" if torch.cuda.device_count() > 1 else None,
        cache_dir=cache_dir,
    )
    drop_multimodal_towers(model)
    if torch.cuda.is_available() and torch.cuda.device_count() == 1:
        model = model.cuda()
    return model


def drop_multimodal_towers(model) -> list[str]:
    """Remove vision and audio encoders; return the names removed.

    Text forwards never reach them, and their attention projections share the
    language model's `q_proj`/`k_proj` names, which LoRA would otherwise wrap.
    """
    inner = getattr(model, "model", None)
    dropped = []
    for name in MULTIMODAL_MODULES:
        if inner is not None and getattr(inner, name, None) is not None:
            setattr(inner, name, None)
            dropped.append(name)
    return dropped


def final_logit_softcap(model) -> float | None:
    """The cap Gemma applies as `cap * tanh(logits / cap)` inside `forward`.

    A projection that bypasses `forward` must apply it too, or serving reads
    different logits from the ones training fitted.
    """
    config = model.config
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    return getattr(text, "final_logit_softcapping", None)


def lora_target_pattern(targets: tuple[str, ...]) -> str:
    """A PEFT `target_modules` regex for the language model's projections only."""
    return rf"^(?!.*(?:vision|visual|audio)).*\.(?:{'|'.join(targets)})$"

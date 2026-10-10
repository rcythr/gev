"""Load a causal LM backbone for text-only decisions, shared by training and serving."""

from __future__ import annotations

from typing import Any


def load_backbone(model_id: str, *, dtype, cache_dir: str | None = None) -> Any:
    """The backbone in `dtype` on the GPU (CPU without one).

    Gemma 4's vision and audio encoders load with it and stay frozen: text
    requests never reach them, and `lora_target_pattern` leaves them unadapted.
    """
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
    if torch.cuda.is_available() and torch.cuda.device_count() == 1:
        model = model.cuda()
    return model


def final_logit_softcap(model) -> float | None:
    """The cap Gemma applies as `cap * tanh(logits / cap)` inside `forward`.

    A projection that bypasses `forward` must apply it too, or serving reads
    different logits from the ones training fitted.
    """
    config = model.config
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    return getattr(text, "final_logit_softcapping", None)


def lora_target_pattern(targets: tuple[str, ...]) -> str:
    """A PEFT `target_modules` regex for the language model's projections only.

    Gemma 4's vision and audio encoders reuse the names `q_proj`..`o_proj`.
    """
    return rf"^(?!.*(?:vision|visual|audio)).*\.(?:{'|'.join(targets)})$"

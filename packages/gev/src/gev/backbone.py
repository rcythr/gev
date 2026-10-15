"""Load a causal LM backbone for text-only decisions, shared by training and serving."""

from __future__ import annotations

from typing import Any


def load_backbone(
    model_id: str, *, dtype, cache_dir: str | None = None, split_across_gpus: bool = True
) -> Any:
    """The backbone in `dtype` on the GPU (CPU without one).

    With several GPUs and `split_across_gpus`, one copy is spread over them, for
    serving. Training reads logits and hidden states with indices built on the
    first parameter's device, so it loads whole onto one GPU instead.

    Gemma 4's vision and audio encoders load with it and stay frozen: text
    requests never reach them, and `lora_target_pattern` leaves them unadapted.
    """
    import torch
    from transformers import AutoModelForCausalLM

    from .train.distributed import world_size

    # Under `torchrun` every rank holds a whole copy on its own GPU. Multi-GPU only: on an
    # Apple machine accelerate dispatches to MPS and the process dies with SIGSEGV
    # part-way through loading.
    data_parallel = world_size() > 1
    split = split_across_gpus and torch.cuda.device_count() > 1 and not data_parallel
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map="auto" if split else None, cache_dir=cache_dir
    )
    if torch.cuda.is_available() and not split:
        model = model.to(f"cuda:{torch.cuda.current_device()}")
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


def set_lora_scale(model, scale: float) -> int:
    """Multiply every LoRA adapter's contribution by `scale`; return the layers scaled.

    The adapter adds `scaling * B @ A` to each weight, so this moves the model along the line
    from the frozen backbone (0) to the trained weights (1), and past it (above 1). The
    original scaling is kept, so the scale is absolute: calling twice does not compound.
    """
    layers = 0
    for module in model.modules():
        scaling = getattr(module, "scaling", None)
        if not isinstance(scaling, dict) or not hasattr(module, "lora_A"):
            continue
        if not hasattr(module, "_gev_base_scaling"):
            module._gev_base_scaling = dict(scaling)
        for name, base in module._gev_base_scaling.items():
            scaling[name] = base * scale
        layers += 1
    return layers

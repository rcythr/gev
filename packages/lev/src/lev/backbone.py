"""Load a causal LM backbone for text-only decisions, shared by training and serving."""

from __future__ import annotations

from typing import Any, Literal

Quantize = Literal["4bit"] | None

# Gemma 4 ships one checkpoint with vision and audio encoders; decisions read text only.
MULTIMODAL_MODULES = ("vision_tower", "audio_tower", "embed_vision", "embed_audio")


def load_backbone(
    model_id: str,
    *,
    dtype,
    cache_dir: str | None = None,
    quantize: Quantize = None,
    offload_per_layer_embeddings: bool = False,
) -> Any:
    """The backbone on the GPU (CPU without one), text-only.

    `quantize="4bit"` loads NF4 weights through bitsandbytes (QLoRA), CUDA only.
    `offload_per_layer_embeddings` keeps a Gemma E-model's per-layer embedding table,
    ~2.8B lookup-only parameters on E4B, in CPU memory.
    """
    import torch
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {}
    if quantize == "4bit":
        if not torch.cuda.is_available():
            raise RuntimeError("quantize='4bit' needs a CUDA GPU (bitsandbytes)")
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        kwargs["device_map"] = {"": 0}
    elif quantize is not None:
        raise ValueError(f"unknown quantize {quantize!r}; expected '4bit' or None")
    else:
        # Multi-GPU only: on an Apple machine accelerate dispatches to MPS and
        # the process dies with SIGSEGV part-way through loading.
        kwargs["device_map"] = "auto" if torch.cuda.device_count() > 1 else None

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, cache_dir=cache_dir, **kwargs
    )
    drop_multimodal_towers(model)
    if offload_per_layer_embeddings:
        offload_per_layer_table(model)
    if quantize is None and torch.cuda.is_available() and torch.cuda.device_count() == 1:
        model = model.cuda()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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


def offload_per_layer_table(model) -> bool:
    """Move the per-layer embedding table to CPU and look it up there.

    The table leaves the module tree, so a later `.cuda()` cannot bring it back
    and it is never trained or saved; only the looked-up rows reach the GPU.
    Returns False for a backbone without one.
    """
    decoder = model.get_decoder()
    table = getattr(decoder, "embed_tokens_per_layer", None)
    if table is None:
        return False
    del decoder.embed_tokens_per_layer
    table = table.to("cpu").requires_grad_(False)
    layers = decoder.config.num_hidden_layers
    width = decoder.hidden_size_per_layer_input

    def lookup(input_ids, *_):
        if input_ids is None:
            raise ValueError("an offloaded per-layer embedding table needs input_ids")
        rows = table(input_ids.to("cpu")).reshape(*input_ids.shape, layers, width)
        return rows.to(input_ids.device, non_blocking=True)

    decoder.get_per_layer_inputs = lookup
    return True


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

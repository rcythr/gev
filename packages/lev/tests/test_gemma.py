"""Gemma 4 backbone handling, on a tiny offline model with every E-model feature:
per-layer embeddings, shared KV layers, a final logit softcap and a vision tower."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
gemma4 = pytest.importorskip("transformers.models.gemma4")

from lev.backbone import (  # noqa: E402
    drop_multimodal_towers,
    final_logit_softcap,
    lora_target_pattern,
    offload_per_layer_table,
)
from lev.model import DecisionEngine, EngineConfig  # noqa: E402
from lev.prompt import CHAT_FRAMES, Style, chat_template_mismatch  # noqa: E402

ROWS = [[2, 5, 6, 7, 8, 9], [2, 5, 6], [2, 5, 6, 7]]


def tiny_gemma(vision: bool = True):
    torch.manual_seed(0)
    text = gemma4.Gemma4TextConfig(
        vocab_size=512,
        vocab_size_per_layer_input=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        global_head_dim=32,
        hidden_size_per_layer_input=8,
        num_kv_shared_layers=2,
        sliding_window=16,
        final_logit_softcapping=30.0,
        layer_types=["sliding_attention", "full_attention"] * 2,
    )
    vision_config = (
        gemma4.Gemma4VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=16,
        )
        if vision
        else None
    )
    config = gemma4.Gemma4Config(text_config=text, vision_config=vision_config)
    return gemma4.Gemma4ForConditionalGeneration(config).eval()


def full_logits(model, row):
    with torch.no_grad():
        out = model(input_ids=torch.tensor([row]), output_hidden_states=True)
    return out.logits[0, -1], out.hidden_states[-1][0, -1]


def test_engine_projection_applies_the_softcap_training_sees():
    """Training reads `forward`'s capped logits; the engine projects the decoder's
    states itself, so it must cap them too or temperatures fit the wrong scale."""
    model = tiny_gemma(vision=False)
    assert final_logit_softcap(model) == 30.0
    engine = DecisionEngine(model, SimpleNamespace(pad_token_id=0), EngineConfig())

    logits, hidden = engine._forward_single(ROWS, want_hidden=True)

    for i, row in enumerate(ROWS):
        expected_logits, expected_hidden = full_logits(model, row)
        torch.testing.assert_close(logits[i], expected_logits)
        torch.testing.assert_close(hidden[i], expected_hidden)


def test_dropping_towers_leaves_text_logits_unchanged():
    model = tiny_gemma()
    before = full_logits(model, ROWS[0])[0]

    assert drop_multimodal_towers(model) == ["vision_tower", "embed_vision"]

    assert not any("vision" in name for name, _ in model.named_parameters())
    torch.testing.assert_close(full_logits(model, ROWS[0])[0], before)


def test_lora_targets_only_the_language_model():
    """The vision tower's projections share the text model's names, and are a
    clipped-linear wrapper PEFT cannot adapt."""
    peft = pytest.importorskip("peft")
    model = tiny_gemma()
    targets = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    wrapped = peft.get_peft_model(
        model,
        peft.LoraConfig(r=4, target_modules=lora_target_pattern(targets), task_type="CAUSAL_LM"),
    )

    adapted = {n for n, _ in wrapped.named_parameters() if "lora_A" in n}
    assert adapted and all("language_model" in n for n in adapted)


def test_offloaded_per_layer_table_gives_the_same_logits_and_stays_on_cpu():
    model = tiny_gemma(vision=False)
    before = full_logits(model, ROWS[0])[0]

    assert offload_per_layer_table(model)
    model.to(torch.float64).to(torch.float32)  # a `.cuda()` stand-in: moves every tracked tensor

    assert not any("embed_tokens_per_layer" in n for n, _ in model.named_parameters())
    torch.testing.assert_close(full_logits(model, ROWS[0])[0], before)


def test_offload_is_a_no_op_without_a_per_layer_table():
    decoder = SimpleNamespace()
    assert not offload_per_layer_table(SimpleNamespace(get_decoder=lambda: decoder))


class TemplateTokenizer:
    """Renders a Gemma 4 style template that trims content, as Gemma's templates do."""

    bos_token = "<bos>"
    chat_template = "stand-in"

    def __init__(self, generation_prompt: str = "<|turn>model\n"):
        self.generation_prompt = generation_prompt

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        turns = "".join(f"<|turn>{m['role']}\n{m['content'].strip()}<turn|>\n" for m in messages)
        return self.bos_token + turns + self.generation_prompt


class TestChatTemplateCheck:
    def test_gemma_frame_matches_the_turn_template(self):
        assert chat_template_mismatch(TemplateTokenizer(), Style.GEMMA) is None

    def test_a_different_generation_prompt_is_reported_with_both_framings(self):
        tokenizer = TemplateTokenizer("<|turn>model\n<|channel>thought\n<channel|>")
        message = chat_template_mismatch(tokenizer, Style.GEMMA)
        assert message and "<|channel>thought" in message and "CHAT_FRAMES" in message

    def test_plain_and_templateless_tokenizers_are_not_checked(self):
        assert chat_template_mismatch(TemplateTokenizer(), Style.PLAIN) is None
        assert chat_template_mismatch(SimpleNamespace(chat_template=None), Style.GEMMA) is None

    def test_every_chat_style_has_a_frame(self):
        assert set(CHAT_FRAMES) == set(Style) - {Style.PLAIN}


class TestPresets:
    def test_every_preset_fits_its_gpu(self):
        from lev.train.config import PRESETS

        for config in PRESETS.values():
            config.validate()

    def test_the_12gb_preset_keeps_the_optimiser_batch_of_the_spark_preset(self):
        from lev.train.config import PRESETS

        spark, small = PRESETS["e4b-instruct"], PRESETS["e4b-instruct-12gb"]
        assert small.examples_per_step == spark.examples_per_step
        assert small.gpu.vram_gb == 12 and small.weights_gb < 6

    def test_bf16_e4b_does_not_fit_12gb(self):
        """Why the 12 GB preset quantizes: bf16 weights plus the per-layer table alone
        leave too little for activations."""
        from dataclasses import replace

        from lev.train.config import PRESETS

        with pytest.raises(ValueError):
            replace(PRESETS["e4b-instruct-12gb"], quantize=None).validate()

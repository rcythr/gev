"""Gemma 4 backbone handling, on a tiny offline model with every E-model feature:
per-layer embeddings, shared KV layers, a final logit softcap and a vision tower."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
gemma4 = pytest.importorskip("transformers.models.gemma4")

from gev.backbone import (  # noqa: E402
    final_logit_softcap,
    lora_target_pattern,
)
from gev.model import DecisionEngine, EngineConfig  # noqa: E402
from gev.prompt import CHAT_FRAMES, Style, chat_template_mismatch  # noqa: E402

ROWS = [[2, 5, 6, 7, 8, 9], [2, 5, 6], [2, 5, 6, 7]]


def tiny_gemma(vocab_size: int = 512):
    torch.manual_seed(0)
    text = gemma4.Gemma4TextConfig(
        vocab_size=vocab_size,
        vocab_size_per_layer_input=vocab_size,
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
    vision_config = gemma4.Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
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
    model = tiny_gemma()  # the encoders stay loaded; text rows must not reach them
    assert final_logit_softcap(model) == 30.0
    engine = DecisionEngine(model, SimpleNamespace(pad_token_id=0), EngineConfig())

    logits, hidden = engine._forward_single(ROWS, want_hidden=True)

    for i, row in enumerate(ROWS):
        expected_logits, expected_hidden = full_logits(model, row)
        torch.testing.assert_close(logits[i], expected_logits)
        torch.testing.assert_close(hidden[i], expected_hidden)


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
        from gev.train.config import PRESETS

        for config in PRESETS.values():
            config.validate()

    def test_bf16_e4b_fits_every_listed_gpu(self):
        from dataclasses import replace

        from gev.train.config import GPUS, PRESETS

        for gpu in GPUS.values():
            replace(PRESETS["e4b-instruct"], gpu=gpu).validate()


def test_training_loads_onto_one_gpu_even_when_several_are_visible(monkeypatch):
    """A copy split over GPUs leaves Gemma's last hidden state on another device than the
    `logits_to_keep` indices the loop builds, and `forward` refuses them."""
    import gev.backbone as backbone
    from gev.train.config import PRESETS
    from gev.train.loop import build_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device_maps = []

    def from_pretrained(model_id, **kwargs):
        device_maps.append(kwargs["device_map"])
        return tiny_gemma()

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", from_pretrained)
    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained", lambda *a, **k: SimpleNamespace(pad_token_id=0)
    )

    build_model(PRESETS["smoke"])  # training, calibration and evaluation all load here
    backbone.load_backbone("served", dtype=torch.float32)

    assert device_maps == [None, "auto"], "serving may still spread one copy over GPUs"


class TestLoraScale:
    """The adapter adds `scaling * B @ A`; scaling that moves the model along the line from the
    frozen backbone to the trained weights."""

    @pytest.fixture
    def adapted(self):
        peft = pytest.importorskip("peft")
        model = peft.get_peft_model(
            tiny_gemma(),
            peft.LoraConfig(
                r=4,
                lora_alpha=8,
                lora_dropout=0.0,
                target_modules=lora_target_pattern(("q_proj", "v_proj", "down_proj")),
                task_type="CAUSAL_LM",
            ),
        ).eval()
        torch.manual_seed(1)
        for name, p in model.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(p, std=0.05)
        return model

    def logits(self, model):
        with torch.no_grad():
            return model(input_ids=torch.tensor([ROWS[0]])).logits[0, -1]

    def test_zero_is_the_frozen_backbone_and_one_the_trained_model(self, adapted):
        from gev.backbone import set_lora_scale

        trained = self.logits(adapted)
        with adapted.disable_adapter():
            frozen = self.logits(adapted)
        assert not torch.allclose(trained, frozen), "the adapter must change something"

        assert set_lora_scale(adapted, 0.0) > 0
        torch.testing.assert_close(self.logits(adapted), frozen)
        set_lora_scale(adapted, 1.0)
        torch.testing.assert_close(self.logits(adapted), trained)

    def test_half_scale_is_the_halfway_point_in_weight_space(self, adapted):
        from gev.backbone import set_lora_scale

        set_lora_scale(adapted, 0.5)
        half = self.logits(adapted)

        set_lora_scale(adapted, 1.0)
        with torch.no_grad():
            for name, p in adapted.named_parameters():
                if "lora_B" in name:
                    p.mul_(0.5)
        torch.testing.assert_close(self.logits(adapted), half)

    def test_the_scale_is_absolute_not_compounding(self, adapted):
        from gev.backbone import set_lora_scale

        set_lora_scale(adapted, 0.5)
        once = self.logits(adapted)
        set_lora_scale(adapted, 0.5)
        torch.testing.assert_close(self.logits(adapted), once)

    def test_the_engine_scales_under_its_lock_and_refuses_without_an_adapter(self, adapted):
        engine = DecisionEngine(adapted, SimpleNamespace(pad_token_id=0), EngineConfig())
        before, _ = engine._forward_single([ROWS[0]], want_hidden=False)

        engine.set_lora_scale(0.0)
        frozen, _ = engine._forward_single([ROWS[0]], want_hidden=False)
        assert engine.lora_scale == 0.0 and not torch.allclose(before, frozen)

        plain = DecisionEngine(tiny_gemma(), SimpleNamespace(pad_token_id=0), EngineConfig())
        with pytest.raises(ValueError, match="no LoRA adapter"):
            plain.set_lora_scale(0.5)
        compiled = DecisionEngine(
            adapted, SimpleNamespace(pad_token_id=0), EngineConfig(compile=False)
        )
        compiled.config.compile = True
        with pytest.raises(RuntimeError, match="compiled graph"):
            compiled.set_lora_scale(0.5)

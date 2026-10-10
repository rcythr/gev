"""Training presets and approximate compute and memory budgets per GPU.

Estimate assumptions are documented in docs/ARCHITECTURE.md §5 and docs/TRAINING.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..prompt import PromptStyle

# ~6*N FLOPs/token forward+backward, plus ~a third for checkpointing's recompute. Holds
# under LoRA: backward still traverses the frozen weights to reach the adapters.
FLOPS_PER_PARAM_PER_TOKEN = 8.0


@dataclass(frozen=True)
class Gpu:
    name: str
    vram_gb: float
    # Achieved, not peak: padding, recompute and data loading included.
    effective_flops: float


# Measured on the released Qwen3.5 4b-instruct run (2.46e18 FLOPs in 7.8 h). docs/TRAINING.md.
H100 = Gpu("H100 80GB", 80.0, 8.75e13)
# The rest scale the H100 figure by peak dense bf16 throughput; unmeasured. Spark's
# 128 GB is unified memory shared with the OS; ~119 GB is visible to CUDA.
DGX_SPARK = Gpu("DGX Spark (GB10)", 119.0, 8.8e12)
A100_80 = Gpu("A100 80GB", 80.0, 2.76e13)
A100_40 = Gpu("A100 40GB", 40.0, 2.76e13)
L4 = Gpu("L4 24GB", 24.0, 1.07e13)
GPUS = {"h100": H100, "a100-80": A100_80, "a100-40": A100_40, "l4": L4, "spark": DGX_SPARK}


@dataclass
class TrainConfig:
    model_id: str = "google/gemma-4-E4B"
    # Parameters every token multiplies through, the tied embedding included.
    params_b: float = 4.5
    # Gemma E-models' per-layer embedding table: memory, but lookups, not FLOPs.
    per_layer_embedding_params_b: float = 2.8
    # Fallback only: `build_head` reads the backbone's own config.
    hidden_size: int = 2560
    dtype: str = "bfloat16"
    gpu: Gpu = H100

    # LoRA leaves room for activations on one H100 (§5.2).
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    # The new head trains in full even when LoRA freezes the backbone.
    train_mode_b_head: bool = True
    mode_b_proj_dim: int = 512
    # None uses the tokenizer limit; larger option sets train Mode B (ADR-026).
    max_label_options: int | None = None
    # Serving must match; the release manifest records this format.
    prompt_style: PromptStyle = "plain"

    n_examples: int = 200_000
    # Re-measure with `lev plan --data <dir>` after changing the mixture (ADR-016).
    avg_tokens_per_example: int = 128
    # A truncation cap, not the training length: batches pad to their longest row.
    max_seq_len: int = 2_048
    schema_first_fraction: float = 0.5
    # Unanswerable rows with a uniform target: spread mass instead of guessing (§5.6).
    abstain_fraction: float = 0.1

    epochs: int = 3
    per_device_batch: int = 32
    # Length-sorting window in batches; limits padding while preserving randomness.
    bucket_window: int = 64
    grad_accum: int = 1
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    gradient_checkpointing: bool = True
    brier_weight: float = 0.5
    # Every ordered readout: Score and Noul, both modes (see readout/mode_b.py).
    ordinal_weight: float = 0.25

    seed: int = 17
    checkpoint_every: int = 2_000
    log_every: int = 25
    output_dir: str = "checkpoints/lev"
    # Older `step-N` directories are deleted past this many; None keeps all.
    keep_last: int | None = None

    @property
    def tokens_per_epoch(self) -> int:
        return self.n_examples * self.avg_tokens_per_example

    @property
    def total_tokens(self) -> int:
        return self.tokens_per_epoch * self.epochs

    @property
    def tokens_per_step(self) -> int:
        # The work per step is the padded batch, not `max_seq_len` (~16x larger).
        return self.avg_tokens_per_example * self.per_device_batch * self.grad_accum

    @property
    def examples_per_step(self) -> int:
        return self.per_device_batch * self.grad_accum

    @property
    def steps_per_epoch(self) -> int:
        return max(1, self.n_examples // self.examples_per_step)

    @property
    def total_steps(self) -> int:
        return self.steps_per_epoch * self.epochs

    @property
    def total_flops(self) -> float:
        return FLOPS_PER_PARAM_PER_TOKEN * self.params_b * 1e9 * self.total_tokens

    @property
    def estimated_hours(self) -> float:
        return self.total_flops / self.gpu.effective_flops / 3600

    @property
    def weights_gb(self) -> float:
        return (self.params_b + self.per_layer_embedding_params_b) * 2  # bf16

    @property
    def optimizer_gb(self) -> float:
        """Weights + grads + AdamW fp32 (m, v, master). ~16 bytes/param full-FT."""
        if self.use_lora:
            return self.weights_gb + 0.3
        return self.params_b * 16 + self.per_layer_embedding_params_b * 2

    @property
    def headroom_gb(self) -> float:
        return self.gpu.vram_gb - self.optimizer_gb

    def summary(self) -> str:
        adaptation = f"LoRA r{self.lora_rank}" if self.use_lora else "full fine-tune"
        prompt = f"{self.prompt_style} prompts, Mode A to the tokenizer limit"
        return "\n".join(
            [
                f"model            {self.model_id}  ({self.params_b}B, {self.dtype})",
                f"adaptation       {adaptation}",
                f"prompt           {prompt}",
                f"data             {self.n_examples:,} examples x "
                f"{self.avg_tokens_per_example} tok x {self.epochs} epochs"
                f"  = {self.total_tokens / 1e9:.2f}B tokens",
                f"steps            {self.total_steps:,} "
                f"({self.examples_per_step} ex/step, ~{self.tokens_per_step:,} tok/step)",
                f"compute          {self.total_flops:.2e} FLOPs",
                f"time estimate    {self.estimated_hours:.1f} hours "
                f"({self.estimated_hours / 24:.1f} days) on {self.gpu.name}",
                f"memory           {self.optimizer_gb:.1f} GB state, "
                f"{self.headroom_gb:.1f} GB headroom of {self.gpu.vram_gb:.0f} GB",
            ]
        )

    def validate(self) -> None:
        if self.optimizer_gb >= self.gpu.vram_gb:
            raise ValueError(
                f"{self.optimizer_gb:.0f} GB of optimiser state does not fit one "
                f"{self.gpu.name}. Enable LoRA or choose a smaller backbone."
            )
        # A quarter of the card: the 20 GB of an H100 that held batch 32 at 128 tokens.
        if self.headroom_gb < self.gpu.vram_gb / 4:
            raise ValueError(
                f"only {self.headroom_gb:.1f} GB left for activations. "
                f"Expect OOM at seq_len={self.max_seq_len}."
            )
        if not 0.0 <= self.schema_first_fraction <= 1.0:
            raise ValueError("schema_first_fraction must be in [0, 1]")


# Copy with dataclasses.replace before applying per-run overrides.
PRESETS: dict[str, TrainConfig] = {
    "e4b": TrainConfig(),
    # A lower learning rate preserves the instruct backbone's existing abilities (ADR-020).
    # Budgeted for a DGX Spark; `lev plan --gpu h100` prices the same run on an H100.
    "e4b-instruct": TrainConfig(
        model_id="google/gemma-4-E4B-it",
        learning_rate=5e-5,
        prompt_style="gemma",
        gpu=DGX_SPARK,
        output_dir="checkpoints/lev-e4b-instruct",
    ),
    "smoke": TrainConfig(
        model_id="google/gemma-4-E2B-it",
        params_b=2.3,
        per_layer_embedding_params_b=2.3,
        hidden_size=1536,
        prompt_style="gemma",
        n_examples=2_000,
        epochs=1,
        max_seq_len=1_024,
        per_device_batch=2,  # CPU-runnable; `make smoke-local` uses this
        checkpoint_every=0,  # one checkpoint, at the end
        output_dir="checkpoints/lev-smoke",
    ),
}

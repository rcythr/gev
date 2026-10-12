"""Progress reporting for a training run."""

from __future__ import annotations

import time
from collections.abc import Callable


class ProgressLog:
    """Periodic one-line progress, flushed: on Modal stdout is the only view, and it is
    block-buffered when not a tty.

    Losses are per mode (Mode B starts near `ln(K)`). Throughput counts real tokens from the
    attention mask, so it can contradict a wrong plan (ADR-016).
    """

    def __init__(
        self,
        total_steps: int,
        log_every: int,
        on_report: Callable[[dict[str, float], int], None] | None = None,
    ):
        self.total = total_steps
        self.on_report = on_report
        self.every = max(1, log_every)
        self.start = time.monotonic()
        self.window: dict[str, list[float]] = {}
        self.window_tokens = 0
        self.mark = self.start
        self.mark_step = 0

    def record(self, step: int, loss: float, mode: str, lr: float, tokens: int = 0) -> None:
        self.window.setdefault(mode, []).append(loss)
        self.window_tokens += tokens
        done = step + 1
        if done % self.every and done != self.total:
            return

        now = time.monotonic()
        elapsed = max(now - self.start, 1e-9)
        # Over the window, not since start: startup skews a cumulative rate ~8x (ADR-017).
        # A coarse clock can report zero on a fast window; never divide by it.
        span = max(now - self.mark, 1e-9)
        rate = (done - self.mark_step) / span
        remaining = (self.total - done) / rate if rate else 0.0
        means = {m: sum(v) / len(v) for m, v in sorted(self.window.items())}
        losses = "  ".join(f"{m}={loss:.4f}" for m, loss in means.items())
        tok_rate = self.window_tokens / span
        memory = _gpu_mem_gb()
        print(
            f"step {done:>6}/{self.total}  {100 * done / self.total:5.1f}%  "
            f"{losses}  lr={lr:.2e}  {rate:.2f} it/s  "
            f"{tok_rate:,.0f} tok/s  "
            f"elapsed {hms(elapsed)}  eta {hms(remaining)}"
            f"{f'  mem {memory:.1f}G' if memory is not None else ''}",
            flush=True,
        )
        if self.on_report is not None:
            metrics = {f"loss/{m}": loss for m, loss in means.items()}
            metrics |= {"lr": lr, "steps_per_s": rate, "tokens_per_s": tok_rate}
            if memory is not None:
                metrics["gpu_mem_peak_gb"] = memory
            self.on_report(metrics, done)
        self.window.clear()
        self.window_tokens = 0
        self.mark = now
        self.mark_step = done


def hms(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _gpu_mem_gb() -> float | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 2**30

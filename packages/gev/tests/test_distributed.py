"""Data-parallel pieces that can be checked on CPU: the shard, and a real two-process
gradient all-reduce over gloo."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gev.train.distributed import Ranks, average_gradients, shard  # noqa: E402


def test_shard_interleaves_and_gives_every_rank_the_same_count():
    groups = list(range(7))
    assert shard(groups, Ranks(0, 2, 0)) == [0, 2, 4]
    assert shard(groups, Ranks(1, 2, 1)) == [1, 3, 5]


def test_one_rank_takes_every_batch():
    assert shard(list(range(7)), Ranks()) == list(range(7))


def _all_reduce_worker(rank: int, port: int, results) -> None:
    import os

    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=2)
    shared, one_rank, unused = (torch.nn.Parameter(torch.zeros(2)) for _ in range(3))
    shared.grad = torch.full((2,), float(rank + 1))  # 1 and 2: mean 1.5
    if rank == 1:
        one_rank.grad = torch.full((2,), 4.0)  # rank 0 counts as zero: mean 2
    average_gradients([shared, one_rank, unused], Ranks(rank, 2, rank))
    results[rank] = (shared.grad.tolist(), one_rank.grad.tolist(), unused.grad)
    dist.destroy_process_group()


def test_gradients_average_across_ranks_and_unused_ones_stay_none():
    import socket

    import torch.multiprocessing as mp

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    results = mp.Manager().dict()
    mp.spawn(_all_reduce_worker, args=(port, results), nprocs=2)

    for rank in (0, 1):
        shared, one_rank, unused = results[rank]
        assert shared == [1.5, 1.5]
        assert one_rank == [2.0, 2.0], "an unused grad counts as zero, as under DDP"
        assert unused is None, "no rank used it: AdamW must skip it, not step on zeros"


class TestWorldSizeGuards:
    def test_a_multi_gpu_preset_refuses_to_run_on_one_process(self, train_config, tmp_path):
        from gev.train.loop import run_training

        train_config.n_gpus = 2
        with pytest.raises(ValueError, match="torchrun --nproc_per_node 2"):
            run_training(train_config, str(tmp_path))

    def test_a_checkpoint_resumes_only_at_its_own_world_size(self, tmp_path, monkeypatch):
        import gev.train.loop as loop
        from test_training_loop import run_resumable

        run_resumable(tmp_path, monkeypatch, max_steps=6)
        monkeypatch.setattr(loop, "setup", lambda: Ranks(0, 2, 0))
        with pytest.raises(ValueError, match="saved by a 1-GPU run"):
            run_resumable(tmp_path, monkeypatch, max_steps=10)


def _train_worker(rank: int, world: int, port: int, accum: int, max_steps: int, out, results):
    """One rank of a real run: tiny Gemma 4 + LoRA + Mode B head, real loop and collator."""
    import os
    from dataclasses import replace

    import gev.train.loop as loop
    import peft
    from conftest import a_choice, an_example
    from gev.backbone import lora_target_pattern
    from gev.train.config import PRESETS
    from test_gemma import tiny_gemma
    from test_training_loop import make_batching_tokenizer

    if world > 1:
        os.environ.update(
            RANK=str(rank),
            LOCAL_RANK=str(rank),
            WORLD_SIZE=str(world),
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
        )
    # 4 options read label tokens; 30 outrun the single-letter codes and take the head.
    rows = [an_example(a_choice(4), target=i % 4, state=f"row {i}" * (i % 3 + 1)) for i in range(8)]
    rows += [an_example(a_choice(30), target=i % 30, state=f"wide {i}") for i in range(8)]

    def build_model(config, model_cache=None):
        torch.manual_seed(0)
        model = peft.get_peft_model(
            tiny_gemma(vocab_size=1100),  # the fake tokenizer's label ids start at 1000
            peft.LoraConfig(
                r=4,
                lora_dropout=0.0,
                target_modules=lora_target_pattern(config.lora_targets),
                task_type="CAUSAL_LM",
            ),
        )
        # Nonzero B, so the first step's gradient reaches A as well.
        for name, p in model.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(p, std=0.02)
        return model, make_batching_tokenizer()

    def capture(model, head, tokenizer, output, step, on_checkpoint=None, state=None):
        weights = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        weights |= {f"head.{n}": p.detach().clone() for n, p in head.named_parameters()}
        results[rank] = weights

    loop.prepare_data = lambda config, data_dir: {"train": rows, "calibration": []}
    loop.build_model = build_model
    loop.save_checkpoint = capture
    config = replace(
        PRESETS["smoke"],
        per_device_batch=2,
        grad_accum=accum,
        bucket_window=1,
        checkpoint_every=0,
        output_dir=str(out),
        learning_rate=1e-2,
    )
    loop.run_training(config, "unused", max_steps=max_steps, fresh=True)


def test_two_ranks_train_the_same_weights_as_one_rank_accumulating_two_batches(tmp_path):
    """Rank r takes batches r, r+2, ...; averaging their gradients is exactly one process
    accumulating each consecutive pair. Mixed Mode A and B batches exercise the head's
    unused-gradient path."""
    import socket

    import torch.multiprocessing as mp

    manager = mp.Manager()
    single, double = manager.dict(), manager.dict()
    mp.spawn(_train_worker, args=(1, 0, 2, 8, tmp_path / "one", single), nprocs=1)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(_train_worker, args=(2, port, 1, 4, tmp_path / "two", double), nprocs=2)

    expected, got = single[0], double[0]
    assert set(got) == set(expected)
    assert any("head." in n for n in got) and any("lora_A" in n for n in got)
    for name, value in expected.items():
        torch.testing.assert_close(got[name], value, rtol=1e-4, atol=1e-6, msg=name)
    assert 1 not in double, "only rank 0 writes checkpoints"

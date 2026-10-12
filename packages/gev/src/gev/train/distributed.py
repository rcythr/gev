"""Data-parallel training across the GPUs of one machine, launched with `torchrun`.

Every rank holds the whole model and takes its own share of each epoch's batches;
gradients are averaged once per optimiser step. Without `torchrun` this is one rank.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Ranks:
    rank: int = 0
    world: int = 1
    local_rank: int = 0

    @property
    def main(self) -> bool:
        return self.rank == 0


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def setup() -> Ranks:
    """Join the process group `torchrun` describes in the environment, if any."""
    if world_size() <= 1:
        return Ranks()
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    return Ranks(dist.get_rank(), dist.get_world_size(), local_rank)


def teardown(ranks: Ranks) -> None:
    if ranks.world > 1:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()


def shard(groups: list, ranks: Ranks) -> list:
    """This rank's batches: every `world`-th, after dropping the remainder.

    Every rank must take the same number of steps, or one waits forever in the
    gradient all-reduce for a peer that has finished the epoch.
    """
    usable = len(groups) - len(groups) % ranks.world
    return groups[ranks.rank : usable : ranks.world]


def average_gradients(params: list, ranks: Ranks) -> None:
    """Average `params`' gradients over all ranks in one all-reduce.

    A parameter this rank did not use (the Mode B head on a Mode A batch) contributes
    zeros: skipping it would desynchronise the collective across ranks. One no rank
    used gets its gradient back as None, so AdamW skips it as it would on one GPU.
    """
    if ranks.world <= 1:
        return
    import torch
    import torch.distributed as dist

    used = torch.tensor([p.grad is not None for p in params], dtype=torch.float32)
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    device = params[0].grad.device
    flat = torch.cat([p.grad.reshape(-1).float() for p in params] + [used.to(device)])
    dist.all_reduce(flat)
    offset = 0
    for p, users in zip(params, flat[-len(params) :].tolist(), strict=True):
        n = p.grad.numel()
        if users:
            p.grad.copy_((flat[offset : offset + n] / ranks.world).view_as(p.grad))
        else:
            p.grad = None
        offset += n

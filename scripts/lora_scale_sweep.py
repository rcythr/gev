#!/usr/bin/env python3
"""Score a served model at several LoRA strengths and print accuracy per subset and scale.

Start the server with the runtime switch, then run this against it:

    uv run gev serve --checkpoint weights/<release> --allow-lora-scale --port 8000
    python scripts/lora_scale_sweep.py --tasks data/s1bench

Scale 0 is the frozen backbone, 1 the trained model. Each scale runs `gevbench eval` on the
same tasks and keeps its raw output under --out. Temperatures were fitted at scale 1, so ECE
at other scales is not meaningful; compare accuracy.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import statistics
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

SUBSET = re.compile(r"^=== (\S+)\s+\(([\d,]+) items\) ===$", re.M)
SCORES = re.compile(r"accuracy ([\d.]+)\s+log-loss ([\d.]+)\s+brier ([\d.]+)\s+ECE ([\d.]+)")


def parse_eval(output: str) -> dict[str, dict[str, float]]:
    """Subset -> accuracy, ECE and item count from a `gevbench eval` report."""
    marks = list(SUBSET.finditer(output))
    results = {}
    for mark, following in zip(marks, [*marks[1:], None], strict=True):
        block = output[mark.end() : following.start() if following else len(output)]
        if scores := SCORES.search(block):
            results[mark.group(1)] = {
                "accuracy": float(scores.group(1)),
                "ece": float(scores.group(4)),
                "items": int(mark.group(2).replace(",", "")),
            }
    return results


def set_scale(base_url: str, scale: float) -> None:
    request = urllib.request.Request(
        f"{base_url}/admin/lora-scale",
        data=json.dumps({"scale": scale}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            applied = json.load(response)["lora_scale"]
    except urllib.error.HTTPError as error:
        hint = " (start the server with --allow-lora-scale)" if error.code == 404 else ""
        sys.exit(f"setting scale {scale} failed: HTTP {error.code}{hint}: {error.read().decode()}")
    if applied != scale:
        sys.exit(f"asked for scale {scale}, server reports {applied}")


def table(scales: list[float], by_scale: dict[float, dict[str, dict[str, float]]]) -> str:
    subsets = sorted({name for results in by_scale.values() for name in results})
    header = f"{'subset':<24}" + "".join(f"{s:>8g}" for s in scales)
    lines = [header, "-" * len(header)]
    for name in subsets:
        cells = [by_scale[s].get(name, {}).get("accuracy") for s in scales]
        lines.append(
            f"{name:<24}" + "".join(f"{c:>8.3f}" if c is not None else f"{'-':>8}" for c in cells)
        )
    complete = [n for n in subsets if all(n in by_scale[s] for s in scales)]
    lines.append("-" * len(header))
    macro = [statistics.mean(by_scale[s][n]["accuracy"] for n in complete) for s in scales]
    lines.append(
        f"{'macro (' + str(len(complete)) + ' subsets)':<24}" + "".join(f"{m:>8.3f}" for m in macro)
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--tasks", default="data/s1bench")
    parser.add_argument("--scales", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--gevbench", default="uv run gevbench", help="how to invoke gevbench")
    parser.add_argument("--out", default="artifacts/lora-sweep")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    by_scale: dict[float, dict[str, dict[str, float]]] = {}
    for scale in args.scales:
        set_scale(args.base_url, scale)
        print(f"scale {scale:g}: scoring {args.tasks} ...", flush=True)
        command = [
            *shlex.split(args.gevbench),
            *["eval", "--backend", "gev", "--base-url", args.base_url],
            *["--tasks", args.tasks, "--concurrency", str(args.concurrency)],
        ]
        run = subprocess.run(command, capture_output=True, text=True)
        (out / f"scale-{scale:g}.txt").write_text(run.stdout + run.stderr)
        if run.returncode:
            sys.exit(f"gevbench failed at scale {scale:g}; see {out / f'scale-{scale:g}.txt'}")
        by_scale[scale] = parse_eval(run.stdout)

    print()
    print(table(args.scales, by_scale))
    (out / "sweep.json").write_text(json.dumps({str(s): r for s, r in by_scale.items()}, indent=2))
    print(f"\nraw reports and sweep.json in {out}")


if __name__ == "__main__":
    main()

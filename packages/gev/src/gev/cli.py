"""`gev`: plan a run, build the data, train, package, publish and serve the model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

PROMPT_STYLES = ["plain", "chat", "gemma"]


def _measure_tokens(config, data_dir: str, sample: int = 1500) -> dict:
    """Length distribution of real rendered prompts; the budget rests on it (ADR-016)."""
    import statistics

    from transformers import AutoTokenizer

    from .data.build import load_split
    from .data.splits import Split
    from .prompt import Style
    from .train.collate import RouteCache, render

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    rows = load_split(data_dir, Split.TRAIN)[:sample]
    # The preset's own prompt style: `chat` adds a system turn to every prompt.
    style = Style(config.prompt_style)
    routes = RouteCache(tokenizer, config.max_label_options, style)
    lengths = sorted(
        len(tokenizer.encode(render(e, routes.route_for(e).codes, style), add_special_tokens=False))
        for e in rows
    )
    return {
        "n": len(lengths),
        "mean": round(statistics.mean(lengths)),
        "median": lengths[len(lengths) // 2],
        "p95": lengths[int(0.95 * len(lengths))],
        "max": lengths[-1],
    }


def _apply_overrides(config, args: argparse.Namespace) -> None:
    """Fit a preset to the GPU at hand; the optimiser batch is batch x accumulation."""
    from .train.config import GPUS

    if args.gpu:
        config.gpu = GPUS[args.gpu]
    if args.per_device_batch:
        config.per_device_batch = args.per_device_batch
    if args.grad_accum:
        config.grad_accum = args.grad_accum
    if getattr(args, "gpus", None):
        config.n_gpus = args.gpus
    if getattr(args, "keep_last", None):
        config.keep_last = args.keep_last


def _add_override_args(parser: argparse.ArgumentParser) -> None:
    from .train.config import GPUS

    parser.add_argument("--gpu", default=None, choices=sorted(GPUS), help="default: the preset's")
    parser.add_argument("--per-device-batch", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)


def cmd_plan(args: argparse.Namespace) -> None:
    from .train.config import PRESETS

    config = replace(PRESETS[args.preset])
    _apply_overrides(config, args)
    if args.data:
        measured = _measure_tokens(config, args.data)
        print(
            f"measured over {measured['n']:,} real prompts: "
            f"mean {measured['mean']}  median {measured['median']}  "
            f"p95 {measured['p95']}  max {measured['max']} tokens"
        )
        if measured["max"] > config.max_seq_len:
            print(
                f"  note: {measured['max']} > max_seq_len {config.max_seq_len}; "
                "the longest prompts will be truncated"
            )
        config.avg_tokens_per_example = measured["mean"]
    config.validate()
    print(config.summary())


def cmd_route(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    from .model import EngineConfig, serving_routes
    from .types import SystemOneRequest

    request = SystemOneRequest.model_validate(json.loads(Path(args.request).read_text()))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = EngineConfig(
        model_id=args.model,
        max_label_options=args.max_label_options,
        prompt_style=args.prompt_style,
        skip_multi_token_codes=not args.no_skip_codes,
    )
    for name, r in serving_routes(request.questions, tokenizer, config).items():
        codes = ""
        if r.codes:
            codes = f"  codes={r.codes[:6]}{'...' if len(r.codes) > 6 else ''}"
        print(f"{name:<24} mode {r.mode.value}  ({r.reason}){codes}")


def cmd_check_data(args: argparse.Namespace) -> None:
    from .data import BLOCKED_SUBSETS, ContaminationError, assert_clean

    names = [n.strip() for n in Path(args.sources).read_text().splitlines() if n.strip()]
    try:
        assert_clean(names)
    except ContaminationError as exc:
        sys.exit(str(exc))
    print(f"{len(names)} sources clean against {len(BLOCKED_SUBSETS)} blocked subsets")


def _preset_names() -> list[str]:
    """Lazy, like every CLI import, so `gev plan` and `gev check-data` run
    without the train extra."""
    from .train.config import PRESETS

    return sorted(PRESETS)


def cmd_data_build(args: argparse.Namespace) -> None:
    from .data.build import build_dataset

    manifest = build_dataset(
        args.out,
        limit_per_source=args.limit_per_source,
        n_examples=args.n_examples,
        schema_first_fraction=args.schema_first_fraction,
        abstain_fraction=args.abstain_fraction,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    print(f"wrote {args.out}")
    for split, count in manifest["split_counts"].items():
        print(f"  {split:<12} {count:>8,}")
    ratio = manifest["oversample_ratio"]
    print(
        f"  unique train rows {manifest['unique_train_rows']:,} "
        f"(oversample {ratio}x)"
        + ("  <- raise --limit-per-source to lower this" if ratio > 2 else "")
    )
    print(f"  sources      {len(manifest['sources'])} ({', '.join(sorted(manifest['sources']))})")


def cmd_data_eval(args: argparse.Namespace) -> None:
    from .data.export_eval import export
    from .data.splits import Split

    index = export(args.data, args.out, Split(args.split), args.limit_per_source)
    print(f"wrote {index['total_items']:,} items from the {index['split']} split to {args.out}")
    for source, info in index["sources"].items():
        print(f"  {source:<20} {info['items']:>6,}  {info['type']}")
    print(f"\n  uv run gevbench eval --backend gev --tasks {args.out}")


def cmd_s1bench_export(args: argparse.Namespace) -> None:
    from .data.s1bench import export

    index = export(args.out, args.subsets)
    print(f"wrote {index['total_items']:,} items to {args.out}")
    print(f"  {'subset':<22}{'items':>7}{'type':>8}{'jev 1.13':>10}")
    for subset, info in index["subsets"].items():
        print(f"  {subset:<22}{info['items']:>7,}{info['type']:>8}{info['jev_published']:>10.3f}")
    print("\n  validate the harness against Jev first:")
    print(f"    uv run gevbench eval --backend jev --tasks {args.out}")
    print("  then score gev on the same files:")
    print(f"    uv run gevbench eval --backend gev --tasks {args.out} --base-url $URL")


def cmd_train(args: argparse.Namespace) -> None:
    from .train.config import PRESETS
    from .train.loop import run_training

    config = replace(PRESETS[args.preset])
    _apply_overrides(config, args)
    if args.output_dir:
        config.output_dir = args.output_dir
    summary = run_training(
        config,
        args.data,
        model_cache=args.model_cache,
        resume_from=args.resume,
        max_steps=args.max_steps,
        fresh=args.fresh,
        mlflow=args.mlflow,
    )
    if os.environ.get("RANK", "0") != "0":
        return
    print(
        f"{summary['steps']} steps  "
        f"loss {summary['first_loss']:.4f} -> {summary['final_loss']:.4f}  "
        f"-> {summary['output_dir']}"
    )


def cmd_calibrate(args: argparse.Namespace) -> None:
    from .train.calibration_run import fit_profile
    from .train.config import PRESETS

    config = replace(PRESETS[args.preset])
    if args.output_dir:
        config.output_dir = args.output_dir
    result = fit_profile(config.output_dir, args.data, config=config, method=args.method)
    print(f"{len(result['temperatures'])} temperatures -> {result['profile']}")
    if args.mlflow:
        from .tracking import log_calibration, start

        tracker = start(config.output_dir, True)
        log_calibration(tracker, result)
        tracker.finish()


def cmd_release_build(args: argparse.Namespace) -> None:
    from .release import build_release
    from .train.checkpoints import resolve_checkpoint
    from .train.config import PRESETS

    checkpoint = args.checkpoint or PRESETS[args.preset].output_dir
    if args.mlflow:
        from .tracking import attach

        # Fail before copying anything if there is no training run to attach to.
        tracker = attach(resolve_checkpoint(checkpoint).parent)
    manifest = build_release(
        checkpoint,
        args.out,
        preset=args.preset,
        calibration=args.calibration,
        name=args.name,
        prompt_style=PRESETS[args.preset].prompt_style,
    )
    print(f"release {manifest['name']} -> {args.out}")
    print(
        f"  base {manifest['base_model']}  step {manifest['step']}  files {len(manifest['files'])}"
    )
    if not manifest["calibrated"]:
        print("  WARNING: no calibration.json; the release serves raw softmax")
    if args.mlflow:
        tracker.log_artifacts(args.out, "release")
        tracker.finish()
        print(f"  logged to MLflow run {tracker.run_id} under release/")
    print(f"\n  uv run gev serve --checkpoint {args.out}")
    print(f"  uv run gev release publish {args.out} --repo <org/name>")


def cmd_release_publish(args: argparse.Namespace) -> None:
    from .release import publish

    url = publish(args.release, args.repo, private=args.private)
    print(f"published -> {url}")
    print(f"\n  uv run gev serve --checkpoint {args.repo}")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .server import create_app

    uvicorn.run(
        create_app(
            checkpoint_dir=args.checkpoint,
            model_cache=args.model_cache,
            calibration=args.calibration,
            model_id=args.model,
            noul_readout=args.noul_readout,
            compile=args.compile,
            prompt_style=args.prompt_style,
            max_pending=args.max_pending,
            max_batch_tokens=args.max_batch_tokens,
        ),
        host=args.host,
        port=args.port,
    )


def main(argv: list[str] | None = None) -> None:
    from dotenv import find_dotenv, load_dotenv

    # The nearest `.env` from the working directory up fills in MLFLOW_*, HF_TOKEN and the
    # like; variables already set in the shell win.
    load_dotenv(find_dotenv(usecwd=True))
    parser = argparse.ArgumentParser(prog="gev", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan_parser = sub.add_parser("plan", help="print a preset's training budget")
    plan_parser.add_argument(
        "--data",
        default=None,
        help="measure token lengths from a built mixture instead of trusting the default",
    )
    plan_parser.add_argument("--preset", default="e4b-instruct", choices=_preset_names())
    _add_override_args(plan_parser)
    plan_parser.add_argument(
        "--gpus", type=int, default=None, help="data-parallel GPUs; default: the preset's"
    )
    plan_parser.set_defaults(func=cmd_plan)

    route_parser = sub.add_parser("route", help="show the readout mode per question")
    route_parser.add_argument("request", help="path to a /v1/systemone request JSON")
    route_parser.add_argument("--model", default="google/gemma-4-E4B-it")
    # Defaults mirror `EngineConfig`, so the preview matches what a server does.
    route_parser.add_argument(
        "--max-label-options",
        type=int,
        default=None,
        help="force Mode B above this many options (default: the tokenizer limit, as served)",
    )
    route_parser.add_argument("--prompt-style", choices=PROMPT_STYLES, default="plain")
    route_parser.add_argument(
        "--no-skip-codes", action="store_true", help="stop at the first two-token label code"
    )
    route_parser.set_defaults(func=cmd_route)

    check_data_parser = sub.add_parser("check-data", help="contamination guard over a source list")
    check_data_parser.add_argument("sources", help="file with one dataset name per line")
    check_data_parser.set_defaults(func=cmd_check_data)

    serve_parser = sub.add_parser("serve", help="serve /v1/systemone")
    serve_parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "a release directory, a Hub id, or a training output directory (or "
            "a step-N inside one). The LoRA adapter, the Mode B head and any "
            "calibration.json beside them are all picked up. Omit to serve the "
            "untrained base model."
        ),
    )
    serve_parser.add_argument("--model", default="google/gemma-4-E4B-it")
    serve_parser.add_argument("--model-cache", default=None)
    serve_parser.add_argument("--calibration", default=None)
    serve_parser.add_argument(
        "--noul-readout",
        choices=["rating", "binary"],
        default=None,
        help="default: rating with a checkpoint, binary without one",
    )
    serve_parser.add_argument(
        "--compile", action="store_true", help="torch.compile + CUDA graphs; warms up at startup"
    )
    serve_parser.add_argument(
        "--prompt-style",
        choices=PROMPT_STYLES,
        default=None,
        help="default: the release manifest's, else plain",
    )
    serve_parser.add_argument(
        "--max-pending",
        type=int,
        default=64,
        help="requests in flight before new ones get 503 (default 64)",
    )
    serve_parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=16384,
        help="padded tokens per batched forward across requests (default 16384)",
    )
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.set_defaults(func=cmd_serve)

    data_parser = sub.add_parser("data", help="build the training mixture and the eval set")
    data_sub = data_parser.add_subparsers(dest="data_command", required=True)

    build_parser = data_sub.add_parser("build", help="download, split and write the mixture")
    build_parser.add_argument("--out", default="data/mixture")
    build_parser.add_argument(
        "--limit-per-source",
        type=int,
        default=20_000,
        help="rows sampled per source (shuffled first, never a head slice)",
    )
    build_parser.add_argument("--n-examples", type=int, default=200_000)
    build_parser.add_argument("--schema-first-fraction", type=float, default=0.5)
    build_parser.add_argument("--abstain-fraction", type=float, default=0.1)
    build_parser.add_argument("--seed", type=int, default=17)
    build_parser.add_argument("--cache-dir", default=None)
    build_parser.set_defaults(func=cmd_data_build)

    eval_parser = data_sub.add_parser("eval", help="export a held-out split as gevbench task files")
    eval_parser.add_argument("--data", default="data/mixture")
    eval_parser.add_argument("--out", default="data/eval")
    eval_parser.add_argument("--split", default="test", choices=["test", "calibration"])
    eval_parser.add_argument("--limit-per-source", type=int, default=None)
    eval_parser.set_defaults(func=cmd_data_eval)

    s1_parser = sub.add_parser(
        "s1bench", help="export the S1Bench evaluation subsets (evaluation only)"
    )
    s1_sub = s1_parser.add_subparsers(dest="s1bench_command", required=True)
    s1_export = s1_sub.add_parser("export", help="write the subsets as gevbench task files")
    s1_export.add_argument("--out", default="data/s1bench")
    s1_export.add_argument(
        "--subsets",
        nargs="*",
        default=None,
        help="subset names; defaults to all 13",
    )
    s1_export.set_defaults(func=cmd_s1bench_export)

    release_parser = sub.add_parser("release", help="package a checkpoint; publish it to the Hub")
    release_sub = release_parser.add_subparsers(dest="release_command", required=True)
    build_release_parser = release_sub.add_parser(
        "build", help="copy adapter, head, tokenizer and calibration into one release dir"
    )
    build_release_parser.add_argument(
        "--checkpoint", default=None, help="a step-N dir or its parent; default: the preset's"
    )
    build_release_parser.add_argument("--out", required=True)
    build_release_parser.add_argument("--preset", default="e4b-instruct", choices=_preset_names())
    build_release_parser.add_argument("--calibration", default=None)
    build_release_parser.add_argument("--name", default=None)
    build_release_parser.add_argument(
        "--mlflow", action="store_true", help="upload the release to the run that trained it"
    )
    build_release_parser.set_defaults(func=cmd_release_build)
    publish_parser = release_sub.add_parser("publish", help="upload a release dir to the Hub")
    publish_parser.add_argument("release", help="a directory written by `gev release build`")
    publish_parser.add_argument("--repo", required=True, help="Hub id, e.g. org/gev")
    publish_parser.add_argument("--private", action="store_true")
    publish_parser.set_defaults(func=cmd_release_publish)

    train_parser = sub.add_parser("train", help="run the LoRA fine-tune")
    train_parser.add_argument("--preset", default="e4b-instruct", choices=_preset_names())
    train_parser.add_argument("--data", default="data/mixture")
    train_parser.add_argument("--output-dir", default=None)
    train_parser.add_argument("--model-cache", default=None)
    _add_override_args(train_parser)
    train_parser.add_argument(
        "--mlflow",
        action="store_true",
        help="log to MLflow (MLFLOW_TRACKING_URI, ..._EXPERIMENT_NAME)",
    )
    train_parser.add_argument(
        "--keep-last", type=int, default=None, help="delete all but the newest N checkpoints"
    )
    train_parser.add_argument(
        "--max-steps", type=int, default=None, help="stop early; for smoke runs"
    )
    train_parser.add_argument(
        "--resume", default=None, help="a step-N directory; default: newest in --output-dir"
    )
    train_parser.add_argument(
        "--fresh", action="store_true", help="ignore existing checkpoints and start over"
    )
    train_parser.set_defaults(func=cmd_train)

    calibrate_parser = sub.add_parser(
        "calibrate", help="fit per-bucket temperatures on the calibration split"
    )
    calibrate_parser.add_argument("--preset", default="e4b-instruct", choices=_preset_names())
    calibrate_parser.add_argument("--data", default="data/mixture")
    calibrate_parser.add_argument(
        "--output-dir", default=None, help="the training output; default: the preset's"
    )
    calibrate_parser.add_argument("--method", default="transfer", choices=["transfer", "rows"])
    calibrate_parser.add_argument(
        "--mlflow", action="store_true", help="add the temperatures to the run's MLflow record"
    )
    calibrate_parser.set_defaults(func=cmd_calibrate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

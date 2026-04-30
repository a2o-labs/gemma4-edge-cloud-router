"""Command-line entry point for the V1.5 router.

    python -m router.cli encode "Hello world"
    python -m router.cli pipeline-v15 "How does photosynthesis work?"
    python -m router.cli health

Outputs JSON to stdout; logs to stderr. V15_MOCK_MODE=true bypasses
GPU/HF requirements and uses canned mock encoder + adapter.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _ensure_v15_settings():
    """Reset settings cache and read V15_* env vars."""
    import router.config

    router.config._settings = None
    return router.config.get_settings()


def cmd_encode(args) -> int:
    """Run only the edge encoder, print the resulting CompactSchemaV15 envelope."""
    settings = _ensure_v15_settings()
    if not settings.v15.enabled:
        print("V15_ENABLED is false. Set V15_ENABLED=true to run.", file=sys.stderr)
        return 2

    from router.v15_pipeline import V15Pipeline

    p = V15Pipeline(settings.v15)
    p.load()
    if not p.is_ready():
        print("V1.5 pipeline failed to load. Check logs.", file=sys.stderr)
        return 3

    schema, _vec = p._encoder.encode(args.prompt, return_schema=True)
    output = json.loads(schema.model_dump_json())
    if args.no_embedding:
        output.pop("embedding_b64", None)
    if args.pretty:
        print(json.dumps(output, indent=2))
    else:
        print(json.dumps(output))
    return 0


def cmd_pipeline_v15(args) -> int:
    """Run the full V1.5 pipeline and print the final response."""
    settings = _ensure_v15_settings()
    if not settings.v15.enabled:
        print("V15_ENABLED is false. Set V15_ENABLED=true to run.", file=sys.stderr)
        return 2

    from router.v15_pipeline import V15Pipeline

    p = V15Pipeline(settings.v15)
    p.load()
    if not p.is_ready():
        print("V1.5 pipeline failed to load. Check logs.", file=sys.stderr)
        return 3

    schema, answer = p.run(args.prompt)
    output = {
        "task_id": schema.task_id,
        "complexity": schema.complexity,
        "answer": answer,
        "schema_version": schema.version,
        "embedding_dim": schema.embedding_dim,
    }
    if args.pretty:
        print(json.dumps(output, indent=2))
    else:
        print(json.dumps(output))
    return 0


def cmd_health(args) -> int:
    """Print V15 readiness state without running anything."""
    settings = _ensure_v15_settings()

    out: dict[str, Any] = {
        "v15_enabled": settings.v15.enabled,
        "mock_mode": settings.v15.mock_mode,
        "device": settings.v15.device,
        "edge_model": settings.v15.edge_model_name,
        "cloud_model": settings.v15.cloud_model_name,
    }

    try:
        import torch

        out["torch_version"] = torch.__version__
        out["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            out["cuda_device"] = torch.cuda.get_device_name(0)
            out["cuda_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
    except ImportError:
        out["torch_version"] = None

    print(json.dumps(out, indent=2 if args.pretty else None))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="router.cli",
        description="V1.5 router CLI - local one-shot inference without HTTP.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    enc = sub.add_parser("encode", help="Run only the edge encoder.")
    enc.add_argument("prompt", help="Text prompt to encode.")
    enc.add_argument("--pretty", action="store_true", help="Indent JSON output.")
    enc.add_argument(
        "--no-embedding",
        action="store_true",
        help="Strip embedding_b64 from output (for human-readable schema).",
    )
    enc.set_defaults(func=cmd_encode)

    pipe = sub.add_parser("pipeline-v15", help="Run the full V1.5 pipeline.")
    pipe.add_argument("prompt", help="Text prompt.")
    pipe.add_argument("--pretty", action="store_true", help="Indent JSON output.")
    pipe.set_defaults(func=cmd_pipeline_v15)

    hp = sub.add_parser("health", help="Report V1.5 settings and CUDA status.")
    hp.add_argument("--pretty", action="store_true", default=True)
    hp.set_defaults(func=cmd_health)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

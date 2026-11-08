"""python -m ca_tool.generate --input examples/core_price_objection.json"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from .pipeline import error_object, exit_code, generate
from .settings import Settings


def reject_constant(value):
    raise ValueError(f"Non-finite JSON number is not accepted: {value}")


def cli_failure(status, code, message):
    return {"status": status, "request_id": None, "run_id": None, "scenario": None,
            "candidates": [], "warnings": [], "error": error_object(code, message),
            "metadata": {"source": "not_run", "attempts": 0, "run_dir": None}}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate three Chinese WeChat drafts through a configured model API")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--preflight", action="store_true", help="Validate input and context without any API call")
    args = parser.parse_args(argv)
    try:
        settings = Settings.from_env()
        if args.runs_dir:
            settings = replace(settings, runs_dir=args.runs_dir)
    except (OSError, ValueError) as exc:
        print(json.dumps(cli_failure("error", "CONFIG_ERROR", str(exc)), ensure_ascii=False, indent=2))
        return 2
    try:
        if args.input.stat().st_size > settings.max_input_bytes:
            raise ValueError("Input file exceeds MAX_INPUT_BYTES")
        raw = json.loads(args.input.read_text(encoding="utf-8-sig"), parse_constant=reject_constant)
    except (OSError, ValueError) as exc:
        print(json.dumps(cli_failure("needs_input", "INPUT_INVALID", str(exc)), ensure_ascii=False, indent=2))
        return 2
    if args.preflight:
        from .context import load_brand, prepare_context
        from .schemas import Request
        try:
            brand = load_brand(settings.brand_path)
        except (ValueError, OSError) as exc:
            print(json.dumps(cli_failure("error", "CONFIG_ERROR", f"Cannot load brand configuration: {type(exc).__name__}"), ensure_ascii=False, indent=2))
            return 2
        try:
            check = prepare_context(Request.model_validate(raw), brand)
            print(json.dumps({"status": check["status"], "warnings": check["warnings"],
                              "error": check["error"], "model_calls": 0}, ensure_ascii=False, indent=2))
            return 0 if check["status"] == "ready" else 2
        except (ValueError, OSError) as exc:
            print(json.dumps(cli_failure("needs_input", "INPUT_INVALID", str(exc)), ensure_ascii=False, indent=2))
            return 2
    result = generate(raw, settings)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return exit_code(result)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())

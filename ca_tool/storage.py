"""Atomic files and unique directories; hashes provide integrity, not signatures."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4


def canonical_hash(value) -> str:
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ_") + uuid4().hex[:12]


class RunStorage:
    def __init__(self, runs_dir: Path | str):
        self.runs_dir = Path(runs_dir)

    def create_run(self, run_id: str) -> Path:
        path = self.runs_dir / run_id
        path.mkdir(parents=True, exist_ok=False)
        return path.resolve()

    def write_json(self, run_dir: Path, name: str, value) -> None:
        if Path(name).name != name:
            raise ValueError("Run filenames must be plain basenames")
        payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        fd, temporary = tempfile.mkstemp(prefix=".pending_", suffix=".json", dir=run_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, run_dir / name)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def finalize(self, run_dir: Path, result: dict, versions: dict) -> None:
        self.write_json(run_dir, "result.json", result)
        files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(run_dir.glob("*.json")) if p.name != "manifest.json"}
        self.write_json(run_dir, "manifest.json", {
            "schema_version": "1.0.0", "run_id": result["run_id"],
            "source": result["metadata"]["source"], "status": result["status"],
            "versions": versions, "files": files, "recorded_at": utc_now(),
        })

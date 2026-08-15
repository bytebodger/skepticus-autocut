"""Stage caching: every stage writes a ``.done`` file holding a hash of its
inputs. Re-running skips any stage whose input hash is unchanged.

The unit of cache invalidation is the *input hash*, not file mtimes — mtimes are
unreliable across machines and git checkouts. An input hash is computed from
whatever the stage declares as its inputs (upstream file hashes + parameters).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DONE_FILE = ".done"


def hash_file(path: Path, *, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes. Used to detect changed raw drops / upstream."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def hash_inputs(inputs: dict[str, Any]) -> str:
    """Stable hash of a dict of input descriptors (paths already reduced to
    hashes, plus scalar parameters). Key order does not matter."""
    blob = json.dumps(inputs, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _done_path(stage_dir: Path) -> Path:
    return stage_dir / DONE_FILE


def is_current(stage_dir: Path, input_hash: str) -> bool:
    """True if the stage's ``.done`` exists and records this exact input hash."""
    done = _done_path(stage_dir)
    if not done.exists():
        return False
    try:
        recorded = json.loads(done.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return recorded.get("input_hash") == input_hash


def mark_done(stage_dir: Path, input_hash: str, extra: dict[str, Any] | None = None) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"input_hash": input_hash}
    if extra:
        payload.update(extra)
    _done_path(stage_dir).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def invalidate(stage_dir: Path) -> None:
    done = _done_path(stage_dir)
    if done.exists():
        done.unlink()

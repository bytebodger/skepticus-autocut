"""EDL schema, validation, and source<->output time mapping.

This module is small and load-bearing. The single most likely bug in the whole
pipeline is emitting output-timebase values where source-timebase is required
(see spec section 5). Everything here is written to make that bug impossible to
introduce silently, and the unit tests guard it.

**All times in an EDL are in the SOURCE timebase** — the moment in the raw
recording. ``source_to_output`` converts to the output timeline after cuts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1

VALID_ACTIONS = {"keep", "drop"}


# --------------------------------------------------------------------------- #
# Time mapping — the core of the source<->output contract.
# --------------------------------------------------------------------------- #

Span = tuple[float, float, float]  # (src_in, src_out, out_start)


def build_time_map(segments: Iterable[dict[str, Any]]) -> list[Span]:
    """Return ``(src_in, src_out, out_start)`` for each kept segment, in order.

    Segments are consumed in list order (the EDL is authored in source order).
    ``out_start`` is the cumulative duration of preceding kept segments.
    """
    spans: list[Span] = []
    cursor = 0.0
    for s in segments:
        if s.get("action") != "keep":
            continue
        src_in = float(s["in"])
        src_out = float(s["out"])
        spans.append((src_in, src_out, cursor))
        cursor += src_out - src_in
    return spans


def source_to_output(t: float, spans: list[Span]) -> float | None:
    """Map a source timestamp to the output timeline.

    Returns ``None`` if ``t`` falls inside a dropped region (was cut). The
    half-open interval ``[src_in, src_out)`` matches how ffmpeg's ``-ss/-to``
    treats segment boundaries.
    """
    for src_in, src_out, out_start in spans:
        if src_in <= t < src_out:
            return out_start + (t - src_in)
    return None


def output_duration(spans: list[Span]) -> float:
    """Total duration of the cut timeline."""
    if not spans:
        return 0.0
    src_in, src_out, out_start = spans[-1]
    return out_start + (src_out - src_in)


def snap(t: float, fps: float) -> float:
    """Snap a timestamp to the nearest frame boundary. Cuts must land on frames."""
    return round(t * fps) / fps


# --------------------------------------------------------------------------- #
# Load / save
# --------------------------------------------------------------------------- #

def load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(edl: dict[str, Any], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(edl, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

class ValidationError(Exception):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("EDL validation failed:\n  - " + "\n  - ".join(errors))


def validate(edl: dict[str, Any], *, source_duration: float | None = None) -> list[str]:
    """Return a list of human-readable errors (empty == valid).

    Checks structure, the source-timebase invariants, ordering, and — when a
    source duration is supplied — that no timestamp exceeds the recording.
    """
    errors: list[str] = []

    if edl.get("version") != SCHEMA_VERSION:
        errors.append(f"version must be {SCHEMA_VERSION}, got {edl.get('version')!r}")

    fps = edl.get("fps")
    if not isinstance(fps, (int, float)) or fps <= 0:
        errors.append(f"fps must be a positive number, got {fps!r}")

    # Tolerance for the out-vs-duration check: the final keep lands on the last
    # frame boundary and its 3dp-rounded time can sit a sub-millisecond above a
    # 6dp source duration. Anything within one frame is frame-snap/rounding, not
    # a real overshoot. Fall back to a small epsilon if fps is unusable.
    one_frame = (1.0 / fps) if isinstance(fps, (int, float)) and fps > 0 else 1e-6

    segments = edl.get("segments")
    if not isinstance(segments, list) or not segments:
        errors.append("segments must be a non-empty list")
        # Can't continue meaningfully without segments.
        return errors

    seen_ids: set[str] = set()
    prev_out: float | None = None
    kept = 0
    for idx, s in enumerate(segments):
        loc = f"segment[{idx}] ({s.get('id', '?')})"
        sid = s.get("id")
        if not sid:
            errors.append(f"{loc}: missing id")
        elif sid in seen_ids:
            errors.append(f"{loc}: duplicate id {sid!r}")
        else:
            seen_ids.add(sid)

        action = s.get("action")
        if action not in VALID_ACTIONS:
            errors.append(f"{loc}: action must be one of {sorted(VALID_ACTIONS)}, got {action!r}")

        in_t, out_t = s.get("in"), s.get("out")
        if not isinstance(in_t, (int, float)) or not isinstance(out_t, (int, float)):
            errors.append(f"{loc}: 'in' and 'out' must be numbers")
            continue
        if out_t <= in_t:
            errors.append(f"{loc}: out ({out_t}) must be > in ({in_t})")
        if in_t < 0:
            errors.append(f"{loc}: in ({in_t}) is negative")
        if source_duration is not None and out_t > source_duration + one_frame + 1e-6:
            overshoot = out_t - source_duration
            msg = (
                f"{loc}: out ({out_t}) exceeds source duration ({source_duration}) "
                f"by {overshoot:.3f}s, more than one frame ({one_frame:.3f}s)."
            )
            # A sub-frame overshoot is just frame-snap/rounding (tolerated above).
            # Only a gross overshoot points at an output-timebase (post-cut)
            # value having leaked into the EDL — don't cry wolf otherwise.
            if overshoot > max(1.0, 10 * one_frame):
                msg += " Likely an output-timebase value leaked into the EDL."
            errors.append(msg)
        # Segments must tile in source order (no overlaps, no going backwards).
        if prev_out is not None and in_t + 1e-6 < prev_out:
            errors.append(
                f"{loc}: in ({in_t}) is before previous segment's out ({prev_out}); "
                "segments must be ordered and non-overlapping in source time"
            )
        prev_out = out_t

        if action == "drop" and "confidence" not in s and s.get("reason") == "retake":
            errors.append(f"{loc}: retake drops must carry a confidence score (spec section 5)")

        if action == "keep":
            kept += 1

    if kept == 0:
        errors.append("no segments are kept — output would be empty")

    spans = build_time_map(segments)

    # Overlays: source_time must land inside a KEPT segment, otherwise the card
    # would fire on a moment that got cut. This is the guard against the
    # output-timebase bug for overlays.
    for idx, ov in enumerate(edl.get("overlays", []) or []):
        loc = f"overlay[{idx}] ({ov.get('id', '?')})"
        st = ov.get("source_time")
        if not isinstance(st, (int, float)):
            errors.append(f"{loc}: source_time must be a number")
            continue
        if source_duration is not None and st > source_duration + 1e-6:
            errors.append(
                f"{loc}: source_time ({st}) exceeds source duration ({source_duration}); "
                "overlay times must be in the SOURCE timebase"
            )
        if source_to_output(st, spans) is None:
            errors.append(
                f"{loc}: source_time ({st}) falls in a dropped region — the overlay "
                "would never appear. Move it into a kept segment."
            )
        if not ov.get("composition"):
            errors.append(f"{loc}: missing composition name")

    return errors


def validate_or_raise(edl: dict[str, Any], *, source_duration: float | None = None) -> None:
    errors = validate(edl, source_duration=source_duration)
    if errors:
        raise ValidationError(errors)


# --------------------------------------------------------------------------- #
# Override preservation — re-running stage 3 must never clobber a human veto.
# --------------------------------------------------------------------------- #

def collect_overrides(edl: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map segment id -> segment, for segments carrying ``override: true``.

    The review gate sets ``action: keep`` + ``override: true`` on vetoed drops.
    """
    return {
        s["id"]: s
        for s in edl.get("segments", [])
        if s.get("override") and "id" in s
    }


def apply_overrides(new_edl: dict[str, Any], overrides: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Re-apply human overrides onto a freshly authored EDL.

    Any segment whose id matches an override is forced back to the human's
    decision (``action`` + ``override`` flag preserved). Segments not present in
    the new EDL are ignored — the human vetoed a drop that no longer exists.
    """
    if not overrides:
        return new_edl
    for s in new_edl.get("segments", []):
        ov = overrides.get(s.get("id"))
        if ov is not None:
            s["action"] = ov.get("action", "keep")
            s["override"] = True
    return new_edl

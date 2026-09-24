"""``python -m autocut go <episode>`` — the full inbox-to-composite chain.

Auto-detects reaction vs monologue format (presence of
``inbox/<ep>_source.*``) and runs every stage for that format in order,
failing fast and loudly on the first error, with a live per-stage progress
line and a final summary (timings, cut breakdown, verifier results, output).

This is orchestration only — every stage is the same function the
individual CLI commands call, so caching, ``--force``, and ``--dry-run`` all
behave exactly as they do when run by hand.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from . import analyze, edl
from .paths import Episode

log = logging.getLogger("autocut.pipeline")

HEARTBEAT_SECONDS = 20.0
LOW_CONFIDENCE = 0.8  # matches review.py's / qc.py's "needs a listen" threshold


class StageError(RuntimeError):
    """A named stage failed; the run stops immediately (fail fast, fail loud)."""

    def __init__(self, stage: str, cause: BaseException):
        super().__init__(f"stage {stage!r} failed: {cause}")
        self.stage = stage
        self.cause = cause


def _fmt_dur(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h:d}:{int(m):02d}:{s:04.1f}" if h else f"{int(m):02d}:{s:04.1f}"


def _fmt_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


class _Heartbeat:
    """Prints a 'still running' line every ``HEARTBEAT_SECONDS`` while a
    stage's synchronous ffmpeg/whisper call blocks — the run can sit for
    minutes with no output otherwise, and there is no way to tell a slow
    stage from a hung one."""

    def __init__(self, name: str):
        self.name = name
        self._stop = threading.Event()
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(HEARTBEAT_SECONDS):
            print(f"    ... {self.name} still running ({_fmt_dur(time.monotonic() - self._t0)})",
                  flush=True)

    def __enter__(self) -> "_Heartbeat":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


class Runner:
    """Runs named stages in sequence, printing progress and recording timing.
    Any exception from a stage is wrapped in ``StageError`` naming the stage
    and re-raised immediately — nothing downstream ever runs after a failure.
    """

    def __init__(self, ep: Episode):
        self.ep = ep
        self.timings: dict[str, float] = {}
        self.t_start = time.monotonic()

    def _elapsed(self) -> str:
        return _fmt_dur(time.monotonic() - self.t_start)

    def stage(self, name: str, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        print(f"[{self._elapsed()}] -> {name}...", flush=True)
        t0 = time.monotonic()
        try:
            with _Heartbeat(name):
                result = fn(*args, **kwargs)
        except Exception as e:
            elapsed = time.monotonic() - t0
            print(f"[{self._elapsed()}] x {name} FAILED after {_fmt_dur(elapsed)}: {e}",
                  file=sys.stderr, flush=True)
            raise StageError(name, e) from e
        elapsed = time.monotonic() - t0
        self.timings[name] = elapsed
        print(f"[{self._elapsed()}] OK {name} ({_fmt_dur(elapsed)})", flush=True)
        return result


def _flagged_drop_count(edl_doc: dict, threshold: float = LOW_CONFIDENCE) -> int:
    return sum(1 for s in edl_doc.get("segments", [])
              if s.get("action") == "drop" and float(s.get("confidence", 1.0)) < threshold)


def _drop_seconds_by_reason(edl_doc: dict) -> dict[str, float]:
    totals: dict[str, float] = {}
    for s in edl_doc.get("segments", []):
        if s.get("action") == "drop":
            reason = s.get("reason", "unspecified")
            totals[reason] = totals.get(reason, 0.0) + (float(s["out"]) - float(s["in"]))
    return totals


def _playback_split(ep: Episode) -> tuple[float, float] | None:
    """(playback seconds, commentary seconds), or ``None`` if either the
    playback map or the probe (for total duration) is missing."""
    if not ep.playback_json.exists() or not ep.probe_json.exists():
        return None
    playback = json.loads(ep.playback_json.read_text(encoding="utf-8"))
    probe = json.loads(ep.probe_json.read_text(encoding="utf-8"))
    total = float(probe.get("source_duration") or 0.0)
    play = sum(s["host_out"] - s["host_in"] for s in playback.get("segments", []))
    return play, max(0.0, total - play)


def _print_summary(ep: Episode, r: Runner, *, is_reaction: bool, did_review: bool,
                   flagged: int, vetoed: int, sync_report: dict | None,
                   cue_report: dict | None, output_path: Path) -> None:
    total = time.monotonic() - r.t_start
    print()
    print("=" * 64)
    print(f"go: {ep.episode_id} summary")
    print("=" * 64)
    print(f"total runtime: {_fmt_dur(total)}")
    print("per-stage:")
    for name, secs in r.timings.items():
        print(f"  {name:<12} {_fmt_dur(secs)}")

    if is_reaction:
        split = _playback_split(ep)
        if split:
            play, commentary = split
            denom = play + commentary
            pct = (play / denom * 100) if denom else 0.0
            print(f"playback vs commentary: {_fmt_dur(play)} playback / "
                 f"{_fmt_dur(commentary)} commentary ({pct:.0f}% playback)")

    edl_doc = edl.load(ep.edl_json)
    by_reason = _drop_seconds_by_reason(edl_doc)
    if by_reason:
        print("seconds cut by reason:")
        for reason, secs in sorted(by_reason.items(), key=lambda kv: -kv[1]):
            print(f"  {reason:<12} {secs:.1f}s")

    if did_review:
        print(f"review: {flagged} low-confidence drop(s) were flagged; you vetoed {vetoed} of them")
    else:
        print(f"review: skipped (--review not passed) -- {flagged} low-confidence drop(s) "
             f"would have been flagged")

    if is_reaction:
        n_segments = len(json.loads(ep.playback_json.read_text(encoding="utf-8"))["segments"])
        print(f"alternation: clean ({n_segments} playback segment(s) -- align would have "
             f"failed loudly otherwise)")
        if sync_report is not None:
            offsets = [abs(s["median_offset"]) for s in sync_report["segments"]
                      if s["median_offset"] is not None]
            if offsets:
                print(f"sync-check: worst offset {max(offsets):+.3f}s "
                     f"({len(offsets)}/{len(sync_report['segments'])} segment(s) trusted)")
            else:
                print("sync-check: no in-window segment had enough matched words to trust")
        if cue_report is not None:
            print(f"cue-check: {len(cue_report['violations'])} violation(s) "
                 f"({cue_report['checked']} transition(s) checked, {cue_report['skipped']} out of window)")

    if output_path.exists():
        print(f"output: {output_path} ({_fmt_size(output_path.stat().st_size)})")
    else:
        print(f"output: {output_path} (MISSING)")
    print("=" * 64)


def run_go(ep: Episode, *, force: bool = False, review: bool = False, preview: bool = False) -> int:
    from . import probe as probe_mod, transcribe as transcribe_mod

    is_reaction = ep.source_video.exists()
    print(f"go: {ep.episode_id} -- {'reaction' if is_reaction else 'monologue'} format detected"
         f"{' (found ' + ep.source_video.name + ')' if is_reaction else ''}"
         f"{', preview mode' if preview else ''}")

    r = Runner(ep)

    r.stage("probe", probe_mod.run, ep, force=force)
    r.stage("transcribe", transcribe_mod.run, ep, force=force)

    if is_reaction:
        from . import align as align_mod
        r.stage("align", align_mod.run, ep, force=force)

    r.stage("autoauthor", analyze.autoauthor, ep)

    errors = r.stage("validate", analyze.validate, ep)
    if errors:
        print(f"go: EDL INVALID ({len(errors)} error(s)):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        raise StageError("validate", RuntimeError(f"{len(errors)} error(s) -- see above"))

    edl_doc = edl.load(ep.edl_json)
    flagged = _flagged_drop_count(edl_doc)
    vetoed = 0

    if review:
        from . import review as review_mod
        print(f"go: {flagged} low-confidence drop(s) flagged -- review gate open at "
             f"http://127.0.0.1:8765 -- Ctrl+C here when you're done to continue")
        t0 = time.monotonic()
        try:
            review_mod.serve(ep.episode_id, root=ep.root)
        except KeyboardInterrupt:
            pass
        r.timings["review"] = time.monotonic() - t0
        edl_doc = edl.load(ep.edl_json)  # reload: review may have written vetoes
        vetoed = sum(1 for s in edl_doc.get("segments", []) if s.get("override"))
        print(f"go: review closed after {_fmt_dur(r.timings['review'])} ({vetoed} veto(es)), continuing")
    else:
        print(f"go: --review not passed -- proceeding unattended "
             f"({flagged} low-confidence drop(s) would have been flagged)")

    sync_report: dict | None = None
    cue_report: dict | None = None

    if is_reaction:
        from . import compose as compose_mod
        r.stage("compose", compose_mod.run, ep, force=force, preview=preview, render_range=None)
        output_path = ep.compose_output

        from . import cuecheck as cuecheck_mod, sync_check as sync_check_mod
        window = (0.0, compose_mod.PREVIEW_SECONDS) if preview else None
        sync_report = r.stage("sync-check", sync_check_mod.run, ep, window=window)
        cue_report = r.stage("cue-check", cuecheck_mod.run, ep)
    else:
        if preview:
            print("go: --preview has no windowed render path for the monologue chain; "
                 "rendering in full", file=sys.stderr)
        from . import (
            captions as captions_mod,
            composite as composite_mod,
            cut as cut_mod,
            grade as grade_mod,
            overlays as overlays_mod,
            qc as qc_mod,
        )
        r.stage("cut", cut_mod.run, ep, force=force)
        r.stage("grade", grade_mod.run, ep, force=force)
        r.stage("captions", captions_mod.run, ep, force=force)
        r.stage("overlays", overlays_mod.run, ep, force=force)
        r.stage("composite", composite_mod.run, ep, force=force)
        r.stage("qc", qc_mod.run, ep)
        output_path = ep.output_mp4

    _print_summary(ep, r, is_reaction=is_reaction, did_review=review, flagged=flagged,
                  vetoed=vetoed, sync_report=sync_report, cue_report=cue_report,
                  output_path=output_path)
    return 0

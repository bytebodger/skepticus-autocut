"""Stage 9 — QC report.

Emits ``out/<ep>_report.md`` and a contact sheet (one frame per cut boundary) so
bad splices are eyeballable. Records the ffmpeg build for determinism auditing.
"""

from __future__ import annotations

import json
import logging
from collections import Counter

from . import edl, ffmpeg
from .paths import Episode

log = logging.getLogger("autocut.qc")


def _words_in(words: list[dict], start: float, end: float) -> str:
    return " ".join(w["word"] for w in words if start <= w["start"] < end).strip()


def _fmt(t: float) -> str:
    m, s = divmod(t, 60)
    h, m = divmod(int(m), 60)
    return f"{h:d}:{int(m):02d}:{s:06.3f}"


def _boundary_frames(spans: list[edl.Span], fps: float) -> list[int]:
    """Output frame numbers where two kept segments meet (a splice)."""
    frames = []
    for _src_in, _src_out, out_start in spans[1:]:
        frames.append(int(round(out_start * fps)))
    return frames


def _contact_sheet(ep: Episode, frames: list[int]) -> None:
    if not frames:
        log.info("qc: no cut boundaries; skipping contact sheet")
        return
    select = "+".join(f"eq(n,{n})" for n in frames)
    n = len(frames)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    ffmpeg.run_ffmpeg([
        "-i", ep.output_mp4,
        "-vf", f"select='{select}',scale=320:-1,tile={cols}x{rows}",
        "-frames:v", "1",
        ep.contact_sheet,
    ])


def run(ep: Episode) -> str:
    """Build the QC report. Returns the report text."""
    edl_doc = edl.load(ep.edl_json)
    probe = json.loads(ep.probe_json.read_text(encoding="utf-8")) if ep.probe_json.exists() else {}
    words = []
    if ep.words_json.exists():
        words = json.loads(ep.words_json.read_text(encoding="utf-8")).get("words", [])

    segments = edl_doc["segments"]
    fps = float(edl_doc["fps"])
    spans = edl.build_time_map(segments)
    out_dur = edl.output_duration(spans)
    src_dur = float(probe.get("source_duration") or (segments[-1]["out"] if segments else 0.0))

    drops = [s for s in segments if s["action"] == "drop"]
    by_reason = Counter(s.get("reason", "unspecified") for s in drops)
    low_conf = [s for s in drops if s.get("confidence", 1.0) < 0.8]

    lines: list[str] = []
    lines.append(f"# QC report — {ep.episode_id}\n")
    lines.append("## Duration")
    lines.append(f"- Source: {_fmt(src_dur)}")
    lines.append(f"- Output: {_fmt(out_dur)}")
    lines.append(f"- Removed: {_fmt(src_dur - out_dur)} "
                 f"({(src_dur - out_dur) / src_dur * 100:.1f}%)" if src_dur else "- Removed: n/a")
    lines.append("")

    lines.append("## Cuts")
    lines.append(f"- Total drops: {len(drops)}")
    for reason, count in sorted(by_reason.items()):
        lines.append(f"  - {reason}: {count}")
    lines.append("")

    lines.append("## Low-confidence drops (< 0.8)")
    if not low_conf:
        lines.append("- none")
    for s in low_conf:
        text = _words_in(words, s["in"], s["out"])
        lines.append(
            f"- `{s['id']}` [{_fmt(s['in'])}–{_fmt(s['out'])}] "
            f"conf={s.get('confidence')} reason={s.get('reason')}: \"{text}\""
        )
    lines.append("")

    lines.append("## Overlays (resolved output times)")
    overlays = edl_doc.get("overlays", []) or []
    if not overlays:
        lines.append("- none")
    for ov in overlays:
        out_t = edl.source_to_output(ov["source_time"], spans)
        out_s = _fmt(out_t) if out_t is not None else "CUT (never appears!)"
        lines.append(
            f"- `{ov['id']}` {ov['composition']} @ src {_fmt(ov['source_time'])} "
            f"-> out {out_s} (dur {ov.get('duration', 4.0)}s)"
        )
    lines.append("")

    lines.append("## Determinism")
    lines.append(f"- ffmpeg: `{ffmpeg.ffmpeg_version()}`")
    lines.append(f"- raw sha256: `{probe.get('raw_sha256', 'n/a')}`")
    lines.append(f"- fps: {fps}")
    lines.append("")

    report = "\n".join(lines)
    ep.out_dir.mkdir(parents=True, exist_ok=True)
    ep.report_md.write_text(report, encoding="utf-8")
    log.info("qc: wrote %s", ep.report_md)

    if ep.output_mp4.exists() or ffmpeg.is_dry_run():
        try:
            _contact_sheet(ep, _boundary_frames(spans, fps))
        except ffmpeg.FFmpegError as e:
            log.warning("qc: contact sheet failed: %s", e)

    return report

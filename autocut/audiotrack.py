"""Phase-2 compositor — reaction audio track (reaction spec section 8).

Only produces anything for reaction episodes (a ``playback.json`` from the
align stage). Assembles a full-window audio track that switches between the
host mic (commentary) and the source file's own clean audio (playback),
muting the host mic completely — not ducking — during playback, crossfading
every boundary, and level-matching the two sources. Built as its own cached
stage producing a full-window track (reaction spec: "same structure as the
content track"); ``compose.py`` mixes it into the composite instead of the
raw speaker-source audio when it exists.

**The host bed is built from the EDL's kept source spans, not a naive
output-time seek into the raw mezzanine.** Output time and source time only
coincide at the very start of the episode; every prior drop (dead air,
filler, retakes, the cues themselves) shifts them apart. Reading the raw
mezzanine with ``-ss <output window start>`` is only correct until the first
drop before the window — after that, the host bed plays back content from
the wrong moment, and the cue-driven mute boundaries (computed correctly in
output time) land on the wrong audio. This mirrors exactly what a real
``cut.mkv`` would contain for the window, without paying for a full-episode
cut: concatenate the kept EDL spans overlapping the window (there are only a
handful per window, not the whole episode), each independently source-time
trimmed.

Two tracks are then built and summed, mirroring how ``content.py`` overlays
timed items on a transparent base instead of hard concatenation:

- **Host bed**: the concatenated kept-span audio above, gated to silence
  during each in-window playback span by a single ``volume=eval=frame``
  expression — a piecewise envelope (1 outside playback, 0 inside, short
  linear ramps at each edge) evaluated once over the whole stream. NOT a
  chain of ``afade`` filters: ``afade=t=in`` silences the *entire* stream
  before its own ``start_time`` (confirmed empirically, contrary to
  ``afade=t=out``'s documented "unaffected before start_time" behaviour) —
  chaining one after an out-fade to "restore" the host mid-stream silenced
  the whole track, including everything before either fade. A single
  expression sidesteps that filter entirely.
- **Source clips**: one ffmpeg input per in-window playback segment, trimmed
  to its own ``[source_in, source_out)``, gain-matched, faded at its own
  edges via plain ``afade`` (each clip is freshly decoded from its own local
  t=0, so the type=in gotcha above never applies — there's nothing before
  its own start to wrongly silence), delayed into its window-relative output
  position, and padded with silence to the full window length.

Both envelopes touch (rather than overlap) at each playback boundary — the
host's ramp down ends exactly where a segment's source clip fades in from
silence, and symmetrically at the far edge — which reads as a soft crossfade
without ever letting host and source sound simultaneously outside the
configured window (source audio bleeding into the wrong instant would desync
the pairing the lip-sync check exists to verify).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

from . import audio as audio_mod, cache, edl, ffmpeg
from .paths import Episode

log = logging.getLogger("autocut.audiotrack")

SAMPLE_RATE = 48000
DEFAULT_CROSSFADE_MS = 75.0
DEFAULT_SOURCE_GAIN_DB = 0.0
GAIN_CLAMP_DB = 24.0  # sanity bound on the measured level-matching offset
SPLICE_FADE = 0.025   # matches cut.py's FADE — inaudible, kills the concat-seam click

# ffmpeg's own -ss/-to INPUT-option trim silently drops a few ms per clip when
# the stream is DECODED (any filter graph, as every host-bed clip is here) —
# confirmed empirically: measured against a from-file-start, no-seek reference
# decode, a plain "-ss X -to Y" read loses ~2-10ms per clip regardless of clip
# length (an artifact of decode-based accurate-seek trimming, not proportional
# to duration). The video concat (compose.py's _host_video_concat_lines) does
# NOT show this loss on this project's all-intra, exactly-CFR mezzanine, so
# only the audio side silently drifts — and because the loss is per CUT, not
# per elapsed second, it accumulates with every kept-span boundary crossed
# (measured: ~30ms by output 60s, ~500ms by the end of an 83-span episode),
# which reads as the speaker video progressively drifting out of sync with
# its own voice. The fix: seek coarsely, then cut exactly with an `atrim`
# filter on the real decoded samples, using bounds RELATIVE to the seek point
# (ffmpeg rebases decoded timestamps to ~0 at the seek by default) — verified
# byte-identical to a from-file-start reference decode regardless of seek
# margin, at every one of a sampled 11 spans spanning the whole episode.
#
# An earlier version of this fix used `-copyts` (absolute source timestamps
# instead of the ~0 rebase) with absolute atrim bounds. That broke the source
# playback clips sharing this same ffmpeg invocation: -copyts is NOT scoped
# to the input it's written before — it's a sticky global flag in ffmpeg's
# CLI parser, so it silently applied to every later input too (confirmed via
# ashowinfo: a source-clip input with no -copyts of its own still decoded
# with its true absolute file timestamp). Those clips are placed on the
# output timeline by their (near-zero, seek-relative) PTS via `adelay` — with
# an absolute source-file PTS instead, they landed far outside the render
# window and were silently dropped, which read as playback audio going
# silent. The relative-bounds approach below never touches -copyts, so nothing
# leaks into unrelated inputs built elsewhere in the same command.
SEEK_MARGIN = 1.0


def _coarse_seek(source_in: float) -> float:
    return max(0.0, source_in - SEEK_MARGIN)


def host_audio_input_args(host_av: Path, source_in: float, source_out: float) -> list[str]:
    """ffmpeg input args for one kept span's audio: seek to a safely-early
    point and read a generously bounded window — see ``SEEK_MARGIN``'s
    docstring. The exact cut happens in the filter graph
    (``host_bed_filter_lines``'s ``atrim``), not here."""
    coarse_ss = _coarse_seek(source_in)
    read_dur = (source_out - coarse_ss) + SEEK_MARGIN
    return ["-ss", f"{coarse_ss:.3f}", "-t", f"{read_dur:.3f}", "-i", str(host_av)]


def _audio_config(ep: Episode) -> dict:
    cfg_path = ep.root / "config" / "layout.yaml"
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    reaction = (data or {}).get("reaction", {}) or {}
    return {
        "crossfade_s": float(reaction.get("crossfade_ms", DEFAULT_CROSSFADE_MS)) / 1000.0,
        "match_levels": bool(reaction.get("match_levels", True)),
        "source_gain_db": float(reaction.get("source_gain_db", DEFAULT_SOURCE_GAIN_DB)),
    }


def _host_av_source(ep: Episode) -> Path:
    """The host media the mic bed is built from — same preference as
    align._host_av_source (the mezzanine shares the source/word timeline)."""
    if ep.mezz.exists():
        return ep.mezz
    if ep.raw.exists():
        return ep.raw
    raise FileNotFoundError(
        f"No host media for the audio track (neither {ep.mezz} nor {ep.raw}). "
        f"Run 'autocut probe {ep.episode_id}' first."
    )


def _windowed_keep_spans(spans: list, window: tuple) -> list[dict]:
    """``edl.windowed_keep_spans`` as the ``{source_in, source_out}`` dicts
    this module's graph-building code expects."""
    return [{"source_in": s, "source_out": e} for s, e in edl.windowed_keep_spans(spans, window)]


def host_bed_filter_lines(keep_spans: list[dict], *, out_label: str = "hostbed",
                          sample_rate: int = SAMPLE_RATE) -> tuple[list[str], str]:
    """Filter-graph lines that concatenate ``keep_spans`` — ffmpeg inputs
    ``[0:a]..[len(keep_spans)-1:a]``, in order — into one continuous,
    resampled stream. Each input must be built with ``host_audio_input_args``
    (a coarse seek, not an exact ``-ss/-to``) — the ``atrim`` here does the
    precise cut, using bounds relative to that same coarse seek point; see
    ``SEEK_MARGIN``'s docstring for why. A small splice fade at each seam
    kills the concat click (matches cut.py's own FADE convention). Returns
    ``(lines, out_label)``.

    Shared with compose.py's monologue audio passthrough: reading raw host
    media for an output window has the exact same source-time/output-time
    problem the reaction host bed does (see module docstring) — this is the
    one place that builds it correctly, so both stages call it rather than
    each growing their own copy.
    """
    lines: list[str] = []
    labels = []
    for i, ks in enumerate(keep_spans):
        s_in, s_out = ks["source_in"], ks["source_out"]
        dur = s_out - s_in
        fade = min(SPLICE_FADE, dur / 2)
        label = f"hk{i}"
        # atrim+asetpts does the real, exact cut (see SEEK_MARGIN's docstring);
        # the input was only seeked coarsely, so this is where precision
        # lives. Bounds are relative to that same coarse seek point — must
        # match host_audio_input_args' own _coarse_seek exactly.
        coarse_ss = _coarse_seek(s_in)
        rel_start, rel_end = s_in - coarse_ss, s_out - coarse_ss
        lines.append(
            f"[{i}:a]atrim=start={rel_start:.6f}:end={rel_end:.6f},asetpts=PTS-STARTPTS,"
            f"aresample={sample_rate},aformat=channel_layouts=stereo,"
            f"afade=t=in:st=0:d={fade:.3f},afade=t=out:st={max(0.0, dur - fade):.3f}:d={fade:.3f}[{label}]"
        )
        labels.append(label)
    if len(labels) == 1:
        lines.append(f"[{labels[0]}]anull[{out_label}]")
    else:
        lines.append(f"{''.join(f'[{l}]' for l in labels)}concat=n={len(labels)}:v=0:a=1[{out_label}]")
    return lines, out_label


def _windowed_segments(spans: list, playback_segments: list[dict], window: tuple) -> list[dict]:
    """Playback segments clipped to the output window, with their window-
    relative output position and whether each edge is a real boundary (get a
    fade) or a window clip (already mid-span, no fade to add)."""
    w0, length = window
    out: list[dict] = []
    for seg in playback_segments:
        out_start = edl.source_to_output(seg["host_in"], spans)
        if out_start is None:
            log.warning("audiotrack: %s host_in %.3f falls in a cut region; skipping",
                        seg.get("id"), seg["host_in"])
            continue
        out_len = seg["host_out"] - seg["host_in"]
        out_end = out_start + out_len
        if out_end <= w0 or out_start >= w0 + length:
            continue
        clipped_start = max(out_start, w0)
        clipped_end = min(out_end, w0 + length)
        trimmed_head = clipped_start - out_start
        out.append({
            "window_rel_start": clipped_start - w0,
            "clip_dur": clipped_end - clipped_start,
            "source_in": seg["source_in"] + trimmed_head,
            "fade_in": abs(trimmed_head) < 1e-6,
            "fade_out": abs(clipped_end - out_end) < 1e-6,
        })
    out.sort(key=lambda s: s["window_rel_start"])
    return out


def _measure_gain_db(ep: Episode, host_av: Path, source_path: Path,
                     keep_spans: list[dict], segments: list[dict], cfg: dict) -> float:
    """Level-matching offset (reaction spec section 8): "measure both and
    apply a gain offset" so the source's natural loudness matches the host
    mic's. Measured once per render from the first kept EDL span's host audio
    and the first in-window segment's source audio — a representative
    sample, not a per-segment analysis (which would cost one ffmpeg pass per
    segment). The host measurement is taken in SOURCE time (the kept span),
    not the output window — ``host_av`` is raw, uncut media, and the window
    is an output-time range (see edl.windowed_keep_spans)."""
    total = cfg["source_gain_db"]
    if not cfg["match_levels"] or not segments or not keep_spans or ffmpeg.is_dry_run():
        return total
    first_keep = keep_spans[0]
    host_measured = audio_mod.analyze(
        ep, str(host_av), (first_keep["source_in"], first_keep["source_out"] - first_keep["source_in"]), {})
    first = segments[0]
    source_measured = audio_mod.analyze(
        ep, str(source_path), (first["source_in"], first["clip_dur"]), {})
    if host_measured is None or source_measured is None:
        log.warning("audiotrack: could not measure levels; using source_gain_db only")
        return total
    offset = float(host_measured["input_i"]) - float(source_measured["input_i"])
    offset = max(-GAIN_CLAMP_DB, min(GAIN_CLAMP_DB, offset))
    log.info("audiotrack: level-matching offset %.1f dB (host %.1f LUFS, source %.1f LUFS)",
             offset, float(host_measured["input_i"]), float(source_measured["input_i"]))
    return total + offset


def _host_volume_expr(segments: list[dict], length: float, xfade: float) -> str:
    """Piecewise volume envelope for the host bed: 1.0 outside playback spans,
    0.0 inside, with a short linear ramp at each real (not window-clipped)
    edge. A single frame-eval expression, not a chain of ``afade`` filters —
    see the module docstring for why that silently breaks."""
    terms: list[tuple[float, float, str]] = []  # (lo, hi, value_expr), time-ordered
    for seg in segments:
        rel = seg["window_rel_start"]
        end_rel = rel + seg["clip_dur"]
        if seg["fade_in"] and rel > 1e-6:
            lo = max(0.0, rel - xfade)
            terms.append((lo, rel, f"(1-(t-{lo:.6f})/{xfade:.6f})"))
        terms.append((rel, end_rel, "0"))
        if seg["fade_out"] and end_rel < length - 1e-6:
            hi = min(length, end_rel + xfade)
            terms.append((end_rel, hi, f"((t-{end_rel:.6f})/{xfade:.6f})"))
    expr = "1"
    for lo, hi, val in reversed(terms):
        expr = f"if(between(t,{lo:.6f},{hi:.6f}),{val},{expr})"
    return expr


def _build_graph(keep_spans: list[dict], segments: list[dict], window: tuple,
                 xfade: float, gain_db: float) -> str:
    w0, length = window
    xfade = max(xfade, 0.001)  # ramp needs a positive duration; treat ~0 as effectively instant

    # Host bed: concatenate the kept EDL spans overlapping this window (each
    # its own ffmpeg input, indices [0, len(keep_spans))) into one
    # output-time-continuous stream, then apply the mute/crossfade envelope.
    lines, _ = host_bed_filter_lines(keep_spans, out_label="hostraw")

    host_expr = _host_volume_expr(segments, length, xfade)
    lines.append(f"[hostraw]volume=eval=frame:volume='{host_expr}'[hostbed]")

    src_base = len(keep_spans)
    src_labels = []
    for i, seg in enumerate(segments):
        dur = seg["clip_dur"]
        chain = [f"[{src_base + i}:a]aresample={SAMPLE_RATE}", "aformat=channel_layouts=stereo"]
        if abs(gain_db) > 1e-6:
            chain.append(f"volume={gain_db:.2f}dB")
        if seg["fade_in"]:
            chain.append(f"afade=t=in:st=0:d={min(xfade, dur):.3f}")
        if seg["fade_out"]:
            chain.append(f"afade=t=out:st={max(0.0, dur - xfade):.3f}:d={min(xfade, dur):.3f}")
        delay_ms = round(seg["window_rel_start"] * 1000)
        chain.append(f"adelay={delay_ms}|{delay_ms}")
        chain.append(f"apad=whole_dur={length:.3f}")
        label = f"src{i}"
        lines.append(",".join(chain) + f"[{label}]")
        src_labels.append(label)

    if not src_labels:
        lines.append("[hostbed]anull[mix]")
    else:
        inputs_txt = "".join(f"[{lbl}]" for lbl in ["hostbed", *src_labels])
        n = 1 + len(src_labels)
        lines.append(f"{inputs_txt}amix=inputs={n}:duration=longest:normalize=0[mix]")
    return ";\n".join(lines) + "\n"


def render_track(ep: Episode, window: tuple, *, force: bool = False) -> Path | None:
    """Render the reaction audio track for the output window, cached. Returns
    its path, or None for a non-reaction episode (no playback.json)."""
    if not ep.playback_json.exists():
        return None
    playback = json.loads(ep.playback_json.read_text(encoding="utf-8"))
    source_path = Path(playback["source_file"])
    host_av = _host_av_source(ep)
    cfg = _audio_config(ep)

    edl_doc = edl.load(ep.edl_json)
    spans = edl.build_time_map(edl_doc["segments"])
    keep_spans = _windowed_keep_spans(spans, window)
    if not keep_spans:
        log.warning("audiotrack: no kept EDL segments in this window; skipping audio track")
        return None

    segments = _windowed_segments(spans, playback.get("segments", []), window)
    gain_db = _measure_gain_db(ep, host_av, source_path, keep_spans, segments, cfg)
    graph = _build_graph(keep_spans, segments, window, cfg["crossfade_s"], gain_db)

    w0, length = window
    dry = ffmpeg.is_dry_run()
    stage_dir = ep.compose_dir / "audio_track"
    input_hash = cache.hash_inputs({
        "host": cache.hash_file(host_av) if (host_av.exists() and not dry) else "dry",
        "source": cache.hash_file(source_path) if (source_path.exists() and not dry) else "dry",
        "graph": graph,
        "gain_db": round(gain_db, 2),
        "window": [round(w0, 3), round(length, 3)],
    })
    if not force and cache.is_current(stage_dir, input_hash) and ep.audio_track.exists():
        log.info("audiotrack: track cache hit")
        return ep.audio_track

    ep.compose_dir.mkdir(parents=True, exist_ok=True)
    ep.audio_filter_script.write_text(graph, encoding="utf-8")
    inputs: list[str] = []
    for ks in keep_spans:
        inputs += host_audio_input_args(host_av, ks["source_in"], ks["source_out"])
    for seg in segments:
        inputs += ["-ss", f"{seg['source_in']:.3f}", "-t", f"{seg['clip_dur']:.3f}", "-i", str(source_path)]
    log.info("audiotrack: %d kept span(s), %d in-window playback segment(s), gain=%.1fdB, xfade=%.0fms -> %s",
             len(keep_spans), len(segments), gain_db, cfg["crossfade_s"] * 1000, ep.audio_track)
    ffmpeg.run_ffmpeg([
        *inputs,
        "-/filter_complex", ep.audio_filter_script,
        "-map", "[mix]",
        "-t", f"{length:.3f}",
        "-c:a", "pcm_s16le", "-ar", str(SAMPLE_RATE),
        ep.audio_track,
    ])
    cache.mark_done(stage_dir, input_hash, extra={"stage": "compose:audio_track"})
    return ep.audio_track

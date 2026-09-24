"""Regression guard: the host VIDEO concat (compose.py's speaker layer) and
the host AUDIO concat (audiotrack.py's host bed, shared with monologue
passthrough) must reconstruct the SAME total duration from the SAME kept EDL
spans.

They were found to silently disagree: ffmpeg's own ``-ss/-to`` INPUT-option
trim, applied to a DECODED (filtered) stream, drops a few ms per clip
regardless of clip length — confirmed against a from-file-start reference
decode. The video concat doesn't show this loss on an all-intra CFR
mezzanine, but the audio concat (which must decode to apply fades/gain/mute)
does, and it accumulates with every kept-span boundary crossed — invisible on
a short preview, and only audible as a growing speaker-window lip-sync drift
across a long episode (confirmed empirically: ~30ms by output 60s, ~500ms by
the end of an 83-span episode). Fixed by seeking coarsely (``-copyts``, so
decoded timestamps stay absolute) and cutting exactly with ``atrim`` on the
already-decoded samples (see audiotrack.py's ``SEEK_MARGIN`` docstring).

This test uses many short spans (as short as 0.1s — the real episode had
several under 0.2s) because the per-clip loss is roughly CONSTANT regardless
of clip duration: a handful of long clips wouldn't have exposed it the way
the real episode's many short cutaways did.
"""

import numpy as np
import pytest

from autocut import audiotrack, compose, edl, ffmpeg


def _make_av_source(path, duration=40.0, sr=48000):
    # Broadband noise, not a tone: a periodic signal would still correlate
    # highly with itself even if shifted by a few ms (aliasing on the period),
    # which would hide exactly the kind of small positional error this test
    # exists to catch (see test_refine.py's synthetic fixtures for the same
    # reasoning).
    # All-intra (-g 1), matching probe.py's real mezzanine: frame-accurate
    # trimming depends on every frame being independently seekable. A normal
    # GOP structure rounds short clips to a much coarser (multi-frame)
    # boundary and isn't representative of what compose.py actually reads.
    # Stereo, matching the real mezzanine's actual layout: a mono source
    # would get upmixed by host_bed_filter_lines' aformat=stereo, and mono
    # <-> stereo isn't a lossless round trip in ffmpeg (per-channel level
    # convention) — any later mono-downmix comparison would then diverge
    # from a reference that skipped that conversion, which is a test-fixture
    # artifact, not something this test is meant to be checking.
    ffmpeg.run_ffmpeg([
        "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=64x64:rate=24",
        "-f", "lavfi", "-i", f"anoisesrc=color=white:duration={duration}:sample_rate={sr}:seed=7",
        "-f", "lavfi", "-i", f"anoisesrc=color=white:duration={duration}:sample_rate={sr}:seed=13",
        "-map", "0:v", "-filter_complex", "[1:a][2:a]join=inputs=2:channel_layout=stereo[aout]",
        "-map", "[aout]", "-c:v", "libx264", "-g", "1", "-pix_fmt", "yuv420p",
        "-c:a", "pcm_s16le", str(path),
    ])


def _render_video_concat(host_av, keep_spans, out_path):
    inputs = []
    for s_in, s_out in keep_spans:
        inputs += ["-ss", f"{s_in:.3f}", "-to", f"{s_out:.3f}", "-i", str(host_av)]
    lines = compose._host_video_concat_lines(keep_spans, out_label="hv")
    graph = ";\n".join(lines) + "\n"
    ffmpeg.run_ffmpeg([
        *inputs, "-filter_complex", graph, "-map", "[hv]",
        "-r", "24", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path),
    ])


def _render_audio_concat(host_av, keep_spans, out_path):
    inputs = []
    for s_in, s_out in keep_spans:
        inputs += audiotrack.host_audio_input_args(host_av, s_in, s_out)
    keep_span_dicts = [{"source_in": s, "source_out": e} for s, e in keep_spans]
    lines, out_label = audiotrack.host_bed_filter_lines(keep_span_dicts, out_label="ha")
    graph = ";\n".join(lines) + "\n"
    ffmpeg.run_ffmpeg([
        *inputs, "-filter_complex", graph, "-map", f"[{out_label}]",
        "-c:a", "pcm_s16le", "-ar", "48000", str(out_path),
    ])


def _probe_duration(path):
    d = ffmpeg.ffprobe_json(["-show_entries", "format=duration", str(path)])
    return float(d["format"]["duration"])


# Many short spans (several under 0.2s) with drops between them, mirroring
# the real episode's mix of brief cutaways and longer commentary blocks.
# Boundaries are snapped to the 24fps frame grid, as edl.snap() always
# produces in real EDLs ("cuts must land on frames") — unsnapped boundaries
# make even the (already-correct) video concat round short clips by up to a
# full frame, which isn't the thing this test checks.
_FPS = 24.0
_EDL_RAW = [
    (0.0, 3.0, "keep"), (3.0, 3.3, "drop"),
    (3.3, 3.4, "keep"), (3.4, 3.6, "drop"),      # 0.1s
    (3.6, 3.75, "keep"), (3.75, 4.0, "drop"),    # 0.15s
    (4.0, 4.12, "keep"), (4.12, 4.3, "drop"),    # 0.12s
    (4.3, 9.0, "keep"), (9.0, 9.2, "drop"),
    (9.2, 9.3, "keep"), (9.3, 9.5, "drop"),      # 0.1s
    (9.5, 15.0, "keep"),
]
_EDL_SEGMENTS = [
    {"id": f"e{i}", "in": edl.snap(a, _FPS), "out": edl.snap(b, _FPS), "action": action,
     **({"reason": "silence"} if action == "drop" else {})}
    for i, (a, b, action) in enumerate(_EDL_RAW)
]


def test_host_video_and_audio_concats_produce_the_same_total_duration(tmp_path):
    host_av = tmp_path / "host.mkv"
    _make_av_source(host_av, duration=16.0)

    spans = edl.build_time_map(_EDL_SEGMENTS)
    keep_spans = edl.windowed_keep_spans(spans, (0.0, edl.output_duration(spans)))
    requested_total = sum(e - s for s, e in keep_spans)

    video_out = tmp_path / "video_concat.mkv"
    audio_out = tmp_path / "audio_concat.wav"
    _render_video_concat(host_av, keep_spans, video_out)
    _render_audio_concat(host_av, keep_spans, audio_out)

    video_dur = _probe_duration(video_out)
    audio_dur = _probe_duration(audio_out)

    # One video frame (1/24s) covers legitimate CFR frame-boundary rounding;
    # the bug this guards against was ~6ms of loss PER SHORT CLIP (~30-40ms
    # total here), not sub-frame noise.
    assert audio_dur == pytest.approx(video_dur, abs=1.0 / 24)
    assert audio_dur == pytest.approx(requested_total, abs=1.0 / 24)


def test_host_audio_atrim_matches_a_from_file_start_reference_decode(tmp_path):
    """The duration match above could hide a systematic sub-frame content
    shift if both sides were shifted the same way. Independently verify a
    single kept span's audio (via the real ``host_audio_input_args`` +
    ``host_bed_filter_lines`` path) matches a no-seek reference decode of the
    same absolute region — the ground truth this fix was validated against.

    Deliberately a single span, not the full multi-span concat: many
    overlapping-window reads of the same short synthetic file concurrently
    (this test's stress fixture packs kept spans much closer together than
    2x SEEK_MARGIN) triggers an unrelated ffmpeg same-file-multi-input
    flakiness that a real episode's actual spacing doesn't hit — verified
    separately against the real production render (all sampled spans across
    a full 83-span, 85-minute episode matched a ground-truth decode at
    corr=1.0000). This test isolates the actual fix (atrim + -copyts) from
    that unrelated multi-input behaviour.
    """
    host_av = tmp_path / "host2.mkv"
    _make_av_source(host_av, duration=16.0)

    s_in, s_out = 4.291666666666667, 9.0  # the longest kept span in _EDL_SEGMENTS
    inputs = audiotrack.host_audio_input_args(host_av, s_in, s_out)
    lines, out_label = audiotrack.host_bed_filter_lines(
        [{"source_in": s_in, "source_out": s_out}], out_label="ha")
    graph = ";\n".join(lines) + "\n"
    audio_out = tmp_path / "single_span.wav"
    ffmpeg.run_ffmpeg([*inputs, "-filter_complex", graph, "-map", f"[{out_label}]",
                        "-c:a", "pcm_s16le", "-ar", "48000", str(audio_out)])

    # Reference: decode the whole file with no seeking at all, then trim with
    # atrim on the same absolute timestamps — unaffected by any seek-related
    # timestamp rebasing.
    ref_out = tmp_path / "single_span_ref.wav"
    ffmpeg.run_ffmpeg([
        "-i", str(host_av),
        "-filter_complex", f"[0:a]atrim=start={s_in:.6f}:end={s_out:.6f},asetpts=PTS-STARTPTS[ref]",
        "-map", "[ref]", "-c:a", "pcm_s16le", "-ar", "48000", str(ref_out),
    ])

    def load(path):
        raw = ffmpeg.run_ffmpeg_capture_stdout_bytes(["-i", str(path), "-map", "0:a", "-f", "s16le", "-ac", "1", "-"])
        return np.frombuffer(raw, dtype="<i2").astype(np.float64)

    # 1s window well clear of the 25ms splice fades at either end of the clip.
    sr = 48000
    a = load(audio_out)[int(1.0 * sr):int(2.0 * sr)]
    r = load(ref_out)[int(1.0 * sr):int(2.0 * sr)]
    n = min(len(a), len(r))
    a, r = a[:n], r[:n]

    # Small bounded lag search (+/-1ms), not a full unbounded xcorr: a coarse
    # -ss seek's own PTS origin can land a handful of samples off true (this
    # is orders of magnitude tighter than the multi-millisecond, ACCUMULATING
    # loss this fix addresses — the real regression guard is the duration
    # match above; this just confirms the fix reads the right AUDIO, not
    # some other moment of the recording).
    max_lag = int(0.001 * sr)
    best_corr = -1.0
    for lag in range(-max_lag, max_lag + 1):
        a_s = a[max(0, lag):n + min(0, lag)]
        r_s = r[max(0, -lag):n + min(0, -lag)]
        a0, r0 = a_s - a_s.mean(), r_s - r_s.mean()
        denom = np.sqrt((a0 ** 2).sum() * (r0 ** 2).sum())
        if denom > 1e-9:
            best_corr = max(best_corr, float((a0 * r0).sum() / denom))
    assert best_corr > 0.99, (
        f"audio span diverges from the from-file-start reference decode within +/-1ms "
        f"(best corr={best_corr:.4f}) — content is shifted, not just concatenated short"
    )

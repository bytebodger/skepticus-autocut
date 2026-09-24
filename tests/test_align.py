"""Reaction alignment: cue detection, alternation validation, and the
cumulative source-position arithmetic (reaction spec build steps 1-3).

Mostly pure logic, plus one dry-run smoke test of the full orchestration. The
end-to-end run against real Whisper output and the eyeball lip-sync check are
done separately."""

import json

import pytest

from autocut import align, ffmpeg
from autocut.paths import resolve


def _words(tokens, t0=0.0, step=0.5):
    """A words.json-shaped list from tokens, evenly spaced."""
    out = []
    for k, tok in enumerate(tokens):
        s = t0 + k * step
        out.append({"i": k, "word": tok, "start": round(s, 3), "end": round(s + step, 3)})
    return out


START = "end my commentary"
STOP = "begin my commentary"


# --------------------------------------------------------------------------- #
# Step 1 — cue detection
# --------------------------------------------------------------------------- #

def test_detect_cues_finds_both_phrases_tagged_and_sorted():
    words = (_words("so anyway".split(), t0=0.0)
            + _words(START.split(), t0=2.0)
            + _words("some playback happens here".split(), t0=6.0)
            + _words(STOP.split(), t0=10.0)
            + _words("back to commentary".split(), t0=13.0))
    cues = align.detect_cues({"words": words}, start_phrase=START, stop_phrase=STOP)
    assert [c["kind"] for c in cues] == ["start", "stop"]
    assert cues[0]["start"] < cues[1]["start"]


def test_detect_cues_requires_the_complete_phrase():
    # "commentary" alone must not fire the cue (spec §2: avoid single words
    # that occur naturally in the subject matter).
    words = _words("let's discuss the biblical commentary on this passage".split())
    cues = align.detect_cues({"words": words}, start_phrase=START, stop_phrase=STOP)
    assert cues == []


def test_detect_cues_accepts_a_start_variant():
    # Whisper regularly mishears "end my commentary" as "and my commentary".
    words = (_words("so anyway".split(), t0=0.0)
            + _words("and my commentary".split(), t0=2.0)
            + _words("some playback happens here".split(), t0=6.0)
            + _words(STOP.split(), t0=10.0))
    cues = align.detect_cues({"words": words}, start_phrase=START, stop_phrase=STOP,
                             start_variants=("and my commentary",))
    assert [c["kind"] for c in cues] == ["start", "stop"]


def test_detect_cues_ignores_variant_when_not_configured():
    words = _words("and my commentary".split(), t0=2.0)
    cues = align.detect_cues({"words": words}, start_phrase=START, stop_phrase=STOP)
    assert cues == []


def test_detect_cues_ignores_words_outside_any_cue():
    words = (_words("filler words here".split(), t0=0.0)
            + _words(START.split(), t0=3.0)
            + _words("more filler".split(), t0=6.0))
    cues = align.detect_cues({"words": words}, start_phrase=START, stop_phrase=STOP)
    assert len(cues) == 1
    assert cues[0]["kind"] == "start"


# --------------------------------------------------------------------------- #
# Step 1 — alternation validation
# --------------------------------------------------------------------------- #

def _cue(kind, t):
    return {"start": t, "end": t + 1.0, "kind": kind}


def test_alternation_accepts_a_clean_sequence():
    cues = [_cue("start", 10), _cue("stop", 20), _cue("start", 30), _cue("stop", 40)]
    align.validate_alternation(cues)   # does not raise


def test_alternation_accepts_a_trailing_open_segment():
    # Ends mid-playback (spec §2: "may end in either state") -- not a violation.
    cues = [_cue("start", 10), _cue("stop", 20), _cue("start", 30)]
    align.validate_alternation(cues)   # does not raise


def test_alternation_rejects_a_leading_stop_cue():
    # The recording is assumed to open in commentary (spec §2).
    cues = [_cue("stop", 5), _cue("start", 10)]
    with pytest.raises(RuntimeError, match="alternation broken"):
        align.validate_alternation(cues)


def test_alternation_rejects_two_starts_in_a_row():
    cues = [_cue("start", 10), _cue("start", 20)]
    with pytest.raises(RuntimeError, match="alternation broken"):
        align.validate_alternation(cues)


def test_alternation_rejects_two_stops_in_a_row():
    cues = [_cue("start", 10), _cue("stop", 20), _cue("stop", 30)]
    with pytest.raises(RuntimeError, match="alternation broken"):
        align.validate_alternation(cues)


def test_alternation_error_reports_violating_timestamps():
    cues = [_cue("start", 10), _cue("start", 25.5)]
    with pytest.raises(RuntimeError, match=r"25\.50s"):
        align.validate_alternation(cues)


# --------------------------------------------------------------------------- #
# Step 2 — segment boundaries from cues
# --------------------------------------------------------------------------- #

def test_segments_from_cues_pairs_start_and_stop():
    cues = [_cue("start", 10), _cue("stop", 20)]
    segs = align.segments_from_cues(cues, episode_duration=100.0, silences=[])
    assert segs == [{"host_in": 11.0, "host_out": 20.0}]   # host_in = start cue's END


def test_segments_from_cues_pairs_multiple_segments_independently():
    cues = [_cue("start", 10), _cue("stop", 20), _cue("start", 50), _cue("stop", 70)]
    segs = align.segments_from_cues(cues, episode_duration=100.0, silences=[])
    assert len(segs) == 2
    assert segs[0] == {"host_in": 11.0, "host_out": 20.0}
    assert segs[1] == {"host_in": 51.0, "host_out": 70.0}


def test_segments_from_cues_closes_a_trailing_segment_at_episode_end():
    cues = [_cue("start", 10), _cue("stop", 20), _cue("start", 50)]
    segs = align.segments_from_cues(cues, episode_duration=90.0, silences=[])
    assert segs[-1] == {"host_in": 51.0, "host_out": 90.0}


def test_segments_from_cues_handles_no_cues():
    assert align.segments_from_cues([], episode_duration=100.0, silences=[]) == []


# --------------------------------------------------------------------------- #
# Step 2 — snapping boundaries onto confirmed acoustic silence (Whisper's word
# timestamps in this project consistently run early).
# --------------------------------------------------------------------------- #

def test_segments_from_cues_snaps_host_in_forward_to_confirmed_silence():
    # cue's claimed end is 11.0, but real speech ("-tary") audibly continues
    # until the confirmed silence at 11.3 - host_in must land past it.
    cues = [_cue("start", 10), _cue("stop", 20)]
    silences = [{"start": 11.3, "end": 11.9}]
    segs = align.segments_from_cues(cues, episode_duration=100.0, silences=silences)
    assert segs[0]["host_in"] == 11.3


def test_segments_from_cues_snaps_host_out_forward_to_confirmed_speech():
    # cue's claimed start is 20.0, but the confirmed silence before it runs
    # until 20.25 - real speech can't have resumed before that.
    cues = [_cue("start", 10), _cue("stop", 20)]
    silences = [{"start": 19.6, "end": 20.25}]
    segs = align.segments_from_cues(cues, episode_duration=100.0, silences=silences)
    assert segs[0]["host_out"] == 20.25


def test_segments_from_cues_never_snaps_backward():
    # A confirmed silence entirely before the cue's claimed end must not pull
    # host_in earlier than the raw timestamp.
    cues = [_cue("start", 10), _cue("stop", 20)]
    silences = [{"start": 9.0, "end": 9.5}]
    segs = align.segments_from_cues(cues, episode_duration=100.0, silences=silences)
    assert segs[0]["host_in"] == 11.0


def test_segments_from_cues_ignores_a_distant_silence():
    # No confirmed silence within CUE_SNAP_MAX_GAP of the cue - trust the raw
    # cue timestamp rather than snapping across an implausibly large gap.
    cues = [_cue("start", 10), _cue("stop", 20)]
    silences = [{"start": 50.0, "end": 50.5}]
    segs = align.segments_from_cues(cues, episode_duration=100.0, silences=silences)
    assert segs[0]["host_in"] == 11.0


# --------------------------------------------------------------------------- #
# Step 2 — cumulative source-position arithmetic
# --------------------------------------------------------------------------- #

def test_cumulative_position_starts_at_zero():
    segs = align.assign_source_positions([{"host_in": 10.0, "host_out": 25.0}])
    assert len(segs) == 1
    s = segs[0]
    assert s["id"] == "pb001"
    assert s["source_in"] == 0.0
    assert s["source_out"] == pytest.approx(15.0)
    assert s["offset_source"] == "cumulative"
    assert s["refinement_delta"] is None
    assert s["seek_detected"] is False


def test_cumulative_position_resumes_where_the_previous_segment_stopped():
    # Pausing stops the source clock (spec §3): segment 2 resumes at segment
    # 1's source_out, regardless of the host-side gap between them.
    raw = [{"host_in": 10.0, "host_out": 25.0}, {"host_in": 100.0, "host_out": 130.0}]
    segs = align.assign_source_positions(raw)
    assert segs[0]["source_in"] == 0.0
    assert segs[0]["source_out"] == pytest.approx(15.0)
    assert segs[1]["source_in"] == pytest.approx(15.0)
    assert segs[1]["source_out"] == pytest.approx(45.0)


def test_cumulative_position_ids_are_sequential():
    raw = [{"host_in": 0.0, "host_out": 5.0}, {"host_in": 10.0, "host_out": 12.0},
           {"host_in": 20.0, "host_out": 21.0}]
    segs = align.assign_source_positions(raw)
    assert [s["id"] for s in segs] == ["pb001", "pb002", "pb003"]


# --------------------------------------------------------------------------- #
# Duration invariant
# --------------------------------------------------------------------------- #

def test_assert_equal_durations_accepts_cumulative_output():
    segs = align.assign_source_positions([{"host_in": 5.0, "host_out": 30.0}])
    align.assert_equal_durations(segs)   # does not raise


def test_assert_equal_durations_rejects_a_divergent_segment():
    bad = [{"id": "pb001", "host_in": 10.0, "host_out": 20.0,
            "source_in": 0.0, "source_out": 9.0}]   # 9s vs 10s
    with pytest.raises(RuntimeError, match="source duration"):
        align.assert_equal_durations(bad)


# --------------------------------------------------------------------------- #
# Reaction config (cue phrase overrides)
# --------------------------------------------------------------------------- #

def _episode(tmp_path):
    from autocut.paths import Episode
    return Episode(episode_id="ep", root=tmp_path)


def test_reaction_config_defaults_when_no_config_file(tmp_path):
    cfg = align._reaction_config(_episode(tmp_path))
    assert cfg["cue_playback_start"] == align.DEFAULT_CUE_PLAYBACK_START
    assert cfg["cue_playback_stop"] == align.DEFAULT_CUE_PLAYBACK_STOP


def test_reaction_config_reads_overrides(tmp_path):
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "layout.yaml").write_text(
        "reaction:\n"
        "  cue_playback_start: roll the clip\n"
        "  cue_playback_stop: pause the clip\n",
        encoding="utf-8",
    )
    cfg = align._reaction_config(_episode(tmp_path))
    assert cfg["cue_playback_start"] == "roll the clip"
    assert cfg["cue_playback_stop"] == "pause the clip"


def test_reaction_config_defaults_when_config_has_no_reaction_section(tmp_path):
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "layout.yaml").write_text("style: default\n", encoding="utf-8")
    cfg = align._reaction_config(_episode(tmp_path))
    assert cfg["cue_playback_start"] == align.DEFAULT_CUE_PLAYBACK_START


# --------------------------------------------------------------------------- #
# End-to-end orchestration (dry-run smoke test)
# --------------------------------------------------------------------------- #

@pytest.fixture
def dry_run():
    ffmpeg.set_dry_run(True)
    try:
        yield
    finally:
        ffmpeg.set_dry_run(False)


def test_run_writes_playback_json_from_a_clean_cue_sequence(dry_run, tmp_path):
    ep = resolve("epcue", root=tmp_path)
    ep.transcript_dir.mkdir(parents=True, exist_ok=True)
    ep.work.mkdir(parents=True, exist_ok=True)
    words = (_words("intro chat happens here".split(), t0=0.0)
            + _words(START.split(), t0=3.0)
            + _words("the source plays for a while".split(), t0=6.0)
            + _words(STOP.split(), t0=20.0)
            + _words("and now we wrap up".split(), t0=23.0))
    ep.words_json.write_text(json.dumps({"words": words}), encoding="utf-8")
    ep.silence_json.write_text(json.dumps({"silences": []}), encoding="utf-8")
    ep.probe_json.write_text(json.dumps({"source_duration": 30.0, "fps": 24.0}),
                             encoding="utf-8")

    playback = align.run(ep)

    assert ep.playback_json.exists()
    assert playback["episode_id"] == "epcue"
    assert len(playback["segments"]) == 1
    seg = playback["segments"][0]
    assert seg["id"] == "pb001"
    assert seg["source_in"] == 0.0
    assert seg["source_out"] == pytest.approx(seg["host_out"] - seg["host_in"], abs=1e-6)
    assert seg["offset_source"] == "cumulative"
    assert seg["refinement_delta"] is None
    assert seg["seek_detected"] is False


def test_run_fails_loudly_on_broken_alternation(dry_run, tmp_path):
    ep = resolve("epbad", root=tmp_path)
    ep.transcript_dir.mkdir(parents=True, exist_ok=True)
    ep.work.mkdir(parents=True, exist_ok=True)
    words = _words(START.split(), t0=0.0) + _words(START.split(), t0=5.0)
    ep.words_json.write_text(json.dumps({"words": words}), encoding="utf-8")
    ep.silence_json.write_text(json.dumps({"silences": []}), encoding="utf-8")
    ep.probe_json.write_text(json.dumps({"source_duration": 30.0, "fps": 24.0}),
                             encoding="utf-8")

    with pytest.raises(RuntimeError, match="alternation broken"):
        align.run(ep)
    assert not ep.playback_json.exists()


def test_run_fails_loudly_when_no_cues_are_found(dry_run, tmp_path):
    ep = resolve("epnocues", root=tmp_path)
    ep.transcript_dir.mkdir(parents=True, exist_ok=True)
    ep.work.mkdir(parents=True, exist_ok=True)
    ep.words_json.write_text(json.dumps({"words": _words("just talking the whole time".split())}),
                             encoding="utf-8")
    ep.silence_json.write_text(json.dumps({"silences": []}), encoding="utf-8")
    ep.probe_json.write_text(json.dumps({"source_duration": 30.0, "fps": 24.0}),
                             encoding="utf-8")

    with pytest.raises(RuntimeError, match="no cues detected"):
        align.run(ep)

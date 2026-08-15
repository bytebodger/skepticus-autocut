"""Tests for the load-bearing EDL logic: time mapping and validation.

The single most important invariant is that EDL times are SOURCE-timebase and
that source_to_output maps them correctly after cuts. These tests guard the most
likely bug in the pipeline (spec section 5).
"""

import pytest

from autocut import edl


def _seg(id, in_, out, action="keep", **extra):
    return {"id": id, "in": in_, "out": out, "action": action, **extra}


# --------------------------------------------------------------------------- #
# Time mapping
# --------------------------------------------------------------------------- #

def test_build_time_map_skips_drops_and_accumulates():
    segments = [
        _seg("s001", 0.0, 10.0),                 # keep -> out [0,10)
        _seg("s002", 10.0, 15.0, "drop"),        # dropped
        _seg("s003", 15.0, 25.0),                # keep -> out [10,20)
    ]
    spans = edl.build_time_map(segments)
    assert spans == [(0.0, 10.0, 0.0), (15.0, 25.0, 10.0)]


def test_source_to_output_maps_kept_time():
    spans = edl.build_time_map([
        _seg("s001", 0.0, 10.0),
        _seg("s002", 10.0, 15.0, "drop"),
        _seg("s003", 15.0, 25.0),
    ])
    # A moment in the first keep maps to itself.
    assert edl.source_to_output(5.0, spans) == 5.0
    # A moment in the second keep shifts left by the 5s that was cut.
    assert edl.source_to_output(20.0, spans) == 15.0


def test_source_to_output_returns_none_for_cut_time():
    spans = edl.build_time_map([
        _seg("s001", 0.0, 10.0),
        _seg("s002", 10.0, 15.0, "drop"),
        _seg("s003", 15.0, 25.0),
    ])
    assert edl.source_to_output(12.0, spans) is None


def test_source_to_output_half_open_boundary():
    spans = edl.build_time_map([_seg("s001", 0.0, 10.0)])
    assert edl.source_to_output(0.0, spans) == 0.0
    # out is exclusive
    assert edl.source_to_output(10.0, spans) is None


def test_overlay_in_back_half_maps_correctly():
    """The exact bug the spec warns about: an overlay late in the timeline must
    account for all preceding cuts."""
    segments = [
        _seg("s001", 0.0, 30.0),
        _seg("s002", 30.0, 60.0, "drop"),   # cut 30s
        _seg("s003", 60.0, 120.0),
    ]
    spans = edl.build_time_map(segments)
    # Overlay at source 90s -> output 90 - 30 = 60s, NOT 90.
    assert edl.source_to_output(90.0, spans) == 60.0


def test_output_duration():
    spans = edl.build_time_map([
        _seg("s001", 0.0, 10.0),
        _seg("s002", 10.0, 15.0, "drop"),
        _seg("s003", 15.0, 25.0),
    ])
    assert edl.output_duration(spans) == 20.0


def test_snap_to_frames():
    assert edl.snap(1.017, 30) == pytest.approx(round(1.017 * 30) / 30)
    assert edl.snap(1.0, 30) == 1.0


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def _valid_edl():
    return {
        "version": 1,
        "episode_id": "ep001",
        "source": "work/ep001/mezz/mezz.mkv",
        "fps": 30,
        "segments": [
            _seg("s001", 0.0, 10.0),
            _seg("s002", 10.0, 12.0, "drop", reason="filler", confidence=0.95),
            _seg("s003", 12.0, 20.0),
        ],
        "overlays": [
            {"id": "ov001", "composition": "lower_third", "source_time": 5.0,
             "duration": 4.0, "props": {"line1": "x"}},
        ],
        "grade": "luts/skepticus_v1.cube",
        "captions": {"style": "styles/captions.ass.template", "enabled": True},
    }


def test_valid_edl_passes():
    assert edl.validate(_valid_edl(), source_duration=20.0) == []


def test_overlay_in_dropped_region_is_error():
    e = _valid_edl()
    e["overlays"][0]["source_time"] = 11.0  # inside the dropped segment
    errors = edl.validate(e, source_duration=20.0)
    assert any("dropped region" in x for x in errors)


def test_overlay_time_beyond_source_is_error():
    e = _valid_edl()
    e["overlays"][0]["source_time"] = 999.0
    errors = edl.validate(e, source_duration=20.0)
    assert any("exceeds source duration" in x for x in errors)


def test_segment_out_beyond_source_flags_output_timebase_leak():
    e = _valid_edl()
    e["segments"][-1]["out"] = 500.0
    errors = edl.validate(e, source_duration=20.0)
    assert any("exceeds source duration" in x for x in errors)


def test_overlapping_segments_rejected():
    e = _valid_edl()
    e["segments"] = [_seg("s001", 0.0, 10.0), _seg("s002", 5.0, 15.0)]
    errors = edl.validate(e, source_duration=20.0)
    assert any("non-overlapping" in x or "before previous" in x for x in errors)


def test_retake_drop_requires_confidence():
    e = _valid_edl()
    e["segments"].append(_seg("s004", 20.0, 22.0, "drop", reason="retake"))
    e["segments"][-2]["out"] = 20.0  # keep tiling contiguous... adjust s003
    # s003 currently 12-20, new s004 20-22, fine.
    errors = edl.validate(e, source_duration=22.0)
    assert any("confidence" in x for x in errors)


def test_wrong_version_rejected():
    e = _valid_edl()
    e["version"] = 2
    assert any("version" in x for x in edl.validate(e))


def test_no_kept_segments_rejected():
    e = _valid_edl()
    for s in e["segments"]:
        s["action"] = "drop"
        s.setdefault("confidence", 0.5)
    errors = edl.validate(e)
    assert any("kept" in x for x in errors)


# --------------------------------------------------------------------------- #
# Override preservation
# --------------------------------------------------------------------------- #

def test_apply_overrides_preserves_human_veto():
    prior = _valid_edl()
    prior["segments"][1]["action"] = "keep"
    prior["segments"][1]["override"] = True

    overrides = edl.collect_overrides(prior)
    assert "s002" in overrides

    fresh = _valid_edl()  # re-authored: s002 is a drop again
    assert fresh["segments"][1]["action"] == "drop"

    merged = edl.apply_overrides(fresh, overrides)
    assert merged["segments"][1]["action"] == "keep"
    assert merged["segments"][1]["override"] is True


def test_apply_overrides_no_op_without_overrides():
    fresh = _valid_edl()
    merged = edl.apply_overrides(fresh, {})
    assert merged["segments"][1]["action"] == "drop"

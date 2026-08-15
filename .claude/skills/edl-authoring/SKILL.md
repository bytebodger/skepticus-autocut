---
name: edl-authoring
description: How to author an Edit Decision List (edl.json) for the Skepticus Autocut pipeline — the schema, the source-timebase rule, the cut heuristics, and the conservatism rules. Use when reading a transcript and deciding what to cut.
---

# Authoring an EDL

Your job in this pipeline is editorial: read the transcript and emit an EDL.
Deterministic tools do everything else. You never invoke ffmpeg.

Inputs: `work/<ep>/transcript/words.json`, `.../silence.json`, `work/<ep>/probe.json`.
Output: `work/<ep>/edl.json`.

## The one rule that matters most: SOURCE timebase

Every timestamp in the EDL — segment `in`/`out`, overlay `source_time` — is a
moment in the **raw recording**, before any cuts. The compositor converts to
output time after cuts are applied (`source_to_output` in `edl.py`).

If you write output-relative times, every overlay in the back half fires at the
wrong moment and the error grows with each cut. `python -m autocut validate`
guards this: an overlay whose `source_time` lands in a dropped region, or exceeds
the source duration, is an error.

## Schema

```json
{
  "version": 1,
  "episode_id": "ep042",
  "source": "work/ep042/mezz/mezz.mkv",
  "fps": 30,
  "segments": [
    {"id": "s001", "in": 12.480, "out": 47.220, "action": "keep", "note": "cold open"},
    {"id": "s002", "in": 47.220, "out": 51.900, "action": "drop",
     "reason": "retake", "confidence": 0.72, "note": "flubbed line, clean take follows"}
  ],
  "overlays": [
    {"id": "ov001", "composition": "lower_third", "source_time": 15.0,
     "duration": 4.0, "props": {"line1": "Skepticus", "line2": "Episode 42"}}
  ],
  "grade": "luts/skepticus_v1.cube",
  "captions": {"style": "styles/captions.ass.template", "enabled": true}
}
```

Segments must tile source time in order, non-overlapping. Emit `drop` entries
rather than deleting segments — the review UI shows what was removed and why.

## What to look for (ordered by how reliably it automates)

1. **Leading/trailing dead air.** Trivial, always safe.
2. **Long silences.** Over ~700ms mid-sentence. Trim to ~250ms, don't remove
   entirely, or the pacing turns robotic.
3. **Filler words.** "um", "uh", "you know", "like" as filler. Use word
   boundaries from words.json. Be conservative near sentence starts.
4. **False starts.** A partial clause abandoned and restarted — near-prefix
   repetition within a short window.
5. **Retakes.** Two adjacent near-identical passages. Keep the later one by
   default. Hardest case: always set `confidence` and always route to review.

## Conservatism rules (non-negotiable)

- Never cut mid-word. Word boundaries only.
- Leave 80–120ms of padding on each side of a keep boundary. Tight cuts sound clipped.
- Never merge two keeps separated by a drop shorter than 150ms — keep the whole thing.
- Retake drops MUST carry a `confidence` score.

## Workflow

- A deterministic baseline exists: `python -m autocut autoauthor <ep>` handles
  dead air, long silences, and single-word fillers. Start there, then add the
  judgement calls (false starts, retakes, overlays).
- Snap boundaries to frames (`edl.snap`); the autoauthor path already does.
- Re-running never clobbers a human veto: segments with `override: true` are
  preserved. Don't remove that flag.
- Finish with `python -m autocut validate <ep>` and fix every error.

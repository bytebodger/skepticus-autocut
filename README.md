# Skepticus Autocut

An automated video editing pipeline. Drop a raw MP4 in `inbox/`, get a cut,
graded, captioned video out in `out/`.

**Claude Code decides. Deterministic tools execute.** Claude Code reads the
transcript and emits an Edit Decision List (`edl.json`). Every stage after that
is a pure, cached function of the EDL plus the source file. See
[`skepticus-autocut-spec.md`](skepticus-autocut-spec.md) for the full design.

## Setup

The pipeline shells out to `ffmpeg`/`ffprobe` and uses `faster-whisper` on CUDA.

1. **Python.** Build the venv from **python.org Python 3.12**, not the Microsoft
   Store build — its app-execution aliases intercept bare `python` calls and
   produce confusing failures in a pipeline that shells out.
   ```
   py -3.12 -m venv .venv
   .venv\Scripts\activate
   pip install -e .[dev]
   ```
2. **FFmpeg.** Install a pinned build and put it on `PATH` (or set
   `AUTOCUT_FFMPEG` / `AUTOCUT_FFPROBE` to absolute paths). Tested against
   FFmpeg 9.0 (uses `-fps_mode cfr` and `-/filter_complex`, the modern
   replacements for the removed `-vsync` and `-filter_complex_script`).
3. **HyperFrames.** `npx hyperframes` is used for overlay cards (stage 6).
4. **GPU.** faster-whisper `large-v3` at float16 uses ~4.7GB VRAM. On Windows,
   install the CUDA runtime DLLs with `pip install -e .[cuda]` — transcribe.py
   registers them automatically at import. Without a CUDA runtime, transcribe on
   CPU by setting `AUTOCUT_WHISPER_DEVICE=cpu` and `AUTOCUT_WHISPER_COMPUTE=int8`
   (optionally a smaller `AUTOCUT_WHISPER_MODEL`).

## Usage

```
# Pre-review pipeline: probe -> transcribe -> baseline EDL -> validate
python -m autocut all ep042

# ... or run stages individually
python -m autocut probe ep042
python -m autocut transcribe ep042
python -m autocut autoauthor ep042      # deterministic baseline EDL
python -m autocut validate ep042

# Review proposed cuts in the browser, veto any you disagree with
python -m autocut review ep042

# Render everything after review
python -m autocut render ep042          # cut -> grade -> captions -> overlays -> composite -> qc
```

Or drive it from Claude Code with `/edit ep042` (see `.claude/commands/edit.md`),
which authors the EDL and stops before rendering.

Global flags: `--dry-run` (log ffmpeg commands, run nothing), `--force` (ignore
caches), `-v` (debug logging).

## Pipeline stages

| stage | command | output |
|-------|---------|--------|
| 1 probe/normalize | `probe` | CFR all-intra mezzanine + 16kHz speech wav + `probe.json` |
| 2 transcribe | `transcribe` | `words.json` + `silence.json` |
| 3 EDL | `autoauthor` / Claude Code | `edl.json` |
| — validate | `validate` | schema + source-timebase checks |
| — review | `review` | FastAPI veto gate |
| 4 cut/concat | `cut` | `cut.mkv` |
| 5 grade | `grade` | `graded.mkv` |
| 6 overlays | `overlays` | alpha WebM cards |
| 7 captions | `captions` | `captions.ass` |
| 8 composite | `composite` | `out/<ep>.mp4` |
| 9 QC | `qc` | `out/<ep>_report.md` + contact sheet |

## Reaction format

A second episode type: reacting to a source video, with the source playing in
a content window and the host in a speaker window (split-screen show frame,
not a raw recording). See
[`skepticus-autocut-reaction-spec.md`](skepticus-autocut-reaction-spec.md) for
the full design — cue phrases, the alignment math, the playback-map schema —
and [`skepticus-autocut-compositor-spec.md`](skepticus-autocut-compositor-spec.md)
for the show-frame layout `compose` renders.

Two files in `inbox/`, alongside the usual `<ep>.mp4` host recording:

- `<ep>_source.<ext>` — the clean source video being reacted to (required).
- `<ep>_thumb.<ext>` — the source's YouTube thumbnail (optional). Shown in the
  content window during opening commentary instead of a frozen first frame;
  falls back to the frozen frame when absent.

The host declares playback boundaries out loud — "end my commentary" starts
playback, "begin my commentary" ends it (both configurable; see below):

```
# Same host-recording prep as the monologue format
python -m autocut probe ep042
python -m autocut transcribe ep042

# Reaction-specific: align host<->source from spoken cues -> playback.json
python -m autocut align ep042
python -m autocut align-check ep042      # renders a lip-sync clip per segment for review

# autoauthor/validate/review are unchanged, but now playback-aware: no drops
# land inside a playback region, and the cue phrases themselves are cut
python -m autocut autoauthor ep042
python -m autocut validate ep042
python -m autocut review ep042           # optional veto gate

# compose (the Phase-2 compositor: background + speaker window + content
# window + captions + audio, one command) detects playback.json and builds
# the content/audio tracks from it automatically -- no manual content.json
python -m autocut compose ep042 --range 0:60   # iterate on a window first
python -m autocut compose ep042                # full render

# QC: catch a cue phrase leaking into the rendered output; verify the
# cumulative source position against a ground-truth re-transcription
python -m autocut cue-check ep042
python -m autocut sync-check ep042
```

| stage | command | output |
|-------|---------|--------|
| align | `align` | `work/<ep>/playback.json` — host<->source segment map |
| align verify | `align-check` | lip-sync clips, `work/<ep>/align/check/*.mp4` |
| compose | `compose` | `work/<ep>/compose/composite.mp4` |
| cue leak check | `cue-check` | `work/<ep>/cuecheck/report.json` |
| sync check | `sync-check` | `work/<ep>/sync_check/report.json` |

### Config (`config/layout.yaml`, `reaction:` section)

All keys are optional; shown values are the defaults used when the section or
an individual key is absent.

```yaml
reaction:
  cue_playback_start: "end my commentary"      # said just before playback starts
  cue_playback_stop: "begin my commentary"     # said just after playback stops
  cue_playback_start_variants: ["and my commentary"]  # accepted mishearings
  match_levels: true           # gain-match host mic vs source audio
  source_gain_db: 0
  crossfade_ms: 75              # crossfade at every host<->source audio switch
  thumbnail: inbox/ep042_thumb.jpg   # optional; overrides thumbnail auto-detect
```

## Design invariants

- **All EDL times are SOURCE-timebase.** `source_to_output` (in `edl.py`) converts
  to output time after cuts. Output-relative times in the EDL are the pipeline's
  most likely bug; validation and unit tests guard against them.
- **Every stage is cached and resumable.** Each writes a `.done` file holding a
  hash of its inputs; unchanged stages are skipped. Segments cache individually.
- **Nothing is destructive.** The raw file is read-only; everything under `work/`
  and `out/` is regenerable.

## Tests

```
python -m pytest
```

The pure-Python, load-bearing logic (time mapping, EDL validation, caption
generation, the baseline author, caching) is unit-tested without needing ffmpeg
or a GPU.

## Layout

```
autocut/        pipeline stages + CLI (see the table above)
compositions/   HyperFrames overlay cards (lower_third, title_card, chapter_marker)
luts/           3D color grades (.cube), versioned by filename
styles/         caption ASS style template
inbox/          drop raw MP4s here
work/           intermediates (gitignored)
out/            finished videos (gitignored)
.claude/        /edit command + edl-authoring & hyperframes-cards skills
tests/          unit tests
```

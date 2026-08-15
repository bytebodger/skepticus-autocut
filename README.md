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

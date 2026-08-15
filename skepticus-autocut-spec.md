# Skepticus Autocut - Build Spec

An automated video editing pipeline. Drop a raw MP4 in a folder, get a cut,
graded, captioned video out.

Stack: Python 3.12, faster-whisper, FFmpeg, HyperFrames, Claude Code.
Target machine: Windows 11, i7-14700F, 32GB RAM, RTX 4070 12GB.

---

## 1. Design principles

**Claude Code decides. Deterministic tools execute.**

Claude Code's only job is reading a transcript and emitting an Edit Decision
List. It never invokes FFmpeg directly. Every stage after the EDL is a pure
function of the EDL plus the source file.

This buys three things. A human-reviewable checkpoint before anything renders.
Cheap reruns when you change one caption. And identical output across episodes,
which is what makes a channel look like a channel.

**Every stage is cached and resumable.**

Stages write to `work/<episode_id>/<stage>/`. Each stage writes a `.done` file
containing a hash of its inputs. Re-running skips any stage whose input hash is
unchanged. You will re-run this pipeline dozens of times per episode while
tuning. Make that cheap.

**Nothing is destructive.**

The raw file is read-only. Everything else is regenerable.

---

## 2. Repo layout

```
skepticus-autocut/
  pyproject.toml
  autocut/
    __init__.py
    cli.py              # python -m autocut <stage> <episode>
    probe.py            # stage 1
    transcribe.py       # stage 2
    analyze.py          # stage 3 helpers (Claude Code writes the EDL)
    cut.py              # stage 4
    grade.py            # stage 5
    overlays.py         # stage 6
    captions.py         # stage 7
    composite.py        # stage 8
    edl.py              # schema, validation, source<->output time mapping
    ffmpeg.py           # thin wrapper: logging, error surfacing, dry-run
    review.py           # FastAPI review gate
  compositions/         # HyperFrames projects
    lower_third/
    title_card/
    chapter_marker/
  luts/
    skepticus_v1.cube
  styles/
    captions.ass.template
  inbox/                # drop raw MP4s here
  work/                 # intermediates, gitignored
  out/                  # finished videos
  .claude/
    commands/
      edit.md
    skills/
      edl-authoring/
      hyperframes-cards/
```

---

## 3. Stage 1 - Probe and normalize

Read the source with `ffprobe`. Record fps, resolution, audio sample rate,
duration, and whether the stream is VFR.

**This stage is not optional.** OBS commonly records variable frame rate. Cutting
VFR footage by timestamp produces audio drift that compounds across the episode.

Build a constant-frame-rate, all-intra mezzanine:

```
ffmpeg -i inbox/EP.mp4 ^
  -vsync cfr -r 30 ^
  -c:v libx264 -preset veryfast -crf 16 -g 1 -pix_fmt yuv420p ^
  -c:a pcm_s16le -ar 48000 ^
  work/EP/mezz/mezz.mkv
```

`-g 1` makes every frame a keyframe. Segment cuts become frame accurate and
fast, with no full re-encode of the timeline. The file is large. It's temporary.

Also extract mono 16kHz audio for Whisper:

```
ffmpeg -i work/EP/mezz/mezz.mkv -ac 1 -ar 16000 -c:a pcm_s16le work/EP/audio/speech.wav
```

Write `work/EP/probe.json` with everything you learned. Downstream stages read
fps from here, never guess it.

---

## 4. Stage 2 - Transcribe

Use `faster-whisper`, not the reference implementation. Roughly 4x faster and
lower VRAM.

```python
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")

segments, info = model.transcribe(
    "work/EP/audio/speech.wav",
    word_timestamps=True,
    vad_filter=True,
    vad_parameters={"min_silence_duration_ms": 300},
    beam_size=5,
    condition_on_previous_text=False,
)
```

`large-v3` at float16 uses about 4.7GB. Comfortable on 12GB.

`condition_on_previous_text=False` matters. Leaving it on causes Whisper to
hallucinate repeated phrases during long silences, which is exactly the material
you're trying to cut.

Emit `work/EP/transcript/words.json`:

```json
{
  "language": "en",
  "words": [
    {"i": 0, "start": 12.480, "end": 12.710, "word": "The", "prob": 0.98},
    {"i": 1, "start": 12.710, "end": 13.020, "word": "argument", "prob": 0.99}
  ]
}
```

Separately, run a silence pass. Whisper's VAD is tuned for speech detection, not
for editorial pauses.

```
ffmpeg -i work/EP/audio/speech.wav -af silencedetect=noise=-35dB:d=0.4 -f null -
```

Parse stderr into `work/EP/transcript/silence.json`. Tune the dB threshold to
your room once, then leave it.

---

## 5. Stage 3 - The EDL

This is where Claude Code works. Input: `words.json`, `silence.json`,
`probe.json`. Output: `work/EP/edl.json`.

### Schema

```json
{
  "version": 1,
  "episode_id": "ep042",
  "source": "work/ep042/mezz/mezz.mkv",
  "fps": 30,
  "segments": [
    {
      "id": "s001",
      "in": 12.480,
      "out": 47.220,
      "action": "keep",
      "note": "cold open"
    },
    {
      "id": "s002",
      "in": 47.220,
      "out": 51.900,
      "action": "drop",
      "reason": "retake",
      "confidence": 0.72,
      "note": "flubbed 'Deuteronomy', clean version follows"
    }
  ],
  "overlays": [
    {
      "id": "ov001",
      "composition": "lower_third",
      "source_time": 15.000,
      "duration": 4.000,
      "props": {"line1": "Skepticus", "line2": "Episode 42"}
    }
  ],
  "grade": "luts/skepticus_v1.cube",
  "captions": {"style": "styles/captions.ass.template", "enabled": true}
}
```

### Critical rule: all times are in SOURCE timebase

`source_time` on an overlay means "the moment in the raw recording where this
belongs." The compositor converts to output time after cuts are applied.

If you let Claude Code write output-relative times, every overlay in the back
half of the video fires at the wrong moment, and the error grows with each cut.
This is the most likely bug in the pipeline. Guard it with a unit test.

### The mapping function

Put this in `edl.py`. It's small and load-bearing.

```python
def build_time_map(segments):
    """Returns list of (src_in, src_out, out_start) for kept segments."""
    spans, cursor = [], 0.0
    for s in segments:
        if s["action"] != "keep":
            continue
        spans.append((s["in"], s["out"], cursor))
        cursor += s["out"] - s["in"]
    return spans


def source_to_output(t, spans):
    """Map a source timestamp to output timeline. None if t was cut."""
    for src_in, src_out, out_start in spans:
        if src_in <= t < src_out:
            return out_start + (t - src_in)
    return None
```

Snap all boundaries to frames before writing the EDL:

```python
def snap(t, fps):
    return round(t * fps) / fps
```

### What Claude Code should look for

Ordered by how reliably it can be automated:

1. **Leading and trailing dead air.** Trivial. Always safe.
2. **Long silences.** Anything over ~700ms mid-sentence. Trim to ~250ms rather
   than removing entirely, or the pacing gets robotic.
3. **Filler words.** "um", "uh", "you know", "like" used as filler. Use word
   boundaries from `words.json`. Be conservative near sentence starts, where
   removal sounds abrupt.
4. **False starts.** A partial clause abandoned and restarted. Detectable by
   near-prefix repetition within a short window.
5. **Retakes.** Two semantically near-identical passages, adjacent. Keep the
   later one by default. This is the hard case. Always mark with a confidence
   score and always route to review.

Rules Claude Code must follow:

- Never cut mid-word. Use word boundaries only.
- Leave 80-120ms of padding on each side of a keep boundary. Cutting tight to
  the waveform sounds clipped.
- Never merge two keeps that were separated by a drop shorter than 150ms. Just
  keep the whole thing. The cut costs more than it saves.
- Emit `drop` entries rather than deleting segments, so the review UI can show
  what was removed and why.

---

## 6. Review gate

`python -m autocut review ep042` starts a FastAPI app on localhost.

Show a table of proposed drops: timestamp, duration, reason, confidence, the
transcript text being removed, and the text on either side. Checkbox to veto.
Audio scrub for anything under 0.8 confidence.

Vetoes write back to `edl.json` as `action: "keep"` with `"override": true`.
Re-running stage 3 must never clobber an override.

Ten minutes here beats three hours in Premiere. Don't skip building it.

---

## 7. Stage 4 - Cut and concat

Encode each kept segment separately, then concat with stream copy.

Per segment:

```
ffmpeg -ss {in} -to {out} -i work/EP/mezz/mezz.mkv ^
  -c:v libx264 -preset veryfast -crf 16 -g 1 -pix_fmt yuv420p ^
  -af "afade=t=in:st=0:d=0.025,afade=t=out:st={dur-0.025}:d=0.025" ^
  -c:a pcm_s16le -ar 48000 ^
  work/EP/segments/{id}.mkv
```

**The fades are mandatory.** Without them every splice produces an audible click.
25ms is short enough to be inaudible as a fade and long enough to kill the
discontinuity.

Note the `afade` start times are relative to the segment, not the source, because
`-ss` comes before `-i`.

Then concat:

```
ffmpeg -f concat -safe 0 -i work/EP/segments/list.txt -c copy work/EP/cut.mkv
```

`list.txt` uses forward slashes even on Windows, and quotes every path:

```
file 'C:/work/ep042/segments/s001.mkv'
file 'C:/work/ep042/segments/s003.mkv'
```

Stream copy works here only because every segment was encoded with identical
parameters. Don't vary the encoder settings per segment.

Segments are independently cacheable. Change one cut, re-encode one segment.

---

## 8. Stage 5 - Grade

Use a 3D LUT, not a stack of `eq` parameters.

```
ffmpeg -i work/EP/cut.mkv -vf lut3d=luts/skepticus_v1.cube ^
  -c:v libx264 -preset slow -crf 18 -c:a copy work/EP/graded.mkv
```

Build the LUT once in DaVinci Resolve (free). Grade a representative still from
your studio setup, export as `.cube`, drop it in `luts/`. Every episode gets the
same look with zero per-episode tuning.

Version the LUT filename. When you change the look in six months you want to
know which episodes used which grade.

---

## 9. Stage 6 - HyperFrames overlays

Each card type is a HyperFrames composition in `compositions/`. Props come from
the EDL.

Setup:

```
npx hyperframes init compositions/lower_third
```

Compositions are HTML with `data-start` and `data-duration` timing attributes,
animated with GSAP or CSS.

**Render with alpha.** The card must composite over your footage, not replace it.
HyperFrames captures PNG frames, which carry alpha. If the CLI's MP4 output
flattens it, take the PNG sequence and encode yourself:

```
ffmpeg -framerate 30 -i work/EP/overlays/ov001/frame_%05d.png ^
  -c:v libvpx-vp9 -pix_fmt yuva420p -lossless 1 ^
  work/EP/overlays/ov001.webm
```

`yuva420p` is the alpha-carrying pixel format. VP9 in WebM is the reliable
alpha-capable container FFmpeg handles well.

Keep compositions short. Render time scales linearly with frame count, and
headless Chrome capture is not fast. A four-second lower third is 120 frames and
renders in seconds. A twelve-minute full-frame overlay is 21,600 frames and will
take the better part of an hour.

Cache aggressively. Key the cache on a hash of the composition source plus the
props. Most episodes reuse the same cards with different text.

---

## 10. Stage 7 - Captions

Generate ASS from `words.json`. Do not use HyperFrames for this.

FFmpeg's subtitle burner handles a full episode in about ninety seconds. The
equivalent HyperFrames render is tens of thousands of browser frames for a
visually similar result.

ASS supports karaoke timing with `\k` tags, so word-by-word highlighting is
available:

```
Dialogue: 0,0:00:12.48,0:00:15.02,Skepticus,,0,0,0,,{\k23}The {\k31}argument {\k28}here
```

`\k` durations are in centiseconds. Compute from word start/end pairs, mapped
through `source_to_output`.

Group words into 2-4 second display chunks, breaking on punctuation and natural
pauses. Never break mid-clause if you can help it.

Style block in `styles/captions.ass.template`. Set your font, outline, shadow,
and margin once.

---

## 11. Stage 8 - Composite and encode

Overlays plus captions in one pass.

**Write the filter graph to a file.** Windows caps command lines at 8191
characters and a graph with 40 overlays exceeds that.

`work/EP/filter.txt`:

```
[0:v][1:v]overlay=enable='between(t,10.4,14.4)':x=0:y=0[v1];
[v1][2:v]overlay=enable='between(t,88.2,92.2)':x=0:y=0[v2];
[v2]subtitles=work/ep042/captions.ass:fontsdir=styles[vout]
```

Then:

```
ffmpeg -i work/EP/graded.mkv -i work/EP/overlays/ov001.webm -i work/EP/overlays/ov002.webm ^
  -filter_complex_script work/EP/filter.txt ^
  -map "[vout]" -map 0:a ^
  -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p ^
  -c:a aac -b:a 192k ^
  -movflags +faststart ^
  out/ep042.mp4
```

The `enable` timestamps are output-timebase, produced by running each overlay's
`source_time` through `source_to_output`.

Subtitle paths inside a filter graph need escaping on Windows. Use forward
slashes and relative paths where possible. This is a known source of cryptic
FFmpeg errors.

---

## 12. Stage 9 - QC report

Emit `out/ep042_report.md`:

- Source duration vs output duration, and total time removed
- Cut count, and count by reason
- Any drop with confidence below 0.8, listed with its transcript text
- Overlay list with resolved output timestamps
- A contact sheet: one frame per cut boundary, so you can eyeball for bad splices

```
ffmpeg -i out/ep042.mp4 -vf "select='eq(n,120)+eq(n,845)',scale=320:-1,tile=5x4" ^
  -frames:v 1 out/ep042_contact.png
```

---

## 13. Claude Code integration

### Slash command

`.claude/commands/edit.md`:

```markdown
Run the autocut pipeline for episode $ARGUMENTS.

1. Run `python -m autocut probe $ARGUMENTS` and `python -m autocut transcribe $ARGUMENTS`.
2. Read work/$ARGUMENTS/transcript/words.json and silence.json.
3. Author work/$ARGUMENTS/edl.json per the edl-authoring skill.
4. Validate with `python -m autocut validate $ARGUMENTS`. Fix any errors.
5. Stop. Tell me to run the review gate. Do not render.
```

The stop is deliberate. Rendering before review wastes twenty minutes.

### Skills

`edl-authoring` holds the schema, the source-timebase rule, the cut heuristics,
and the conservatism rules from section 5. `hyperframes-cards` holds your
composition conventions, prop schemas, and brand tokens.

### What Claude Code must not do

- Invoke `ffmpeg` directly. Always through `python -m autocut`.
- Write output-timebase values into the EDL.
- Edit files under `work/` other than `edl.json`.
- Touch `inbox/`.

---

## 14. Build order

Do not build this in one pass. Each step should produce a watchable video.

1. Probe, transcribe, and a hand-written two-segment EDL. Cut and concat. Prove
   the splice is clean and audio stays in sync. **This is the whole risk.**
2. Silence and filler removal from Claude Code. Add the review gate. At this
   point the pipeline is already saving you hours.
3. The LUT grade.
4. ASS captions.
5. HyperFrames cards, starting with one lower third.
6. Retake detection. Last, because it's hardest and least reliable.

Ship after step 2. Everything after that is polish on a working system.

---

## 15. Determinism checklist

You will want byte-identical reruns when debugging.

- Pin the FFmpeg build. Record `ffmpeg -version` output in the QC report.
- Pin the Whisper model revision, not just `large-v3`.
- Set `beam_size` explicitly and don't change it mid-project.
- Hash the raw MP4 on ingest. Store it in `probe.json`. If the hash changes, the
  cache is invalid.
- Pin `faster-whisper`, `ctranslate2`, and the CUDA runtime in `pyproject.toml`.

Given the Microsoft Store Python situation on this machine, build the venv from
python.org Python before starting. App execution aliases intercepting bare
`python` calls will produce confusing failures in a pipeline that shells out.

---

## 16. Known hard parts

**Retake detection.** Semantic near-duplicate matching over adjacent passages.
Expect false positives. Always review.

**Breath and mouth noise.** Whisper doesn't transcribe them, so they're invisible
to a transcript-driven EDL. A loud inhale sitting inside a "silence" will survive
the cut. If this bothers you, run a separate audio pass, or handle it in your
audio chain before ingest.

**Overlapping overlays.** Two cards active at once means chained `overlay`
filters and a z-order decision. Keep the EDL flat and non-overlapping until you
actually need it.

**HyperFrames render time.** Budget for it. Cache by composition hash plus props
hash. Most cards repeat across episodes.

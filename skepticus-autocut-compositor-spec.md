# Skepticus Autocut - Compositor Spec

Phase 2. Turns a cut, trimmed video into a produced show-format frame:
background wallpaper, keyed portrait of the speaker, a large content window,
and burned captions.

Assumes the Phase 1 trim pipeline is working: probe, transcribe, autoauthor,
review, cut.

Source and output are both 4K (3840x2160).

---

## 1. What changes architecturally

Phase 1 treated overlays as things placed on top of footage. This format is
different. The raw recording is no longer the frame. It's one layer inside a
composed frame.

That means:

- **FFmpeg does the layout.** Chromakey, crop, scale, overlay, caption burn.
  HyperFrames can't composite live video.
- **HyperFrames renders chrome, not layout.** Static frames around the content
  window, animated lower thirds, transition wipes. Rendered once as PNG with
  alpha and reused across episodes.
- **The compositor stage replaces the overlay stage.** Overlays become one
  input to compositing rather than a stage of their own.

Pipeline shape:

```
cut.mkv
  → keyed speaker layer          (chromakey + despill + crop)
  → content track                (images/clips → timed video layer)
  → composite                    (background + content + speaker)
  → captions burn                (ASS)
  → final.mp4
```

---

## 2. Layout config

Everything declarative from day one. No hardcoded geometry.

`config/layout.yaml`:

```yaml
canvas:
  width: 3840
  height: 2160
  fps: from_source        # never hardcode; read probe.json

background:
  image: assets/backgrounds/skepticus_v1.png
  # must match canvas dimensions exactly

speaker:
  side: right             # left | right
  rect: [2640, 240, 1080, 1680]   # x, y, w, h in canvas space
  source_crop: auto       # auto = center vertical strip at target aspect
  key:
    color: "0x00b140"
    similarity: 0.12
    blend: 0.05
    despill: true
  corner_radius: 24       # 0 to disable

content:
  rect: [160, 240, 2400, 1680]
  fit: contain            # contain | cover
  background: "#0b0b0d"   # letterbox fill when fit=contain
  gap_behavior: hold      # hold | background | expand_speaker
  transition:
    type: crossfade       # cut | crossfade
    duration: 0.35

captions:
  enabled: true
  style: styles/captions.ass
  margin_bottom: 120
  word_highlight:
    enabled: true
    base_color: "&HFFFFFF"      # white
    highlight_color: "&H40FF40" # bright green
```

Colors in ASS are `&HBBGGRR`, not RGB. Easy to get backwards.

Every geometry value lives here. Changing the speaker to the left side, or
resizing the content window, should be a config edit and a re-render. Nothing
else.

---

## 3. Speaker layer

Crop a vertical strip from the 4K frame, key it, scale to the target rect.

At 3840x2160, a 1080x1680 target is a 0.643 aspect. The matching source crop is
1389x2160, so you're scaling down slightly. Sharp.

```
[0:v]crop=1389:2160:1225:0,
     chromakey=0x00b140:0.12:0.05,
     despill=type=green,
     scale=1080:1680:flags=lanczos
[spk]
```

The crop x offset centers you in frame. **Measure it once from a real
recording** rather than assuming center. Where you actually sit relative to
frame center is a fixed property of your setup, and it should live in config as
`source_crop: [x, y, w, h]` once measured.

Order matters. Crop before keying so you're not keying 3x the pixels you need.

**Corner radius**, if enabled, needs an alpha mask via `geq` or a pre-rendered
PNG mask overlaid with `alphamerge`. The PNG mask is faster and easier to get
right. Generate it once at the speaker rect's dimensions.

---

## 4. Content track

The content window is the part that will eventually be fed by b-roll
automation. For now it's a folder and a manifest.

`inbox/<episode_id>_content/` holds images. `content.json` sits beside the EDL:

```json
{
  "items": [
    {"file": "temple_mount.jpg",   "source_time": 42.0,  "duration": 12.0},
    {"file": "codex_sinaiticus.png","source_time": 61.5, "duration": 8.0},
    {"file": "clip_excavation.mp4", "source_time": 96.0, "duration": 15.0}
  ]
}
```

**`source_time` is in source timebase**, same rule as overlays in Phase 1. The
compositor maps through `source_to_output` after cuts. Getting this wrong puts
every image after the first cut at the wrong moment.

Build the content track as its own video layer first, then composite it. Don't
try to do it inside one giant filter graph.

```
content_track.mkv   # full output duration, canvas-sized content rect,
                    # letterboxed, with gap behavior applied
```

Separate stage, separately cacheable. You'll iterate on content far more often
than on the key.

**Gap behavior**, per config:

- `hold` (default) - last image stays until the next one starts. Reads as
  intentional.
- `background` - content rect shows the background image through.
- `expand_speaker` - speaker rect grows to fill. More work; defer.

**Transitions.** Crossfade between items via `xfade`. Keep the default short.
Anything over 0.5s draws attention to itself.

**Fit.** `contain` letterboxes and never crops the image. `cover` fills and
crops. Default to `contain` - for a channel dealing with manuscripts, maps, and
paintings, cropping the subject is worse than letterboxing.

---

## 5. Composite

Three layers, bottom to top: background, content, speaker.

```
[bg][content] overlay=x=160:y=240        [tmp];
[tmp][spk]    overlay=x=2640:y=240       [out]
```

Background is a static image looped to the output duration:

```
-loop 1 -i assets/backgrounds/skepticus_v1.png
```

Write the filter graph to a file and use `-filter_complex_script`. Same Windows
8191-char limit as Phase 1.

---

## 6. Captions

ASS, generated from `words.json`, burned at composite time.

**Word highlight** uses `\k` karaoke tags. White base, bright green current
word, per the existing look.

```
Dialogue: 0,0:00:12.48,0:00:15.02,Skepticus,,0,0,0,,{\k23}The {\k31}argument {\k28}here
```

`\k` durations are centiseconds and **accumulate across the line**. Compute
cumulative positions and take differences. Rounding each word independently
lets error compound, and the highlight slides off the audio by the end of a
long caption.

Style block in `styles/captions.ass`, driven by the layout config. Font, size,
outline, shadow, colors, and bottom margin all configurable per objective 4.

Turning captions off is `captions.enabled: false`, which skips the burn
entirely rather than rendering an empty track.

**Font size at 4K.** A size that looks right at 1080p will be tiny. Scale
accordingly, and check on a phone, since that's where most viewing happens.

---

## 7. Audio chain

Not part of compositing, but it's the other thing DaVinci's Audio Assistant
does for you, and it's deterministic enough to automate.

```
-af "highpass=f=80,
     afftdn=nf=-25,
     acompressor=threshold=-18dB:ratio=3:attack=5:release=120,
     loudnorm=I=-14:TP=-1.5:LRA=11"
```

- `highpass` kills rumble and HVAC.
- `afftdn` is broadband noise reduction. Start gentle; too much sounds
  underwater.
- `acompressor` evens out delivery.
- `loudnorm` targets YouTube's -14 LUFS.

`loudnorm` is two-pass for accuracy. Run the analysis pass, capture the JSON,
feed it to the second pass. Single-pass works but is less precise.

This won't match the Audio Assistant exactly. It should get most of the way,
deterministically, with values you can tune once and reuse.

---

## 8. Render cost

4K compositing is expensive. Chromakey, two scales, two overlays, and a
subtitle burn across 56 minutes of 3840x2160 is hours of CPU work.

**Mitigations:**

**`--preview` flag.** Render at 1080p with `-vf scale=1920:1080` early in the
chain and a fast preset. You'll re-render constantly while tuning geometry, and
a 5-minute preview beats a 3-hour full render for that.

**NVENC for the final encode.** `-c:v hevc_nvenc -preset p5 -cq 22`. Helps the
encode; filtering stays on CPU.

**Cache the speaker layer.** Keying doesn't change when you swap a content
image. Separate stage, separate cache.

**Render a range.** `--range 300:360` to composite one minute for checking a
specific moment.

Budget the full render as an overnight job. Design the iteration loop so you
almost never need it.

---

## 9. Build order

1. **Layout config plus static composite.** One background, one placeholder
   image in the content rect, keyed speaker. No timing, no captions. Render 10
   seconds. Get the geometry right by looking at it.
2. **Speaker layer as its own cached stage.** Measure the real crop offset.
   Tune key parameters against actual footage.
3. **Content track from `content.json`.** Timing, gap behavior, transitions.
4. **Captions with word highlight.**
5. **Audio chain.**
6. **`--preview` and `--range`.** Sooner if iteration gets painful, which it
   will.

Steps 1 through 4 give you a complete produced video from a raw recording plus
a folder of images. That's the format.

---

## 10. What this unblocks

Once the content window exists and is fed by a manifest, b-roll automation has
somewhere to deliver to. The seven-stage pipeline with CLIP reranking produces
`content.json` and a folder of images. Nothing about the compositor changes.

Same for retake detection. It writes drops into the EDL, and everything
downstream is unaffected.

Build the frame first. The interesting work plugs into it.

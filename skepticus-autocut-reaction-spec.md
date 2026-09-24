# Skepticus Autocut - Reaction Format Spec

Adds a second episode format: reacting to a source video, with the source
playing in the content window and the host in the speaker window.

Same tool, same compositor. Playback boundaries come from spoken cues.

**This revision replaces transcript-based playback detection.** That approach
produced a large false-positive class: quoting or paraphrasing the source looks
identical to playing it, and on a reaction channel that happens constantly.

---

## 1. The format

Two input files.

- **The host recording.** Talking head, recorded while the source video plays on
  screen. The mic picks up the source audio as bleed.
- **The source video.** The thing being reacted to, as a clean file.

The recording alternates: host talks, source plays, host talks, source plays.
Never both at once.

Output:

- **Speaker window** shows the host throughout, including during playback.
  Visual reactions stay visible.
- **Content window** shows the source video during playback, frozen on its last
  frame during commentary.
- **Audio** switches between host mic and source. **The host's mic is fully
  muted during playback**, so a cough or chair creak never reaches the output.

---

## 2. Spoken cues

The host declares the boundaries out loud. Same principle as the retake cue.

- **End-commentary cue** - said just before starting playback. Host audio stops,
  source unfreezes and plays.
- **Start-commentary cue** - said just after pausing playback. Source freezes,
  host audio resumes.

Both cues are dropped from the output along with any dead air around them.

### Choosing the cues

**Avoid "commentary."** It's a normal word on a channel discussing biblical
commentaries, and a false hit would invert the audio state for everything
downstream. If the natural phrasing is kept, match the complete phrase rather
than any single word.

Safer: two distinctive words that never occur in the subject matter. Same test
as the retake cue - verify Whisper transcribes them consistently before relying
on them (section 10).

```yaml
reaction:
  cue_playback_start: "end my commentary"
  cue_playback_stop: "begin my commentary"
```

### Two cues, not one toggle

A single toggle cue would invert everything after a miss. Two distinct cues
keep errors local: a missed cue breaks one boundary, not the rest of the
episode.

### Alternation is validated

Cues must alternate. Two starts in a row, or two stops, means one was missed or
misheard. **Fail loudly** and report the timestamps. Do not guess.

The recording opens in commentary and may end in either state. Both are handled.

---

## 3. Source position

With cues giving exact host-side boundaries, the source offset follows from
arithmetic rather than search.

Pausing stops the source clock. So playback segment N resumes where segment N-1
stopped:

```
source_in(1) = 0
source_in(N) = source_out(N-1)
source_out(N) = source_in(N) + (host_out(N) - host_in(N))
```

That's the estimate. Two things can break it, and both are handled by
refinement rather than by assumption.

**Latency.** You say the cue, then reach for the mouse. The gap between the cue
ending and playback actually starting is dead air in the recording, and it
shifts the boundary by a fraction of a second.

**Seeking.** Skipping a dull stretch or rewinding to re-watch breaks the
cumulative chain for every segment after it.

---

## 4. Refinement

Cross-correlate the host bleed against the source audio, searching a narrow
window around the arithmetic estimate rather than the whole source.

This is a much easier problem than the original design posed. Correlation is
confirming an offset, not discovering one, so a weak peak no longer means
failure - it means the estimate stands.

- Band-limit both to roughly 300Hz to 3.5kHz. The bleed has been through
  speakers, a room, and a mic.
- Correlate on a spectral envelope, not raw samples.
- Search roughly ±3s around the estimate.
- Sample several windows across the segment and take the consensus lag. Windows
  disagreeing is a signal in itself.

**The chain is self-healing.** Each segment's estimate is built from the
*corrected* offset of the previous segment, not from the original arithmetic.
So a seek shifts one segment and the rest follow from the new position.

**Large disagreement means a seek.** If refinement lands more than a few seconds
from the estimate, the source was moved. Accept the corrected value, flag the
segment, and continue from there.

---

## 5. The playback map

`work/<ep>/playback.json`:

```json
{
  "version": 1,
  "episode_id": "reaction001",
  "source_file": "inbox/reaction001_source.mp4",
  "segments": [
    {
      "id": "pb001",
      "host_in": 47.2,
      "host_out": 112.8,
      "source_in": 0.0,
      "source_out": 65.6,
      "offset_source": "cumulative",
      "refinement_delta": 0.12,
      "seek_detected": false
    }
  ]
}
```

`host_in` and `host_out` are in host source timebase. The compositor maps them
through `source_to_output` after cuts.

`source_out - source_in` must equal `host_out - host_in`. Assert it.

---

## 6. Constraints on autoauthor

**No drops inside playback regions.** Not silence, not filler, not retakes. The
source was edited by its creator, and cutting inside it desynchronises
everything after that point.

`autoauthor` reads `playback.json` and excludes those spans from cut discovery.
A drop straddling a boundary is truncated to the commentary side.

The cues themselves *are* dropped, along with dead air between a cue and the
actual state change.

Retake cues still work during commentary. Flubs while talking are normal.

Log the split: how much of the episode is playback versus commentary, and how
much was cut from the commentary portion only.

---

## 7. Content track

Playback segments become video items in `content.json`:

```json
{
  "file": "inbox/reaction001_source.mp4",
  "source_time": 47.2,
  "duration": 65.6,
  "clip_in": 0.0
}
```

`clip_in` is the in-point within the source file. The content track already
handles video items and `fit: contain`.

### Commentary segments

Default is a held frame of the source, frozen where playback stopped. That's the
format convention and it reads as intentional.

```yaml
reaction:
  talk_window: held_frame    # held_frame | cards | blank
```

`cards` routes commentary through the normal shot list and visuals pipeline.
Worth trying for long commentary stretches, at the cost of breaking convention.

### Opening commentary: thumbnail instead of frame 1

The pre-roll span (output 0 until the first playback segment starts) shows the
source's YouTube thumbnail instead of a frozen frame 1 — a still of frame 1 is
usually a black frame or a mid-scroll blur, where the thumbnail is composed to
read at a glance.

- Auto-detected at `inbox/<ep>_thumb.<ext>`, same convention as `<ep>_source.<ext>`.
- `reaction.thumbnail` overrides with an explicit path (relative to the repo
  root). Falls back to auto-detection if the configured path doesn't exist.
- Optional: falls back to the frozen first frame (the prior behaviour) when
  no thumbnail is found either way.
- Crossfades into pb001's first frame at the content track's normal
  `transition.duration` — no special-cased handoff.
- Thumbnails are 16:9; the content rect is not, so `fit: contain` letterboxes
  it. That's expected, not a bug to fix.

```yaml
reaction:
  thumbnail: inbox/reaction001_thumb.jpg   # optional; else auto-detected
```

---

## 8. Audio track

Assembled from two sources.

- **Commentary:** host mic.
- **Playback:** source file's clean audio. Host mic muted completely, not ducked.

Build as its own cached stage producing a full-length track, then mux. Same
structure as the content track.

**Crossfade every boundary.** 50 to 100ms. Hard switches between two acoustic
spaces are jarring in a way a same-source splice isn't.

**Level matching stays on even when audio processing is off.** The source and
your mic will not naturally match, and that mismatch is far more noticeable than
any processing artifact. Measure both and apply a gain offset.

```yaml
reaction:
  match_levels: true
  source_gain_db: 0
  crossfade_ms: 75
```

---

## 9. Config

```yaml
episode:
  type: reaction           # monologue | reaction
  source_file: inbox/reaction001_source.mp4
```

Existing `speaker.side` and rect geometry carry over unchanged.

---

## 10. Verification before relying on it

Before recording a full episode, record ninety seconds containing both cues
three or four times at different speeds. Run transcribe and grep `words.json`.

You're checking that Whisper renders each cue identically every time.
Inconsistent transcription is a silent failure - a missed cue means the audio
state never switches.

Do this before changing recording habits, not after.

---

## 11. Review gate

A row per playback segment, showing:

- Host span and source span
- Whether a seek was detected
- Refinement delta from the arithmetic estimate
- **A short rendered clip from the start of the segment** - source video with
  host bleed audio over it, so a bad offset is visible as lips disagreeing

That pairing matters. Source video with source audio would always look synced
and prove nothing.

Provide a nudge control: shift `source_in` by a few frames and re-render the
check clip. Manual correction is often faster than re-running refinement.

Flag any segment where cue alternation was repaired or a seek was detected.

---

## 12. Build order

1. **Cue detection and alternation validation.** Deterministic. Verify against a
   real recording.
2. **Cumulative source position.** Arithmetic, no correlation.
3. **Verification harness:** two-second check clips. Build early, use constantly.
4. **Refinement** by cross-correlation with self-healing chain.
5. **autoauthor constraints.**
6. **Content track** with `clip_in`.
7. **Audio track** with switching, crossfades, level matching.
8. **Review gate** with video preview and nudge.

Steps 1 through 3 may be enough on their own. If latency is small and consistent,
the arithmetic estimate could be accurate enough without refinement. Check the
clips before building step 4.

---

## 13. Recordings made without cues

An existing recording has no cues and can't be segmented this way. Options:

- **Re-record.** Cleanest, and it is what it sounds like.
- **Manual marking.** A scrub-and-mark UI. Tedious at scale - a long episode can
  have seventy or more segments.
- **Acoustic discrimination.** Bleed audio has been through speakers and a room;
  direct mic audio hasn't. Reverb, high-frequency rolloff, and noise floor are
  properties of the recording chain rather than the voice, so they should
  separate the two regardless of who is speaking or what is said. Promising, but
  it needs validation against hand-labelled examples before it's trusted.

Only worth pursuing for an episode already recorded. All future recordings use
cues.

---

## 14. Known hard parts

**Forgetting a cue.** Alternation validation catches it, but the fix is manual.
The failure is loud, which is the right direction.

**Cue latency.** The gap between saying the cue and the state actually changing
varies. Refinement absorbs it; without refinement, boundaries may sit a fraction
of a second early or late.

**Headphones break the refinement step.** Cue detection still works, and
arithmetic still works, but there'd be no bleed to correlate against.

**Very short playback segments.** Little audio for refinement. The arithmetic
estimate carries them.

**Source audio in the mix.** During playback the output carries the source's
audio at full level. That's a platform-policy question rather than a pipeline
one.

**Frozen frames during long commentary.** The format's main visual weakness.
`talk_window: cards` is the escape hatch.

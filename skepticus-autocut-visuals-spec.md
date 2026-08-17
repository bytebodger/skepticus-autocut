# Skepticus Autocut - Visuals Spec

Phase 3. Produces the visual thread that runs through the content window:
rendered cards, diagrams, and vector scenes, with generated imagery as a
supplement.

Depends on Phase 2 (compositor) being complete. Changes nothing downstream.

**This supersedes the earlier illustration spec, which had the balance wrong.**

---

## 1. HyperFrames is the primary engine

The content window is a slide surface, not an image slot. Think of it as a
PowerPoint deck brought to life: the visual thread of the argument, rendered.

**HyperFrames renders the majority.** HTML, CSS, and SVG to deterministic
frames. Text is text, so it's perfect. Data is plotted, so it's correct. Vector
art is vector art, so it's crisp at 4K and identical every render.

**Generated images are the exception.** A handful per episode, for scenes that
genuinely need depicting and can't be composed from parts.

### Why this is the right split, not just the requested one

Flat, limited-palette illustration is what vector graphics *are*. Asking a
diffusion model to approximate the look of an SVG is backwards: you pay per
image, get variance you have to review, and land near a result that a vector
renderer produces exactly.

The properties follow from that:

- **Determinism.** Same input, same output. Matches how the rest of this
  pipeline is built.
- **Zero marginal cost.** Regenerate 200 cards for free.
- **Perfect consistency.** Every card uses the same tokens. Style holds
  automatically instead of being something you police at review.
- **Native animation.** Bars grow, lines draw, bullets appear, elements move on
  authored paths. This is what makes it feel animated rather than like a
  slideshow.
- **Restyling is a stylesheet change.** Swap tokens, re-render the episode.
  That's the per-video style swapping you asked for, and it's near-free.

---

## 2. Composition taxonomy

The earlier spec offered only illustration or infographic. That binary is why 57
of 60 shots came back as illustrations: anything that wasn't a chart fell
through to the image model by default.

Shot list authoring picks from a real set of composition types instead.

### Text and argument

- `pull_quote` - a passage with attribution. Scripture, a scholar, an apologist
  being quoted accurately. **External sources only.**
- `statement` - the speaker's own words, set large for emphasis. No quotation
  marks, no attribution line. This is a point being underlined, not a citation.
- `term_card` - a word and its definition. Hebrew or Greek terms, technical
  vocabulary.
- `comparison` - two or three things side by side. Almah versus betulah.
  Competing translations. Two positions.
- `bullet_reveal` - enumerated points appearing in sequence as you speak them.
- `argument_diagram` - premises leading to a conclusion. Branching for a
  dilemma, chains for causal claims.
- `title_card` - a section marker.

### Data and time

- `chart` - line, bar, stacked. Only from spoken numbers (section 5).
- `timeline` - dated events in sequence.
- `table` - structured comparison across several rows.

### Space and structure

- `map` - SVG geography. The ancient Near East, the spread of a movement,
  council locations.
- `diagram` - relationships, hierarchies, structures. A manuscript stemma. A
  canon's formation.

### Pictorial

- `vector_scene` - a scene composed from the SVG component library (section 4).
  Flat, illustrative, on-brand, deterministic.
- `generated_image` - the escape hatch. Only when a scene needs depicting and
  can't be composed. Section 6.

### None

- `none` - no visual. The compositor's `hold` behaviour covers it.

### Selection guidance for shot list authoring

Prefer the most structured type the passage supports. In rough priority order:

1. If the passage quotes an external source, `pull_quote`.
2. If it's the speaker's own line worth underlining, `statement`.
3. If it defines a term, `term_card`.
4. If it contrasts two things, `comparison`.
5. If it enumerates, `bullet_reveal`.
6. If it makes a structured argument, `argument_diagram`.
7. If it contains spoken numbers, `chart`.
8. If it walks through time, `timeline`.
9. If it's about place, `map`.
10. If it describes a scene the library can compose, `vector_scene`.
11. If it describes a scene the library cannot compose, `generated_image`.
12. Otherwise `none`.

**Never attribute the speaker's own words to the speaker.** Words spoken in the
episode are never a `pull_quote`, never get quotation marks, and never get a
byline. Quoting yourself back at the audience with your own name underneath
reads as arrogant. Use `statement`.

### Continuity

**The content window must never be empty.** A large empty panel beside a talking
head looks broken, and the instinct to avoid holding one graphic too long is
weaker than the need for something to be there.

Two rules follow:

- **Shot durations are contiguous.** Each shot runs until the next one starts.
  Authoring should not leave gaps; a shot's duration extends to fill.
- **`none` is not a gap.** Choosing no visual for a passage means the previous
  card holds through it, not that the panel goes empty. `none` reduces churn,
  it doesn't create voids.

If the pacing feels static, the answer is a different composition or a longer
animation, not an empty panel.

### Anti-flood caps

The classifier floods into whichever category has the lowest bar. The old binary
put 57 of 60 shots into illustration; adding `statement` put 51 of 88 into it
next. Prompt wording alone does not hold this — it must be enforced structurally
in authoring:

- **`statement` is capped near 10% of shots** and reserved for genuine thesis or
  punchline lines — the single sentence that carries a section, not ordinary
  narration. Keep only the strongest; drop the rest to `none`.
- **Never two statements in a row.** The rebalance pass enforces it.
- **`none` is a good outcome, not a failure.** Ordinary narration that doesn't fit
  a structured type is `none`. With alpha-aware hold and contiguous durations the
  previous card stays up, so `none` costs nothing. The captions already show the
  words; a statement just repeats them larger.
- **No single type over ~25% of shots.** If one type dominates, the rebalance pass
  drops its weakest members back toward the cap.

**`generated_image` requires justification.** The shot record carries a
`why_generated` field explaining what made a vector scene impossible. If that
field is weak, the classification is probably wrong.

Expect a healthy episode to land somewhere near 70% structured types, 20%
vector scenes, 10% generated. If generated exceeds 20%, the taxonomy isn't
being used properly.

---

## 3. Compositions

Each type is a HyperFrames composition in `compositions/`, taking props and
using the active style's tokens.

```
compositions/
  pull_quote/         built
  statement/          built
  term_card/          built
  comparison/         built
  bullet_reveal/      built
  argument_diagram/   built
  title_card/         built
  chart/              built (bar; line/stacked to follow)
  timeline/           built
  table/              built
  map/                built (stylised; real geography awaits the component library, section 4)
  diagram/            built
  vector_scene/       needs the SVG component library (section 4)
```

Each Phase-3 content card declares `<meta name="hyperframes-surface"
content="content-card">` so the render stage can tell it apart from the Phase-1
overlay compositions that share the `compositions/` directory (e.g. `title_card`).

Build them incrementally. `pull_quote`, `comparison`, and `bullet_reveal` alone
cover a surprising share of a talky episode, and they're the three the `context`
shot list already surfaced organically.

### Animation

Every composition animates. This is the difference between a deck and a video.

- `pull_quote` - text fades in by line, attribution last.
- `bullet_reveal` - items appear on a timed sequence.
- `chart` - bars grow, lines draw left to right.
- `comparison` - sides arrive in turn, not together.
- `timeline` - the playhead advances.
- `vector_scene` - elements drift, parallax, slow push.

Time reveals against the transcript where possible. A bullet appearing as you
say it is much better than five bullets arriving at once.

### Duration

Compositions receive the shot's duration and fit themselves to it. A bullet
reveal with five items over 20 seconds paces differently than the same five over
8 seconds. Don't render a fixed-length animation and hope.

---

## 4. The vector component library

This is what makes `vector_scene` viable, and it's the piece that replaces most
of what generated images were doing.

`style/<name>/components/` holds SVG parts: figures, buildings, landscape
elements, objects, symbols. Flat, in-palette, designed to combine.

A scene composes several parts with placement, scale, and layering:

```json
{
  "composition": "vector_scene",
  "props": {
    "background": "hillside",
    "elements": [
      {"part": "figure_standing_robed", "x": 0.42, "y": 0.55, "scale": 1.0},
      {"part": "crowd_seated", "x": 0.5, "y": 0.78, "scale": 1.4}
    ]
  }
}
```

That gives you the Sermon on the Mount without an image model.

### Building the library

You need maybe 40 to 80 parts to cover a channel like yours. Figures in a few
poses, architectural elements, landscape pieces, scrolls and codices, common
symbols.

Three ways to get them, and mixing is fine:

- **Draw them.** Most control, most work.
- **Generate then vectorize.** Produce flat images, trace to SVG, clean up. The
  vectorizing step is what makes them consistent and reusable.
- **License a flat icon or illustration set** and restyle to your palette.

This is a real upfront investment. It's also the thing that makes every
subsequent episode nearly free, and it compounds. Start with 15 or 20 parts
covering your most common subjects and grow it as gaps appear.

The library lives inside the style directory, so swapping styles swaps
components too.

---

## 5. Data handling

**The transcript is the only data source.** If you said a number on camera, it
can be charted. If you didn't, there is no chart.

If a passage warrants a chart but the numbers weren't spoken, the shot is
demoted at authoring time to another composition type, or dropped. Never a
placeholder, never a blocked render, never "source needed" on screen. You should
be able to start a render and walk away.

**Never invent data.** Not estimates, not plausible figures, not interpolations
between two real points.

If you want a chart from data you didn't say aloud, add it to `shotlist.json` by
hand before rendering. A deliberate act, not something the pipeline pushes you
toward.

---

## 6. Generated images

The escape hatch, not the engine.

Legitimate uses: a historical scene with detail the component library can't
compose, an atmospheric establishing shot, a subject that appears once and
doesn't justify new components.

If a subject recurs across episodes, that's a signal to build components for it
instead.

### Engine

Because this is now a minority of shots, the choice matters much less than it
did. Keep the backend interface and defer the decision.

Worth knowing: the OpenAI image API requires a separate paid API account from a
ChatGPT subscription, billed per image. Gemini's models accept reference images
directly for style conditioning. Either works at this volume.

### Style consistency

Generated images must sit beside vector scenes without clashing. Reference-image
conditioning against your own approved outputs is the mechanism. Put a handful
of rendered vector scenes in `style/<name>/references/` so generated shots are
conditioned toward the vector look rather than away from it.

The anti-photorealism negative prompt stays mandatory (section 8).

---

## 7. Style as a swappable object

```
styles/
  default/
    STYLE.md          # prose statement, prompt fragments, named variants
    tokens.css        # palette, type scale, spacing - drives all compositions
    components/       # SVG parts for vector_scene
    references/       # approved images for generation conditioning
```

Episode config selects one by name. Swapping restyles every card, chart, and
scene in the episode.

Because compositions read from `tokens.css`, a restyle is a re-render with no
regeneration. That's the per-video flexibility you asked for, and with
HyperFrames primary it costs almost nothing.

---

## 8. Content policy

Most of what needed policing was a consequence of generated imagery. With
rendering primary, the surface shrinks.

**Illustration, never photorealism.** A drawing claims nothing; a photorealistic
render claims to be a record. Vector compositions are inherently interpretive.
For generated shots, anti-photorealism terms stay in the negative prompt,
enforced mechanically.

**No depictions of Muhammad.** Categorically different from the Abraham case.
Stylization solves affront and gore; it doesn't address a prohibition on
depiction as such. Route Islamic subject matter to architecture, geometric
abstraction, maps, or text compositions. Applies to vector scenes as much as
generated images.

**Named living people are an editorial call.** Stylized caricature is closer to
editorial cartooning than fabrication, but it reads as mockery, and your stated
value is steelmanning before critique. Alternatives: their book cover, their
argument diagrammed, an empty lectern. Set the policy in config.

---

## 9. Pipeline

```
words.json
  → shotlist.json           (LLM authoring, composition type per shot)
  → render                  (HyperFrames, per composition)
  → generate                (image model, generated_image shots only)
  → review
  → motion                  (zoompan for generated; native for rendered)
  → content.json + assets
```

Render and generate are separate stages, separately cached. Most episodes will
barely touch generate.

**Caching.** Compositions cache on composition name plus props hash plus style
version. Once your compositions stabilise, most renders across episodes will be
cache misses only on props, and re-renders after a style change are the cheap
path.

**Render cost.** HyperFrames captures frames through headless Chrome, which is
not fast. A 20-second animated card at 24fps is 480 frames. Sixty of those is
real time. Budget for it, cache aggressively, and keep compositions short.

---

## 10. Review gate

Extends the existing pattern. Per shot:

- The transcript passage that motivated it
- Position on the timeline
- The rendered composition, playing
- For generated shots, candidates side by side
- For charts, the transcript numbers next to the rendered figures
- Accept, reject, or edit props and requeue

Rendered compositions need far less review than generated images, because they
can't be off-style or malformed. Most review time goes to whether the *choice*
was right, not whether the output is usable.

---

## 11. Build order

1. **Style tokens.** `tokens.css` for the default style. Palette, type, spacing.
   This drives everything.
2. **Three compositions:** `pull_quote`, `comparison`, `bullet_reveal`. Verify
   HyperFrames output with alpha at 4K.
3. **Rework shot list authoring** for the full taxonomy. Re-run on `context` and
   check the distribution. If `generated_image` exceeds 20%, the guidance needs
   tightening.
4. **Render stage** with caching, wired to `content.json`.
5. **Checkpoint:** run a few minutes through the compositor and watch it.
6. **More compositions** as the shot lists demand them.
7. **Component library** and `vector_scene`.
8. **Generation stage** last, and small.

Step 3 is the one to get right. The taxonomy determines everything downstream,
and it's currently wrong.

---

## 12. Known hard parts

**Composition coverage.** Early episodes will want types you haven't built. Let
the shot lists tell you what to build rather than guessing.

**The component library is real work.** 40 to 80 SVG parts is a project. It's
also the highest-leverage asset in this pipeline, since it makes every future
episode nearly free.

**HyperFrames render time.** Headless Chrome frame capture is slow. Cache
aggressively and keep animations short.

**Text-heavy fatigue.** Too many text cards in a row reads as a lecture slide
deck. Mix in vector scenes and maps for rhythm. The coherence pass should watch
for runs of the same type.

**Timing reveals to speech.** Animating a bullet as you say it needs word-level
timing from `words.json` passed into the composition. Worth doing; it's the
difference between synced and approximate.

---
name: hyperframes-cards
description: Conventions for Skepticus HyperFrames overlay cards — composition list, prop schemas, brand tokens, and timing rules. Use when adding or editing overlays in an EDL, or authoring a new composition.
---

# HyperFrames cards

Overlays are HyperFrames compositions under `compositions/`. Each is HTML with
`data-start` / `data-duration` timing attributes, animated with CSS (or GSAP).
Props come from the EDL. They render to a PNG sequence (which carries alpha) and
are encoded to an alpha-capable VP9 WebM so the card composites over footage.

## Available compositions

| composition     | required props | optional props        | default duration |
|-----------------|----------------|-----------------------|------------------|
| `lower_third`   | `line1`        | `line2`, `x`, `y`     | 4.0s             |
| `title_card`    | `title`        | `subtitle`, `x`, `y`  | 5.0s             |
| `chapter_marker`| `name`         | `number`, `x`, `y`    | 3.5s             |

Each composition ships a `props.schema.json` — match it. `x`/`y` are the overlay
offset in the final composite (default 0,0 = full frame).

## Referencing a card in the EDL

```json
{"id": "ov001", "composition": "lower_third", "source_time": 15.0,
 "duration": 4.0, "props": {"line1": "Skepticus", "line2": "Episode 42"}}
```

`source_time` is in the SOURCE timebase (see the edl-authoring skill). It must
land inside a KEPT segment or the card never appears — validation enforces this.

## Brand tokens

- Accent: `#ffcf33` (Skepticus yellow)
- Text: `#ffffff` on footage, always with shadow for legibility
- Font: Montserrat (weights 500 / 800 / 900)

## Timing rules

- Keep cards short. Render time scales linearly with frame count and headless
  Chrome capture is slow. A 4s lower third is 120 frames — seconds. A 12-minute
  full-frame overlay is 21,600 frames — most of an hour.
- Keep overlays flat and non-overlapping until you actually need z-ordering.
- Rendering is cached on composition source + props + timing. Reusing the same
  card with different text across episodes is cheap.

## New compositions

`npx hyperframes init compositions/<name>`. Read props from `window.PROPS` with
sensible fallback defaults so the card also previews standalone in a browser.
Add a `props.schema.json`.

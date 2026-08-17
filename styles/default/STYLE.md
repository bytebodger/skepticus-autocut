# Style bible — `default` (placeholder)

A swappable per-episode style. Select it in the episode config with `style: default`.
This is a placeholder to develop against — replace every value below during
Phase 3A (visuals spec section 3). The illustration/infographic split reads as
one channel only if both halves share this palette.

## Style statement

> Placeholder. Two or three sentences describing the look in prose. The governing
> constraint: **illustration, never photorealism** — nothing produced here should
> be mistakable for a photograph (visuals spec section 3).

## Prompt fragments

- **Style fragment** (appended to every generation, never varies):
  `flat vector illustration, limited palette, simple shapes, obviously drawn, editorial`
- **Negative fragment** (always includes anti-photorealism terms):
  `photograph, photorealistic, 3d render, realistic, hyperdetailed, text, letters, watermark`

## Named variants

Selectable per shot via a shot's `variant`; empty by default.

| variant     | fragment (placeholder)                         |
|-------------|------------------------------------------------|
| `engraving` | `pen-and-ink engraving, cross-hatching`        |
| `woodcut`   | `bold woodcut print, high contrast`            |
| `diagram`   | `clean technical diagram, labelled, schematic` |

## Palette

Four to six colours, held across illustrations **and** infographics (must match
`tokens.css`).

- `--ink`    `#1a1a1a`
- `--paper`  `#f4f1ea`
- `--accent` `#c8462d`
- `--cool`   `#2d6a8c`
- `--muted`  `#8a8578`

## Parameters

- model: `<local SDXL + LoRA, TBD>`
- sampler / steps / CFG / aspect ratio: `<TBD>`
- LoRA version: `<TBD>`

## Notes on what fails

- Placeholder. Record subjects/compositions the style handles badly here as the
  bible develops; this feeds back from the review gate.

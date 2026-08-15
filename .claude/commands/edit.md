Run the autocut pipeline for episode $ARGUMENTS.

1. Run `python -m autocut probe $ARGUMENTS` and `python -m autocut transcribe $ARGUMENTS`.
2. Read `work/$ARGUMENTS/transcript/words.json` and `work/$ARGUMENTS/transcript/silence.json`.
3. Author `work/$ARGUMENTS/edl.json` per the `edl-authoring` skill.
   - You may start from the deterministic baseline: `python -m autocut autoauthor $ARGUMENTS`,
     then refine it (false starts, retakes, overlay placement).
4. Validate with `python -m autocut validate $ARGUMENTS`. Fix any errors.
5. Stop. Tell me to run the review gate (`python -m autocut review $ARGUMENTS`). Do NOT render.

The stop is deliberate. Rendering before review wastes twenty minutes.

## Hard rules

- Never invoke `ffmpeg` directly. Always go through `python -m autocut`.
- All times in the EDL are in the SOURCE timebase. Never write output-relative times.
- Do not edit files under `work/` other than `edl.json`.
- Never touch `inbox/`.

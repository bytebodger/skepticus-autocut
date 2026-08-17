# Skepticus Autocut - Declared Retakes Spec

Detects flubbed lines that were re-recorded in place, using a spoken cue rather
than semantic inference.

Runs inside `autoauthor`, emits drops into the EDL, changes nothing downstream.

---

## 1. The design

You say a specific word before redoing a line. The pipeline keys on that word.

That single decision removes the hardest problem in this feature. General retake
detection has to distinguish an accidental repeat from a deliberate one, and
that distinction is genuinely subjective. Rhetorical repetition, callbacks,
parallel construction, and steelman-then-rebut all look identical to a
similarity detector.

With a cue, none of that matters. Repetition without the cue is content.
Repetition with the cue is a retake. There is nothing to infer.

Consequences worth naming:

- **No LLM in the critical path.** Detection is a string match. It cannot
  hallucinate a retake.
- **Near-zero false positives.** The only way to lose real content is to say the
  cue by accident.
- **Recall is under your control.** Forget the cue and the retake survives, same
  as today. That's an acceptable failure and it's the safe direction.
- **Verifiable.** You can test it in thirty seconds by saying the cue and
  checking the output.

The cost is a small habit change while recording. Given what it buys, that's a
good trade.

---

## 2. Choosing the cue

The cue must satisfy three things.

**It never occurs naturally in your content.** This rules out most obvious
candidates. "Cut," "again," "sorry," "scratch that," and "take two" are all
plausible in a video about textual criticism or a discussion of editing. Your
subject matter is unusually good at producing false hits on ordinary words.

**Whisper transcribes it reliably.** A cue Whisper renders inconsistently is
worse than no cue, because failures are silent. Verify before committing.

**It's easy to say mid-flow.** You'll use it dozens of times per recording.

Candidates that satisfy all three: a distinctive uncommon noun like "mulligan,"
a doubled word like "redo redo" (natural doubling is rare), or a coined
nonsense token. Avoid anything with common homophones.

Configure it, don't hardcode it:

```yaml
retakes:
  enabled: true
  cue: "mulligan"
  scope: previous_utterance    # previous_utterance | previous_pause
  min_pause_after_cue: 0.3
```

### Matching

Match on normalised text, not exact. Lowercase, strip punctuation, collapse
whitespace. Whisper will sometimes capitalise the cue, sometimes attach a comma,
occasionally split a longer cue across word tokens.

Allow a small edit distance for single-word cues, but keep it tight. One
character of tolerance catches transcription wobble without inviting false hits.

Log every match with its raw transcribed form. Over a few episodes that tells
you whether Whisper is hearing your cue consistently.

---

## 3. Scope

The cue marks that a retake happened. It doesn't say how far back to cut.

**Default: back to the previous utterance boundary.** Segment the transcript
into utterances using sentence-final punctuation plus pause boundaries above
roughly 400ms. The drop spans from the start of the utterance containing the
flub, through the cue itself, to the start of the next utterance.

That covers the common case: you flub a sentence, stop, say the cue, restart the
sentence.

**Alternative: back to the previous pause.** Configurable. The drop extends back
to the last silence of at least a set duration. Useful if you naturally pause
before the material you intend to redo, since it gives you direct control over
scope by where you pause.

**Multi-utterance flubs are handled at review,** not by inference. If you
restarted a whole paragraph, the default scope catches only the last sentence of
it. The review gate lets you extend the drop backward one utterance at a time
with a keystroke. Fast, and it keeps the automatic behaviour predictable.

**The cue is always dropped** along with the flubbed material.

---

## 3.5 Every cue always cuts

**A detected cue always produces a cut. There are no exceptions and no
confidence threshold that suppresses one.**

If the cue was spoken, a retake happened. That's the whole point of declaring it
rather than inferring it. A cue left uncut is a failure, not caution.

The uncertainty in this feature is not *whether* to cut. It's *where the
boundaries go*. Those are separate questions and only the second one is
uncertain.

So confidence governs boundary strategy, never the decision to cut:

- **High confidence** means the redo cleanly repeats the flub's opening, so the
  boundary can be inferred tightly by prefix alignment.
- **Low confidence** means it can't. Fall back to a conservative boundary - the
  nearest confirmed silence before the flub, or the previous utterance start -
  and flag the row prominently at review.

A conservative boundary may remove slightly more than necessary. That's the
correct failure direction. Leaving the flub in the video is not.

Never emit a "detected but not cut" state.

---

## 4. Boundary resolution

Reuse the mechanism that fixed the earlier dialogue-loss bug.

Where acoustic silence from `silence.json` sits at the drop's edges, snap the
boundaries to it. A cut bounded by confirmed silence cannot clip a word.

You'll almost always pause before and after saying the cue, so silence should be
available at both edges most of the time. Where it isn't, fall back to word
boundaries with the standard padding and flag it in the review row.

---

## 5. Merging

Retake detection runs on the full word list, before silence and filler removal
have been applied. Filler often sits inside a flub, and the drop should swallow
it rather than leaving fragments behind.

Drops from all three passes get merged and de-overlapped. Each merged drop
retains its originating reason so the review UI can show why.

---

## 6. Review gate

Lighter than semantic detection would have needed, because there's nothing to
approve. Every cue is already cut. What you're adjusting is scope.

Each retake row shows:

- The dropped text, with the cue highlighted
- The text that follows, so you can see the replacement
- A keystroke to extend the drop backward by one utterance
- A keystroke to pull the drop forward, if it took too much
- A keystroke to veto, for the rare case where the cue was spoken by accident

**Audio playback is still worth having.** Not to decide whether it's a retake,
but to confirm the cut sounds clean when there was no silence at a boundary.

Sort low-confidence rows first. Those are the ones with inferred boundaries that
may need nudging. High-confidence rows can be scanned quickly.

Vetoes are sticky, as with existing overrides.

---

## 7. Verification before you rely on it

Before recording a real episode with this, do a thirty-second test.

Record yourself saying a line, flubbing it, saying the cue, and redoing it.
Three or four times, with the cue in different positions and at different
speaking speeds. Then run transcribe and grep `words.json` for the cue.

You're checking that Whisper renders it the same way every time. If it comes
back as three different spellings, pick a different cue.

Do this before changing your recording habit, not after.

---

## 8. Build order

1. **Utterance segmentation.** Deterministic, testable, no model involved.
   Verify boundaries look sane against a real transcript.
2. **Cue detection and scope resolution.** The core feature.
3. **Boundary resolution** with silence intersection.
4. **Review gate** with scope extension.

Small enough to build in one pass. There's no phased risk here because there's
no inference.

---

## 9. What this unlocks later

Once spoken commands work, the mechanism generalises. Other cues could mark
chapter breaks, flag a moment for a specific visual, or note a spot to revisit.
All the same machinery: a configured phrase, a scope rule, and an action.

Worth keeping the implementation general enough that adding a second command
isn't a rewrite. Don't build any of them yet.

---

## 10. Optional: semantic detection later

Semantic retake detection remains possible as a supplement for the times you
forget the cue. It's explicitly deferred, and it may never be worth building.

If you do add it, it should never act on its own. It would surface low-confidence
suggestions at review, clearly separated from declared retakes, and default to
no action. The asymmetry that governs the whole feature still holds: a missed
retake costs one manual cut, a false positive costs a broken video.

---

## 11. Known hard parts

**Forgetting the cue.** The main limitation, and it degrades gracefully. You cut
that one manually, exactly as you do today.

**Saying the cue by accident.** The only path to a false positive. Choosing a
word that never appears in your subject matter reduces this to near zero, and
review catches it.

**Whisper mishearing the cue.** Silent failure: the retake survives and you
don't know why. Section 7's verification is what prevents this, and the match
logging is what catches it drifting later.

**Scope on long flubs.** Restarting a whole paragraph needs the drop extended at
review. Predictable, but it's manual work on the cases that need it most.

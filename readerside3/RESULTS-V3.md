# Reader-side evaluation v3 — the digest channel

*2026-10-12. v1 and v2 reached the same structural conclusion from opposite directions: at 148×80, raw-plane region reading never beats the per-question constant-answer composite (0.475), because readers confabulate tallies instead of performing them. The Typesafe lane's design claim says region truth should travel via a computed **digest**. v3 measures that claim — in the same batch as a fresh raw-plane control, because v2 taught us the batch is a variable.*

## Setup

- **Frames, questions, truth, scorer:** identical to v1/v2. Same 40 docks frames, same 5 questions, `questions.jsonl` copied **byte-identical** (sha256 `86cdbda1…` in all three directories) — truth was not regenerated. Parser, normalizers, callers and scorer are source-identical to v2's (gate-checked with `inspect.getsource`). Same readers (MiniMax-Text-01; GLM glm-4.5-flash, thinking disabled), temperature 0.7 (caller defaults), one batched call per (reader, condition, frame). **Coverage 160/160 successful records, 0 failed attempts** — the first batch in the series with no 429 backfill.
- **Condition DIGEST:** the reader receives **no grid**. It receives a computed digest of the frame, derived **only** from the same source NV12 Y plane the frozen truth was computed from, using `gen_dataset.py`'s definitions verbatim — never from the truth answers. The digest is a perception channel, not a leak: it contains sums, means, extrema and tallies, and *no* argmax, comparison result, or answer-shaped quantity.
- **Condition LEARNED_CAL (control):** the v2 adopted mapping (learned micro table, calibrated render), re-run **fresh in this batch** so digest-vs-plane is a same-batch comparison. Its prompts reproduce v2's recorded prompts **byte-for-byte on all 40 frames** (gate-checked).
- **Blind** was not re-run; v1's blind cells are the control reference and lifts inherit v1→v3 drift (±3–7 points, per v2's measurement).

### The digest format (exact)

One text block per frame (~1.6k chars; verbatim samples in `digest_samples.md`, all 40 in `digests.jsonl`):

- **Brightness:** quadrant luma sums (TL/TR/BL/BR) and quadrant mean luma; left/right half sums; frame total, mean, min, max luma.
- **Brightness centroid:** x̄ and ȳ in cell units (x̄ is the Q3 quantity; centre line x = 74.00 printed alongside).
- **Edges** (Sobel |gx|+|gy| > 200 on Y, the frozen definition): edge-pixel **counts** and **mass** by half; counts and mass by horizontal band.
- **Strip tallies:** the v2 guide's intermediate step, precomputed — brightness sum and edge count per half for each of the 8 strips of 10 cell rows.

The reader guide (`GUIDE-DIGEST.md`) turns the v2 procedures into digest lookups: Q1/Q2 = largest/smallest quadrant sum; Q3 = compare centroid x̄ with 74.00 (explicitly *not* inferred from half sums); Q4/Q5 = largest edge **count** (mass printed as a cross-check; where count and mass disagree — 3 frames for Q4, 2 for Q5 — the guide states counts are what the questions ask).

### Audit: does the digest contain the answers' information?

Three gates ran before any reader was called:

- **Gate A — fidelity:** every region number in every digest equals the frozen computation recorded in `questions.jsonl`, on all 40 frames. The digest *is* the truth's arithmetic, republished as a channel — that is the design under test, stated plainly.
- **Gate B — consistency:** the strip tallies sum exactly to the half sums, quadrant sums and half edge counts they decompose, all 40 frames.
- **Gate C — oracle:** a scripted reader that parses the digest **text** (the exact string the models see) and applies the frozen decision rules scores **200/200**. The digest provably suffices for every question. A second oracle restricted to the strip tallies alone also recovers Q1/Q2/Q4 at 40/40 — and its Q3 heuristic (side with the larger half total) scores only **35/40**: on 5 frames the brighter half is not the half the centroid falls in. The centroid line is load-bearing; half totals would not have sufficed for Q3. Had Gate C failed, the experiment would have been about the digest; it did not fail.

## Headline (strict frozen scoring; 200 answers per cell)

| reader | digest | learned_cal (same batch) | digest − cal | blind (v1) | const ceiling |
|---|---|---|---|---|---|
| MiniMax | 0.850 | 0.370 | **+0.480** | 0.310 | 0.475 |
| GLM | **1.000** | 0.400 | **+0.600** | 0.280 | 0.475 |

(The ceiling's exact value is 95/200 = 0.475; v1/v2 printed 0.476 from rounded per-question components. Same ceiling.)

| reader | digest parsed-only | cal parsed-only | digest clear-margin | cal clear-margin |
|---|---|---|---|---|
| MiniMax | **1.000** | 0.379 | 0.841 | 0.397 |
| GLM | **1.000** | 0.430 | **1.000** | 0.341 |

Per-question (digest | cal), with the constant baselines:

| q | type | MiniMax | GLM | constant |
|---|---|---|---|---|
| Q1 | brightest quadrant | 0.850 \| 0.275 | **1.000** \| 0.350 | BR 0.425 |
| Q2 | darkest quadrant | 0.850 \| 0.250 | **1.000** \| 0.325 | TL 0.375 |
| Q3 | centroid side | 0.850 \| 0.375 | **1.000** \| 0.525 | right 0.525 |
| Q4 | edge-mass side | 0.850 \| 0.500 | **1.000** \| 0.425 | right 0.55 |
| Q5 | edge band | 0.850 \| 0.450 | **1.000** \| 0.375 | top 0.50 |

Paired head-to-head, digest vs same-batch cal (answer slots of 200): MiniMax — digest-only wins **103**, cal-only 7, both 67, neither 23. GLM — digest-only **120**, cal-only **0**, both 80, neither **0**: GLM's digest record strictly dominates its plane record, slot by slot, with no exceptions.

**The control behaved normally.** Same-batch cal redraws v2's cal cells within the known noise floor (MiniMax 0.370 vs v2's 0.345; GLM 0.400 vs v2's 0.380 — both +2–2.5 points, inside ±5–7). The batch was not anomalous; the plane failed in this batch the way it failed in the last two. The digest margins (+48, +60 points) are roughly **ten times the protocol's noise floor** — every conclusion below is stated against that floor, and none of them is close to it.

## The MiniMax autopsy: 0.850 is a format number

MiniMax's digest row is 0.850 on *every* question, with exactly 6 unparsed answers per question (30 of 200) — the fingerprint of whole-record format failures, and parsed-only accuracy is **1.000**. The six records (frames 46, 53, 68, 106, 129, 197) were opened and read:

- Frames 46/53/68/106: answers given as `**Answer Q1:** TR` on one line — the frozen parser's anchored pattern requires the answer line to *start* with Qn, so all five slots go unparsed. The intended answers score **5/5 on all four frames**.
- Frame 129: answers present three times over, all correct — but the first parser-matching line is the model's own question heading (`**Q1: Which quadrant…**`), whose segment normalizes to nothing, and first-match-wins does the rest. Intended: **5/5**.
- Frame 197: answers wrapped in backticks (`` `Q1: BR` ``), which the anchored pattern cannot see. Intended: **5/5** — including frame 197's Q4, one of the three count-vs-mass disagreement frames, answered by counts as the guide instructs.

**In intent, MiniMax's digest reading was 200/200 — identical to GLM's scored 200/200. Across both readers, not one comparison error occurred in the digest condition.** Every residual digest error is an output-format event of the same species v1 documented for GLM's blind template echo. The strict headline stands as the protocol's number — format discipline is part of the protocol — but the substantive finding is the stronger one: *given exact numbers, both readers compare them flawlessly at temperature 0.7; the failure surface moves from perception to formatting, which is a cheaper problem by orders of magnitude.*

## Named cases

- **Frame 83** (v1's MiniMax fixed-plane confabulation case; v2's "no reader in any condition answered Q1 correctly"): on digest, **both readers go 5/5** — Q1 = TR by the printed quadrant sums (3,438,422, the wide margin of the earlier disputes), Q4 = left by the printed edge counts (left 1,417 vs right 486, a 3:1 margin the plane readings in both earlier batches somehow inverted or missed; on same-batch cal both readers again answer Q4 "right" ✗). The frame that convicted the plane in two consecutive batches is a non-event through the digest.
- **Count-vs-mass disagreements** (where the guide's "counts decide" instruction is load-bearing): Q4 frames [0, 23, 197] — GLM digest 3/3, MiniMax 2/3 scored and 3/3 in intent (frame 197 again). Q5 frames [15, 83] — both readers 2/2 on digest. The instruction held wherever it could be parsed.

## Cost — measured, and the 100× expectation examined

| measure | digest | grid (learned_cal) | ratio |
|---|---|---|---|
| frame representation block | 1,592 chars (mean) | 11,919 chars | **7.5× smaller** |
| whole prompt | 6,924 chars (mean) | 16,549 chars | **2.4× smaller** |
| reader output | 644 (GLM) / 1,194 (MiniMax) chars | 2,740 / 3,351 chars | 2.8–4.3× shorter |
| mean latency per call | 13.2 s (GLM) / 16.9 s (MiniMax) | 49.4 s / 33.1 s | 2.0–3.7× faster |

The design brief's "~100× cheaper" does **not** survive measurement, and the reason is structural, not a tuning failure: the protocol's fixed overhead — guide, questions, answer format — is ~5.3k chars on *both* channels, so even a zero-length frame representation would cap the whole-prompt saving at ~3×. At the level where the comparison is honest — what the frame itself costs per call, the marginal channel cost once the guide is a cached prefix — the saving is 7.5× on input, compounded by 2.8–4.3× shorter outputs (no tally theatre to generate) and 2–3.7× lower latency. The digest's real economy is not the character count anyway: it replaces a ~50-second confabulation with a ~15-second lookup.

## Verdict

**The digest channel clears the ceiling the plane never could — decisively, for both readers, in a batch whose control re-failed on schedule.** GLM scores 200/200 strict; MiniMax scores 170/200 strict with the 30 missing answers all present, correct, and lost to answer-line formatting. Against the constant-answer composite that capped every plane cell in v1 and v2 (0.475), the digest posts +37.5 (MiniMax, strict) and +52.5 (GLM) points — margins an order of magnitude beyond the ±5–7 noise floor v2 measured. The Typesafe lane's design claim is confirmed from the reader side, with the strength the earlier experiments could only suggest: **region truth should travel as computed sums; the raw plane should be reserved for what it measurably does — position and structure impressions.**

Three consequences, stated at their true strength:

1. **The stage-5 reader question splits in two.** For region questions, the learned-vs-fixed plane comparison is moot — both planes are dominated by a channel that costs less and reads perfectly. The plane comparison retains meaning only for the impression questions (position, structure) that a digest of region sums cannot answer. Any future gate should name which kind of question it is gating.
2. **The digest is question-shaped, and that is the design, not a cheat — with a boundary.** The digest fields were chosen because a question battery exists (the Typesafe verdict-battery pattern: compute the digest *for* the battery). It cannot answer a question it does not carry; a new question type needs a new digest field, at integer-math prices. What v3 rules out is the alternative hope that a general-purpose grid read would cover unlisted region questions — three batches say it covers none of them reliably.
3. **The residual digest problem is format discipline.** A production digest reader needs its answer format enforced (constrained decoding, or a parser matched to the channel's own templates) — the frozen parser's 0.850 for a flawless reading is the protocol being honest about a real deployment hazard, not noise.

## Caveats

- **The digest is computed by the same arithmetic that defines the truth** (Gate A makes this exact). v3 measures the *channel* claim — can readers use region truth when it travels as sums? — not the difficulty of computing the sums. In the Syzygy design that computation is the kernel's job (integer, O(pixels), already specified); no learned component is involved, and no claim is made here about learned digests.
- Temperature 0.7 throughout, as in v1/v2. Temperature-0 digest runs would presumably remove even the format variance's randomness but not its possibility; the noise floor (±5–7 points on 200-answer cells) applies to the *control* comparisons in this report, not to the digest margins.
- One clip, one harbour, daylight; 40 frames. The learned table (plane condition only) was trained on this same clip. The digest condition involves no table at all.
- The digest carries *only* region quantities. Nothing in v3 speaks to object position, identity, or structure questions — v1's verdict stands that those are the plane's (impressionistic) territory, untested here by design.
- Blind not re-run; lifts against v1 blind inherit cross-batch drift. No v3 conclusion depends on a lift.

## Artifacts (all in this directory)

`gen_digest.py` (digest computation + Gates A/B/C + strip oracles) · `digests.jsonl` (40 digests, text + structured fields) · `digest_meta.json` (gate results, sufficiency audit, size stats) · `digest_samples.md` (verbatim digests for frames 0, 23, 83) · `GUIDE-DIGEST.md` (the digest reader guide) · `GUIDE.md`, `questions.jsonl`, `frames_sampled.json` (byte-identical copies from v1) · `renders/learned_cal/` (byte-identical copies from v2) · `gate_check.py` + `gate_check.out` (frozen-code identity, v2 prompt reproduction 40/40) · `run_readers3.py` (runner) · `reader_outputs.jsonl` (160 records, 0 failed attempts) · `summarize3.py` + `summarize3.out` · `scores_v3.json` (all stats incl. paired head-to-head, count-vs-mass frames, frame 83) · `failures_v3.json` (276 wrong answers with excerpts — 246 plane, 30 digest, all 30 digest entries being the unparsed format cases) · `run3.log`.

*Run cost note: 160 successful calls, ZERO failed attempts, ≈27 minutes of wall time at 4 workers — against v2's ≈2.3 hours for 240 grid calls. The digest batch was cheaper to run than the experiment that motivated it.*

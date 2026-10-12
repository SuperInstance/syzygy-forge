# Reader-side evaluation v2 — the render mapping on trial

*2026-10-12. v1's verdict named a suspect: the forge renderer squeezes the 16 learned codewords onto the 10-step ramp by centroid-luma **rank**, preserving brightness order but destroying intervals — and the interval questions (Q1 quadrant mass, Q3 centroid) were exactly where GLM's learned-plane reading collapsed (Q3 0.600 fixed → 0.325 learned-rank), with frame 83 as the named case. v2 varies that one variable. Its real finding is about the courtroom, not the suspect.*

## The one variable

Same 40 frames, same 5 questions, same pixel-computed ground truth (`questions.jsonl`, copied byte-identical from v1 — truth was **not** regenerated), same frozen guide, same parser/scorer/callers (verified byte-identical code), same readers (MiniMax-Text-01; GLM glm-4.5-flash, thinking disabled), temperature 0.7 as in v1. Three frame conditions:

- **fixed** — the kernel's fixed-table tone plane (v1's renders, byte-identical copies).
- **learned-rank** — the learned micro table (id `0x7ccb88b5`) under v1's mapping: codewords ordered by centroid mean_luma, `idx = min(rank · 10 // 16, 9)`.
- **learned-cal** — the *same per-cell assignments*, but each codeword's character is chosen by the kernel's own luma scale applied to the centroid's actual mean luma: **`idx_cw = min((L_cw · 9) // 255, 9)`**, where `L_cw` = `table rows[cw][0]` (feature 0, mean_luma — the quantity the rank ordering sorted by; the micro table's space is features 0–7, so the kernel's cell_luma/feature 10 is not available to map). This is literally the base term of `syz_fused_tone` (`idx = ((uint32_t)luma * 9u) / 255u + sel`), with luma := the centroid's mean luma and no `sel` term. Full per-codeword mapping in `mapping.json`.

Consequences of honesty: rank uses all 10 ramp chars; calibrated uses **8** (centroid mean_lumas span 18–207, so the brightest codeword lands at idx 7, `#` — no `%`, no `@`; a centroid is an average and never reaches per-cell extremes). Codewords within one ramp step (~28.3 luma units) share a character. Calibrated vs rank differ on a mean of 10,534 of 11,840 cells per frame (89%) — the mappings are genuinely different renderings of the same assignment.

**Sanity gate (passed before any reader ran):** regenerated learned-rank planes are byte-identical to v1's render files on all 40 frames; the forge compare panels for frames 10/120 reproduce; v1's fixed files match the shipped syzygy-video tone files on all 40 frames; and the grids embedded in all 160 successful v1 reader prompts match the render files byte for byte. Recomputing v1's scores from v1's raw outputs with this summariser reproduces v1's frozen `scores.json` exactly.

**Protocol note:** all three frame conditions were re-run fresh in one batch (240/240 successful records; 6 failed GLM 429 attempts retained and backfilled by a resume run, as in v1). Blind was not re-run; v1's blind numbers are the control reference, and lifts below are computed against them.

## Headline (strict frozen scoring; 200 answers per cell)

| reader | fixed | rank | cal | blind (v1) | lift F / R / C | cal − rank | cal − fixed |
|---|---|---|---|---|---|---|---|
| MiniMax | 0.385 | 0.390 | 0.345 | 0.310 | +0.075 / +0.080 / +0.035 | −0.045 | −0.040 |
| GLM | 0.320 | 0.350 | **0.380** | 0.280 | +0.040 / +0.070 / +0.100 | +0.030 | +0.060 |

## The elephant: the batch is a variable too

Before any mapping conclusion, the controls. The two *unchanged* conditions were redrawn in this batch, and they moved:

| cell | v1 frozen | v2 fresh | Δ |
|---|---|---|---|
| MiniMax fixed | 0.315 | 0.385 | +0.070 |
| MiniMax rank | 0.430 | 0.390 | −0.040 |
| GLM fixed | 0.350 | 0.320 | −0.030 |
| GLM rank | 0.325 | 0.350 | +0.025 |
| GLM Q3 fixed | 0.600 | 0.450 | −0.150 |
| GLM Q3 rank | 0.325 | 0.475 | +0.150 |
| GLM Q1 fixed | 0.150 | 0.300 | +0.150 |

Per-cell sampling σ at n=200 is ≈3.5 points; observed identical-protocol batch movement is 2.5–7 points overall and up to 15 on a per-question cell. **v1's headline contrast — MiniMax learned − fixed = +0.115 — redraws as +0.005** (paired head-to-head on the same 200 slots: fixed-only wins 43, rank-only wins 44). This does not make v1's numbers wrong; it makes them *one draw*. Every v2 conclusion below is stated against this floor.

## Per-question (fixed | rank | cal), v2 fresh

| q | type | MiniMax | GLM | constant |
|---|---|---|---|---|
| Q1 | brightest quadrant | 0.275 \| 0.325 \| 0.225 | 0.300 \| **0.175** \| **0.350** | BR 0.42 |
| Q2 | darkest quadrant | 0.375 \| 0.325 \| 0.225 | 0.225 \| 0.325 \| 0.350 | TL 0.38 |
| Q3 | centroid side | 0.450 \| 0.450 \| 0.500 | 0.450 \| 0.475 \| 0.400 | right 0.53 |
| Q4 | edge-mass side | 0.525 \| 0.475 \| 0.525 | 0.325 \| 0.500 \| 0.550 | right 0.55 |
| Q5 | edge band | 0.300 \| 0.375 \| 0.250 | 0.300 \| 0.275 \| 0.250 | top 0.50 |

## The named question: Q3, and frame 83

**Q3 — the collapse did not happen.** In v2's fresh samples GLM reads the centroid from the rank plane at 0.475, statistically indistinguishable from fixed (0.450) — v1's 0.325-vs-0.600 chasm is gone. And the calibrated mapping, built to repair exactly this, is GLM's *worst* Q3 cell (0.400), losing the paired head-to-head to rank (cal-only frame wins 6, rank-only 9, both 10, neither 15). If interval destruction by the rank map were the mechanism of the collapse, calibrated should have recovered Q3 specifically. It recovered nothing on Q3. For MiniMax, Q3 is flat across fixed/rank (0.450/0.450) with cal nominally ahead (0.500; head-to-head vs rank: cal-only 9, rank-only 7) — again inside noise.

**Frame 83 — the named case acquits everyone by convicting no one.** Truth: Q1 = TR by a wide margin (TR 3,438,422 vs runner-up BL 2,941,908); centroid right (x̄ = 76.8 cells). In v1, this frame produced GLM-fixed TR ✓, GLM-rank BL ✗ (invented strip tallies: left 1,621–2,023 vs right 688–1,038 in every strip), and a MiniMax rank 5/5 sweep. In v2, **no reader in any condition answered Q1 correctly**: MiniMax BR / TL / BR; GLM TL / TL / TL. GLM's v2 rank output invents quadrant totals with TL ≈ TR (≈86,000 each) and picks TL; its v2 *calibrated* output is the degenerate failure — the guide's template repeated verbatim ("Left half (cols 1-74): 10 rows * 74 cols = 740 cells", strip after strip) with no tally ever performed. A single frame at temperature 0.7 is a coin flip with a story attached; v1 attached the story to the mapping. The frame cannot carry that weight, and in this batch it points at nothing.

## Parsed-only and clear-margin (the adjustments, as in v1)

| reader | scoring | fixed | rank | cal | blind (v1) |
|---|---|---|---|---|---|
| MiniMax | parsed-only | 0.416 | 0.413 | 0.363 | 0.339 |
| MiniMax | clear-margin | 0.397 | 0.437 | 0.357 | — |
| GLM | parsed-only | 0.362 | 0.380 | **0.415** | 0.400 |
| GLM | clear-margin | 0.373 | 0.357 | **0.389** | — |

Unparsed slots (of 200): MiniMax 15 / 11 / 10 (F/R/C); GLM 23 / 16 / 17. Format does not explain the pattern: MiniMax's calibrated drop survives parsed-only adjustment (it is accuracy, not format), and GLM's calibrated lead *grows* under parsed-only.

## Paired head-to-head (discordant slots of 200; the discriminator this n can support)

| pair | MiniMax (A-only : B-only) | GLM (A-only : B-only) |
|---|---|---|
| rank vs cal | 45 : 36 (rank) | 30 : 36 (cal) |
| fixed vs rank | 43 : 44 | 36 : 42 (rank) |
| fixed vs cal | 37 : 29 (fixed) | 32 : 44 (cal) |
| GLM Q1, rank vs cal | — | 4 : 11 (cal) |

No split reaches |z| ≈ 2 (the largest, GLM fixed-vs-cal overall, is z ≈ 1.4; GLM Q1 rank-vs-cal, 11 of 15 discordant, z ≈ 1.8, is the single most lopsided and the one place the ordering matches the interval hypothesis exactly). The two mappings produce genuinely different readings, not noise around one: same-answer rate between rank and cal versions of the same frame is 0.41 (MiniMax) and 0.53 (GLM).

## Verdict — mapping, table, or reader?

**The reader — meaning the reader as a sampling process. Not the mapping, and not the table.**

- *Not the mapping:* the crime scene doesn't recur. The Q3 collapse the rank map was accused of causing is absent when the identical rank plane is redrawn (0.475), and the interval-preserving repair posts the lowest Q3 of the three (0.400) and loses its head-to-head. A mechanism that explains one batch and is contradicted by the next was never measured — it was sampled.
- *Not the table:* both learned conditions share one assignment; if that assignment destroyed centroid information, rank and cal would suffer together against fixed in both batches. Instead GLM's learned − fixed contrast flipped sign between batches (v1: −0.025; v2: +0.030 rank, +0.060 cal), and in v2 GLM reads the learned assignment at or above its fixed-plane level under *both* mappings.
- *The reader:* what is stable about GLM across both batches is not a deficit on any question — it is variance itself. Its per-question cells swing 15 points between identical batches; its tallies are confabulated in all conditions (v2 frame 83 offers the full menagerie: invented totals on rank, template recitation on cal); its same-frame agreement between two renderings is barely above half. v1 measured one draw of that process and, finding it extreme on exactly the questions a mapping story could explain, attributed it to the mapping. v2 is the control v1 lacked.

Two residues, stated at their true (sub-conviction) strength:

1. **For GLM, calibrated is the best cell under all three scorings in this batch** (strict 0.380, parsed-only 0.415 — its only cell above its own blind parsed-only — clear-margin 0.389), and the gain concentrates where the interval hypothesis points: Q1 (+17.5 over rank; paired 11-vs-4) and Q4 (+5 over rank, +22.5 over fixed). The rank map plausibly *does* tax GLM's mass-ratio questions; this batch can suggest it, not convict.
2. **For MiniMax the sign reverses**: calibrated is its worst cell under all three scorings, rank its best or tied-best. A labelled interpretation: the calibrated ramp's honest top-end compression (no chars above idx 7) removes precisely the exaggerated contrast the rank map fabricates, and MiniMax's impressionistic reading — the one that gained most from the learned plane in v1 — was partly reading that fabrication.

**Consequence for the stage-5 gate:** the reader bar (learned ≥ fixed + 5 points) is *smaller than this protocol's noise floor* (±5–7 points per cell between identical batches). Neither v1's "met by MiniMax" nor v2's "cal leads for GLM" can adjudicate it. A decisive run needs temperature 0 and/or roughly 4× the frames — and until one exists, the mapping choice should be made on principle, not on measured gain: rank asserts brightness intervals that do not exist in the data; calibrated renders the intervals that do. Ship the truthful rendering; claim no reader benefit for it yet. v1's other structural finding is untouched by all of this: every cell in both batches sits below the per-question constant-answer composite (0.476; best v2 cell 0.380) — at 148×80, region truth still belongs to the digest.

## Caveats

- Blind was not re-run; if blind drifts like the frame conditions do (±3–7 points), every lift quoted inherits that drift. Lifts are indicative, not exact.
- Temperature 0.7, both batches, both readers — by design (v1's protocol), and the direct cause of the noise floor this document is about.
- One clip, one harbour, daylight; the learned table was trained on this same clip (a readability comparison of frozen planes, not a generalisation test — as in v1).
- The calibrated quantity is the centroid's mean_luma (feature 0). The fixed kernel's base ramp runs on cell_luma (feature 10), which is not in the micro table's feature space; mean_luma is the centroid's own brightness summary and the same quantity v1's rank ordering used. A future table carrying feature 10 could map on the kernel's exact input.
- Q4/Q5 truth definition sensitivity from v1 (count- vs mass-argmax disagreements) is unchanged — same frozen truth file.

## Artifacts (all in this directory)

`gen_calibrated.py` (mapping, sanity gate, renders) · `mapping.json` (per-codeword rank vs calibrated chars) · `frames_meta_v2.json` (gate results, per-frame changed-cell counts) · `renders/fixed/` + `renders/learned/` (byte-identical copies of v1's, gate-verified) + `renders/learned_cal/` (the new condition) · `questions.jsonl`, `frames_sampled.json`, `GUIDE.md` (byte-identical copies from v1) · `run_readers2.py` (v1 runner; parser/scorer/callers byte-identical, conditions extended) · `summarize2.py` (v1 summariser extended; recomputes v1 exactly) · `reader_outputs.jsonl` (240 successful records + 6 retained failed attempts) · `scores_v2.json` (all stats incl. Q3 dive and frame 83) · `failures_v2.json` (766 wrong answers with excerpts) · `run2.log`, `backfill2.log`, `summarize2.out`.

*Run cost note: 240 successful calls at ~16.5k-char prompts (MiniMax mean latency 39.0 s, GLM 62.9 s, max 219.6 s) plus backfill — ≈2.3 h wall at 4 workers, roughly double v1's 75 minutes for the same call count; GLM's account-wide 429 pressure was worse this session. The noise floor documented above was purchased at that price; it is the most reusable number in either experiment.*

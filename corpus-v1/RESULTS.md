# Corpus v1 — does the learned codebook generalise beyond one clip?

*2026-10-12. Forge v0 trained its T1 tables on a single 49-second
harbour walk, and every result since — quantisation errors, reader
scores — carries that caveat. This run extends the corpus to all five
stage-0 sources and asks the v0 open question (forge README, item 6)
with data: one general table, or per-domain tables?*

## Setup

- **Corpus.** Stage 0's five real sources, reused as its extracted
  NV12 streams (`syzygy-program/stage0/work/*.nv12`, provenance in
  stage 0's `frames_manifest.jsonl`) — no video was re-decoded:
  docks (Casey's harbour walk, 1,481 frames), the-arena (render, 925),
  chiaroscuro (Casey's screen recording, 1,039), 204-lighthouses
  (render, 917), seven-millisecond-judge-v2 (render, 1,032). All at
  the stage-0 raster: 160×96 px → **80×24 cells**, 5,394 frames,
  10,356,480 cells.
- **Extraction.** The v0 extractor (`forge/src/extract_features.c`,
  compiled unchanged) ran over each stream: same 16 integer features
  per cell, same per-cell reads replicating `syz_fused`, fixed
  decision from the kernel's own `syz_ste_select4`.
  **Verification:** FNV-1a over each frame's tone plane (ASCII ramp
  bytes) matches stage 0's independent native implementation on
  **5,394 / 5,394 frames, 0 mismatches** — the forge features for the
  new clips are provably the kernel's own view of them.
- **Split.** Frozen per clip with v0's rule: frame index % 5 == 4 →
  test (`split_manifest_v1.json`). All errors below are held-out.
- **Methods.** *fixed* (the kernel's hand-written table, assignment =
  extractor's sel byte); *docks* (the shipped v0 tables, micro
  `0x7ccb88b5` / full `0x6f81750a`, trained at 148×80 on the v0 docks
  corpus); *pooled* (new tables trained on all five clips' train
  cells: micro `0xdf32b48c`, full `0xfe573b61`); *specialist* (new
  tables trained per clip, evaluated only on their home clip — the
  per-domain ceiling). Training is v0's integer Lloyd with the bias
  term, copied: seed 20261011, 200,000-cell subsample, integer
  rounding every step.
- **Metric.** V0's partition quantisation error, unchanged: post-hoc
  centroids = mean 16-dim feature vector of each method's *own*
  training cells per codeword (identical treatment for every method —
  generous to fixed); error = mean Euclidean distance of test cells
  to their assigned codeword's centroid. **Anchor gate:** the same
  machinery on the v0 corpus reproduces v0's published numbers —
  fixed 151.18 (got 151.1819), micro 44.63 (44.6331), full 29.87
  (29.8692). **Identity check:** the pooled tables' biased inference
  equals direct nearest-centroid on 11,520 cells, 0 mismatches.

## Headline: held-out quantisation error per clip ↓

| clip | fixed | docks micro | pooled micro | spec micro | docks full | pooled full | spec full |
|---|---|---|---|---|---|---|---|
| docks | 145.01 | 48.70 | 52.02 | **48.39** | 33.92 | 38.12 | **33.57** |
| arena | 218.68 | 44.63 | 40.17 | **35.68** | 35.30 | 29.78 | **23.55** |
| chiaroscuro | 201.63 | 62.43 | 58.58 | **56.42** | 46.19 | 42.90 | **39.90** |
| lighthouses | 231.75 | 44.68 | 39.27 | **34.40** | 34.16 | 28.65 | **23.69** |
| judge | 168.21 | 52.27 | 47.63 | **39.98** | 41.49 | 30.99 | **25.85** |

## (a) The shipped docks tables, taken travelling

The v0 tables were trained on one harbour, at a different raster.
On foreign clips they hold up:

- Against fixed, they win everywhere by **3.2–5.2× (micro)** and
  **4.1–6.8× (full)**. The fixed table is at its *worst* off the
  docks (218.68 arena, 231.75 lighthouses vs 145.01 at home) — its
  partition was never earning anything, and the renders expose it
  most.
- Their absolute errors off-domain (micro 44.63–62.43, full
  34.16–46.19) sit in the same band as at home (48.70 / 33.92 on
  docks at this raster; 44.63 / 29.87 at their native raster). No
  clip is pathological for them; chiaroscuro — a screen recording of
  ASCII art, about as far from a harbour as this corpus gets — is
  their worst, and even there they beat fixed 3.2× / 4.4×.
- On their home clip at the new raster they are within **0.6%
  (micro) / 1.0% (full)** of a specialist trained natively at this
  raster (48.70 vs 48.39; 33.92 vs 33.57). The raster change itself
  costs the full table ~13% against its native-raster number
  (29.87 → 33.92; different test frames, so indicative only).

## (b) Pooled training, and what a specialist still buys

- **Pooled beats the docks tables on every foreign clip**, both
  profiles: micro −6% to −12%, full −7% to −25% (largest: judge
  full, 41.49 → 30.99). The price is home turf: on docks the pooled
  tables trail the docks tables by +6.8% (micro) / +12.4% (full) —
  docks is only ~27% of the pooled training cells.
- **The specialist premium is real, bounded, and roughly uniform.**
  Pooled vs home specialist, per clip: micro +3.8% (chiaroscuro) to
  +19.1% (judge); full +7.5% (chiaroscuro) to +26.4% (arena). No
  clip shows the signature of a domain the general table cannot
  represent — no error blow-up, no collapse (below); the premium
  looks like ordinary specialisation gain, not a rescue.

## Usage: who collapses, who doesn't

- **Fixed collapses on all five clips.** Codeword 2 takes 99.0–
  100.0% of test cells everywhere (chiaroscuro rounds to 100.0%:
  397,384 of 397,440). Codeword 3 is dead on all five clips;
  codeword 1 is additionally dead on lighthouses and judge (zero
  uses across those *entire* clips, train and test). On two of five
  sources the fixed tokenizer is a two-symbol device; on all five it
  is effectively a one-symbol device. V0's headline was not a docks
  quirk.
- **No learned table collapses anywhere.** Worst learned max-share
  on any clip's test set: docks-micro on lighthouses, 0.396 — vs
  fixed's 0.993 on the same cells. Micro tables (every method,
  every clip) have **zero dead codewords** in train and test.
- **Full tables shed a few specialised rows off-domain.** The
  shipped docks-full leaves 5 of 64 codewords unused on arena and
  judge, 3 on lighthouses, 0 on chiaroscuro and at home; pooled-full
  leaves 2 / 1 / 3 on arena / lighthouses / judge. These are rows
  fitted to structure the foreign clip doesn't contain. No test cell
  is ever assigned to a codeword that was dead in its table's own
  training (checked explicitly, 0 cells). One pooled-full codeword
  is nearly docks-specific in reverse: 13 uses in 568,320 docks test
  cells — the generalist keeps niche rows the mixture needs
  elsewhere. The only fit-dead rows anywhere: judge's own full
  specialist (2 of 64 dead on its own train).

## Verdict

**One general table — at this scale, with this evidence.** A single
pair of tables trained on the five-clip mixture is better than the
single-clip tables on four of five clips, within 4–19% (micro) /
8–26% (full) of a per-domain specialist *on the specialist's own
clip*, and never collapses. Per-domain tables are a bounded,
uniform premium — a deployment-time optimisation for a genuinely
single-purpose camera (where training one is cheap and the premium
is free accuracy), not a precondition for the representation to
travel. The v0 caveat "one clip, one harbour" is retired for the
quantisation proxy: the codebook generalises across every source we
have, including renders and a screen recording of ASCII art.

## Scope limits, stated

- The
...[truncated 2697 chars]
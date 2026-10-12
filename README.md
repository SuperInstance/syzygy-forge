# syzygy-forge — v0: teaching the tokenizer a codebook

## What this is, in plain English

Syzygy's kernel turns each little block of a camera picture into a tone
character — how bright that spot is, roughly, in one of ten steps from
space to `@`. The choice is made by a tiny fixed table of numbers that
somebody wrote by hand. It was never taught anything.

A **learned codebook** is the same table, except the numbers are found by
showing the system a lot of real footage and letting it sort what it sees
into piles: *blocks that look like this get this symbol*. Each pile gets a
prototype — the average block in the pile. Nothing about the kernel
changes. The table is still just integers, still scored the same way,
still gives the same answer on every machine. The only difference is that
the numbers in the table came from the world instead of from a guess.

This folder is the first working prototype of that idea (step **T1** of
the Syzygy Phase 2 architecture plan): codebooks trained by integer
k-means on the docks footage, frozen as data files, and honestly
evaluated against the hand-written table they would replace.

## What v0 found

Three results matter.

**1. The fixed table is effectively a one-symbol device on real footage.**
Across all 3,516,480 cells of the docks walk, its four codewords were
used [1,452 · 793 · 3,514,235 · 0] times: codeword 2 takes 99.94% of
cells, codeword 3 is dead (it provably can never win — its scores are
dominated by codeword 2's for every possible input), and the other two
are rounding errors. The tone plane still reads fine because the tone
character is mostly a direct brightness ramp with the codeword as a
±2-step nudge — but as a *learned-representation slot*, the fixed table
carries almost no information.

**2. Training with the kernel's literal scoring rule collapses the same
way.** If new prototypes are assigned cells by raw dot-product argmax
(the existing selector, unchanged), one prototype takes **every** cell in
both profiles, in a dozen iterations. The features all point in roughly
the "brighter" direction, so the biggest prototype wins everything. This
is the fixed table's disease, reproduced from data. Those tables are
kept (`tables/table_*_dotproduct_collapsed.json`) as a negative result,
not deleted.

**3. The fix is one bias term — and then the learned tables are
healthy and measurably better.** Train the same integer k-means with
nearest-centroid assignment instead. Nearest-centroid *is* dot-product
argmax once each codeword carries a bias (`argmax 2⟨c,x⟩ − ||c||²`), an
exact integer identity — verified cell-for-cell in Python (35,568 cells)
and again in C against the generated header. The resulting tables use
**every** codeword, and on held-out frames they quantise the feature
space far better than the fixed table:

| table | codewords | dead (train/test) | held-out quantisation error ↓ |
|---|---|---|---|
| fixed (hand-written) | 4 | 1 / 1 | 151.18 |
| learned, raw dot-product (collapsed) | 16 or 64 | 15 or 63 | 151.32 |
| **learned micro** | 16 | **0 / 0** | **44.63** |
| **learned full** | 64 | **0 / 0** | **29.87** |

Error = mean Euclidean distance from a held-out cell's feature vector to
its assigned codeword's centroid (centroids fitted on train cells for
every method alike — generous to the fixed table; details below). The
fixed table scores at the one-codeword floor (151.32): its partition of
the feature space is worth almost nothing. Micro is 3.4× better, full
5.1× better.

**Readability spot-check:** tone planes rendered from the learned tables
for frames 10, 120 and 260 (`render/`) keep the scene — dock perspective,
pilings, the fishing boat's hull and rigging all survive at least as well
as the fixed kernel's own tone output, while reassigning 69–78% of cells
to a different tone step. The learned planes are not a degraded copy of
the fixed one; they are a different, denser reading of the same cells.

**Verdict: the learned tables look viable.** The concept survives contact
with real footage, the failure mode of the naive approach is now known
and priced (one bias term), and both profiles come out the size the plan
promised — small enough for a browser tab or a microcontroller.

**4. Reader-side evaluation: the gate verdict is split** (full write-up
in [`readerside/RESULTS.md`](readerside/RESULTS.md)). Quantisation error
is only a proxy, so two models (MiniMax, GLM) answered 200 programmatically
ground-truthed questions over 40 docks frames rendered with the fixed
plane vs the learned micro plane, same grid, same characters per frame:

- **MiniMax: learned wins on every question type** — 0.430 vs 0.315
  fixed (+11.5 points); lift over blind +0.120 for learned vs +0.005 for
  fixed (on the fixed plane it read no more region truth than it
  guessed); on clear-margin frames the gap grows to +16.6. Sharpest
  exhibit: on the three frames where one band is provably empty of edges,
  the fixed plane went 0/3 — confident invented tallies naming the empty
  band — and the learned plane went 3/3. Mechanism: the fixed character
  tracks absolute brightness, so glare and sky read as "busy"; the
  learned assignment tracks local contrast, so the ink gathers where
  structure actually is.
- **GLM: a wash** — 0.325 learned vs 0.350 fixed. Edge-mass reading
  improved sharply (0.350 → 0.525) but centroid reading collapsed
  (0.600 → 0.325), cancelling out.
- **The stage-5 reader bar (learned ≥ fixed + 5 points) is met by one
  reader and missed by the other — the gate is not cleanly passed.**
- Over both planes sits a ceiling: a constant-answer baseline scores
  0.476, above every reader × version cell — at 148×80 the readers
  sketch the *shape* of counting rather than counting. Region truth at
  full grid belongs in the precomputed digest channel; the raw plane's
  measurable strength is position and structure.
- Named next variable: the render mapping. The forge's renderer squeezes
  16 codewords onto 10 ramp steps by brightness rank, distorting
  intervals — and interval questions (mass ratios, balance points) are
  exactly where GLM fell (frame 83 is the named suspect). A
  luma-calibrated mapping is the obvious next experiment.

**Correction, from v2 (2026-10-12)** — the named experiment ran, and it
acquitted the mapping (full argument in
[`readerside2/RESULTS-V2.md`](readerside2/RESULTS-V2.md)). All three
renderings — fixed, learned-rank, and the new learned-calibrated plane
(each codeword drawn by its centroid's actual mean luma on the kernel's
own scale) — were re-run in one fresh batch under the identical
protocol. The *unchanged* conditions moved 2.5–7 points between batches
at temperature 0.7 (up to 15 on a single question), and v1's headline
contrast above — MiniMax learned − fixed, +11.5 — redrew as +0.5
paired. The split verdict therefore stands as **one draw**, not a
measurement: the stage-5 reader bar (+5 points) is smaller than this
protocol's noise floor, and the gate cannot be adjudicated without
temperature 0 and/or roughly 4× the frames. The calibrated mapping is
adopted anyway, on principle — it renders the brightness intervals that
actually exist in the data rather than asserting intervals by rank —
with **no reader benefit claimed**: in v2 it was GLM's best cell and
MiniMax's worst, both inside noise. One finding is untouched: at
148×80, region truth travels via the digest — every reader cell in both
batches still sits below the constant-answer composite.

---

## Technical record

### Corpus

- Source: Casey's docks walk, the exact NV12 stream that produced the
  shipped text corpus (`syzygy-video/work/docks_base.nv12`, 296×320 px,
  297 frames at 6 fps → 148×80 cells per frame, **3,516,480 cells**).
- Extraction (`src/extract_features.c`) includes the kernel headers
  read-only (the kernel repo is never modified) and replicates the fused
  loop's per-cell reads: the 8 luma taps plus the ingest cell RGB. The
  fixed tokenizer's decision per cell is computed by the kernel's own
  `syz_ste_select4` / `syz_fused_tone`.
- **Verification:** the extracted tone output matches the shipped
  `docks-text/tone` corpus on **all 3,516,480 cells (0 mismatches)** —
  the forge trains on provably the same numbers the kernel produced.
- Split (frozen in `eval/split_manifest.json`): by frame,
  `index % 5 == 4` → test. Train 238 frames / 2,817,920 cells;
  test 59 frames / 698,560 cells.

### Features (16 per cell, all integer)

Computed from the cell's 2×4 block — the same pixels the kernel reads.
Means use round-half-up; the full definitions are in
`src/extract_features.c` and `corpus/features_meta.json`.

| # | feature | # | feature |
|---|---|---|---|
| 0 | mean_luma (8 taps) | 8 | red (cell RGB) |
| 1 | min_luma | 9 | blue (cell RGB) |
| 2 | max_luma | 10 | cell_luma = `syz_luma8(RGB)` |
| 3 | top_mean | 11 | green − (red+blue)/2 |
| 4 | bottom_mean | 12 | grad_y = bottom − top |
| 5 | left_mean | 13 | grad_x = right − left |
| 6 | right_mean | 14 | diag_diff (zigzag diagonal contrast) |
| 7 | green (cell RGB) | 15 | texture (mean abs deviation) |

Features 7 and 10 are the exact two values the fixed tokenizer scores.
**Micro (N=16, K=8)** uses features 0–7; **full (N=64, K=16)** uses all 16,
matching the plan's profile shapes. (Feature *semantics* are the forge's
v0 choice; the plan's schema freeze has not happened yet.)

### Training (`src/train_kmeans.py`)

Integer Lloyd iterations: seeded init from real corpus cells
(seed 20261011), 200,000-cell seeded subsample for iteration speed,
centroids **rounded back to integers every step**, dead codewords kept
and recorded (never resurrected). Scores are computed in float64 during
iteration — exact for these ranges (all dots < 2²¹ ≪ 2⁵³), so assignments
are bit-identical to int64; final usage histograms are pure int64 over
the full train set.

- micro: exact convergence in 41 iterations.
- full: stopped at the 100-iteration cap inside an integer limit cycle —
  2–6 of 64 rows oscillate by ±1 while assignment counts stay stable
  (per-codeword subsample counts move by ≤4 cells). The frozen table is
  the cap iteration; a partition-stability stopping rule is the proper
  fix and is noted for 0010 below.
- dot-product variants "converged" (12 / 18 iterations) to the collapsed
  single-codeword state.

### Frozen tables (`tables/`)

| table | id (FNV-1a) | rows (int32) | notes |
|---|---|---|---|
| micro | `0x7ccb88b5` | 512 B | + 512 B inference rows + 64 B bias |
| full | `0x6f81750a` | 4,096 B | + 4,096 B inference rows + 256 B bias |

- `table_micro.json` / `table_full.json`: centroid `rows`, the
  `inference_rows` (= 2·rows) and `inference_bias` (= −‖row‖²) forms,
  training provenance, iteration logs, full-train usage.
- `syz_tok_tables.h`: generated SYZT-style C header (both tables,
  descriptor array); compiles against the kernel headers, and a C smoke
  test confirms biased selection == direct nearest-centroid.
- Table id = FNV-1a/32 over centroid rows serialised little-endian
  int32, row-major (the plan's identity rule). The plan's ~340 B / ~2.4 KB
  size targets correspond to int16 row packing (centroid values lie in
  [−255, 255], so int16 loses nothing); biases need int32.
- `table_*_dotproduct_collapsed.json`: the negative-result tables.

### Evaluation detail (`eval/eval_results.json`)

- Usage, test set: fixed max/min per codeword 698,121 / 0 (of 698,560);
  micro 78,508 / 12,832; full 29,922 / 2,527; collapsed 698,560 / 0.
  No learned (euclid) codeword is dead on train or test.
- Partition quantisation error (the headline metric): post-hoc centroids
  fitted on train assignments **for every method identically**, including
  fixed — the fixed table thereby gets 16-dim centroids it never earned,
  biasing the comparison *against* the learned tables. Mean squared
  errors are in the JSON (fixed 30,127; micro 2,738; full 1,183;
  collapsed 30,183).
- Native-space error (distance to the *winning table row* in the table's
  own space): micro 26.54 (8-dim), full 29.90 (16-dim) — not comparable
  across different K. For reference, distance to the fixed basis rows in
  their 2 active dims is 115.09, but basis rows are not centroids, so
  that number is indicative only.
- Identity check: biased inference == direct nearest-centroid on 3 test
  frames × 2 profiles, 0 mismatched cells (35,568 cells), plus the C
  smoke test above.
- Caveats: one clip, one location, daylight harbour only. Quantisation
  error is a proxy — the plan's stage-5 gate also requires reader/probe
  gains, which v0 does not measure. The fixed tone character is mostly a
  direct luma ramp, so the partition metric understates the fixed tone
  plane's readability (the renders show all three planes reading well).

### What the real shard-0010 build still needs

1. **Ratify the bias term.** The collapse result is the evidence: the
   learned selector cannot be the raw `syz_ste_select` on these
   features. 0010 must add a per-codeword bias (or a constant-1 feature
   column) to the selection rule — or switch the learned plane to a
   normalised (cosine-style) integer score, untested here.
2. **Freeze the feature schema** (exact integer definitions + rounding)
   and the SYZT blob serialisation (layout, endianness, version), then
   the (kernel, table, frame) golden-hash triple and the 4,096
   conformance vectors from the plan.
3. **A stopping rule for integer Lloyd** (partition-stable, not
   centroid-stable) — the N=64 run limit-cycles at ±1.
4. **T2**: distillation against teacher labels with distortion + usage
   terms (T1 here is purely unsupervised); nothing in v0 shapes usage
   beyond what the data does on its own.
5. **Range proof, formalised**: at the frozen feature ranges, biased
   scores are bounded by ~2²¹ in magnitude — int32 suffices with wide
   margin; the kernel scores in int64 regardless. Needs writing into the
   shard's mark, plus the MCU perf witness the repo still owes overall.
6. **A bigger, more varied corpus** (more clips, light, weather, scenes)
   and the open question of per-domain tables vs one general table.
7. **Reader-side evaluation** — *done; v1 split, v2 noise floor* (see
   finding 4 above, `readerside/`, and `readerside2/`): v1 drew MiniMax
   +11.5 on the learned plane and GLM a wash; v2's identical-protocol
   redraw showed that contrast is inside batch noise at temperature
   0.7, so the stage-5 reader gate cannot be adjudicated without
   temperature 0 and/or ~4× frames. The luma-calibrated mapping is
   adopted on principle, no benefit claimed. Next: route region
   questions through the digest.

### Layout

```
corpus/   features_meta.json (+ features.bin locally — 114 MB of raw
          per-cell features, regenerable with src/extract_features.c
          from the syzygy-video corpus; excluded from this repo)
src/      extract_features.c, verify_and_meta.py, train_kmeans.py,
          eval.py, gen_header.py
tables/   table_micro.json, table_full.json, syz_tok_tables.h,
          tables_index.json, table_*_dotproduct_collapsed.json
eval/     split_manifest.json, eval_results.json
render/   frame_{0010,0120,0260}_tone_compare.{txt,png}
readerside/  the reader-side evaluation: RESULTS.md, GUIDE.md,
          questions.jsonl, reader_outputs.jsonl (240 records),
          scores.json, failures.json, renders/{fixed,learned}/
readerside2/  the v2 follow-up — the render mapping on trial, and the
          batch indicted instead: RESULTS-V2.md, mapping.json,
          scores_v2.json, reader_outputs.jsonl (240 records),
          renders/{fixed,learned,learned_cal}/
```

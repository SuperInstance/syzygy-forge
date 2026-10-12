#!/usr/bin/env python3
"""Reader-side evaluation dataset: learned micro table vs fixed table,
read by models, on real footage (the stage-5 gate's reader-side question).

For ~40 frames spread across the docks clip:
  * render (a) the kernel's FIXED-table tone plane — the shipped
    syzygy-video docks-text/tone frames (cross-checked cell-for-cell
    against the forge corpus features.bin TONE, which the forge verified
    matches the shipped corpus on all 3,516,480 cells);
  * render (b) the LEARNED micro table's tone plane, using the forge's
    own renderer (src/eval.py `learned_plane`, copied verbatim below):
    biased-argmax assignment over the micro table, codewords ordered by
    centroid mean_luma (dark->bright), rank mapped onto the SAME 10-step
    ramp " .:-=+*#%@". The only difference between (a) and (b) is which
    codeword each cell gets;
  * 5 closed-form questions per frame whose answers are computed
    PROGRAMMATICALLY from the source pixels (the NV12 Y plane) — never
    from a model — with the full computation recorded per frame.

Ground-truth definitions (pixel grid 296 wide x 320 high; cell grid
148x80, each cell 2x4 px):
  quadrants: cols [0,148)/[148,296), rows [0,160)/[160,320)
  Q1 brightest quadrant = argmax of Y sums; Q2 darkest = argmin.
  Q3 centroid side: xbar = sum(x*Y)/sum(Y); left iff xbar < 148.
  Sobel on Y (3x3, interior only), magnitude m = |gx| + |gy|;
  edge pixel iff m > 200. Q4 = side (left/right col halves) with more
  edge pixels. Q5 = band with most edge pixels; bands by pixel rows
  [0,108) top, [108,216) middle, [216,320) bottom (cell rows 1-27 /
  28-54 / 55-80 in 1-based terms).
  Mass sums (|gx|+|gy| totals) are recorded alongside the counts; if
  count-argmax and mass-argmax ever disagree, the record says so and
  the count is the scored truth (the task's wording is "counts").
"""
import json
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORGE = Path.home() / "workspace/syzygy-forge"
VIDEO = Path.home() / "workspace/syzygy-video"
NV12 = VIDEO / "work/docks_base.nv12"
W, H, NFRAMES = 296, 320, 297
COLS, ROWS = 148, 80
RAMP = " .:-=+*#%@"
N_SAMPLE = 40
EDGE_THRESH = 200

# ---------- sampling (deterministic spread across the clip) ----------
sample = sorted({int(round(v)) for v in np.linspace(0, NFRAMES - 1, N_SAMPLE)})
assert len(sample) == N_SAMPLE, sample

# ---------- renders ----------
dt = np.dtype([("feat", "<i2", (16,)), ("sel", "u1"), ("tone", "u1")])
data = np.fromfile(FORGE / "corpus/features.bin", dtype=dt).reshape(NFRAMES, ROWS, COLS)
FEAT = data["feat"].astype(np.int64)
TONE = data["tone"]
micro = json.loads((FORGE / "tables/table_micro.json").read_text())


def assign_learned(fr, table):
    """Verbatim from syzygy-forge/src/eval.py."""
    X = FEAT[fr].reshape(-1, 16)[:, table["feature_indices"]]
    R = np.array(table["inference_rows"], dtype=np.int64)
    B = np.array(table["inference_bias"], dtype=np.int64)
    return np.argmax(X @ R.T + B, axis=1)


def learned_plane(fr, table):
    """Verbatim from syzygy-forge/src/eval.py."""
    a = assign_learned(fr, table).reshape(ROWS, COLS)
    rows = np.array(table["rows"], dtype=np.int64)
    order = np.argsort(rows[:, 0], kind="stable")  # by row mean_luma, dark->bright
    rank = np.empty(len(rows), dtype=np.int64)
    rank[order] = np.arange(len(rows))
    idx = np.minimum(rank[a] * 10 // len(rows), 9)
    return idx


def plane_to_text(plane):
    return "\n".join("".join(RAMP[int(v)] for v in row) for row in plane)


# reproduction check: my copied renderer must reproduce the forge's shipped
# compare panels (middle panel = learned micro) for frames 10 and 120.
for chk in (10, 120):
    shipped = (FORGE / f"render/frame_{chk:04d}_tone_compare.txt").read_text().splitlines()[1:]
    mine = plane_to_text(learned_plane(chk, micro)).splitlines()
    theirs = [ln.split(" | ")[1] for ln in shipped]
    assert mine == theirs, f"learned renderer mismatch on frame {chk}"
print("learned renderer reproduces forge compare panels (frames 10, 120): OK")

# ---------- ground truth from pixels ----------
nv12 = np.fromfile(NV12, dtype=np.uint8)
frame_bytes = W * H * 3 // 2
assert nv12.size == NFRAMES * frame_bytes
Y = nv12[: NFRAMES * W * H].reshape(NFRAMES, H, W).astype(np.int64)

xs = np.arange(W, dtype=np.int64)[None, :]


def sobel_mag(y):
    gx = (-y[:-2, :-2] - 2 * y[1:-1, :-2] - y[2:, :-2]
          + y[:-2, 2:] + 2 * y[1:-1, 2:] + y[2:, 2:])
    gy = (-y[:-2, :-2] - 2 * y[:-2, 1:-1] - y[:-2, 2:]
          + y[2:, :-2] + 2 * y[2:, 1:-1] + y[2:, 2:])
    return np.abs(gx) + np.abs(gy)  # shape (H-2, W-2), interior pixels


QUADS = ["TL", "TR", "BL", "BR"]
questions = []
frames_meta = []
mismatch_cells = 0

for fr in sample:
    y = Y[fr]
    # --- renders ---
    # shipped files carry a one-line header ("# frame N peak_bin=.. cols=..
    # rows=.."); strip it so both versions are bare 148x80 grids — the grid
    # content below the header is the shipped fixed-table tone plane.
    raw_lines = (VIDEO / f"out/docks-text/tone/frame_{fr:04d}.txt").read_text().splitlines()
    if raw_lines and raw_lines[0].startswith("#"):
        raw_lines = raw_lines[1:]
    assert len(raw_lines) == ROWS and all(len(ln) == COLS for ln in raw_lines), fr
    fixed_text = "\n".join(raw_lines)
    # cross-check shipped file vs forge corpus TONE (proven identical corpus-wide)
    corpus_text = plane_to_text(TONE[fr].astype(np.int64))
    if fixed_text != corpus_text:
        # count differing cells for the record; shipped file stays authoritative
        a = np.array([[RAMP.index(c) for c in ln] for ln in fixed_text.splitlines()])
        b = np.array([[RAMP.index(c) for c in ln] for ln in corpus_text.splitlines()])
        mismatch_cells += int(np.count_nonzero(a != b))
    learned_text = plane_to_text(learned_plane(fr, micro))
    (HERE / f"renders/fixed/frame_{fr:04d}.txt").write_text(fixed_text + "\n")
    (HERE / f"renders/learned/frame_{fr:04d}.txt").write_text(learned_text + "\n")
    changed = int(np.count_nonzero(
        np.array([[RAMP.index(c) for c in ln] for ln in fixed_text.splitlines()])
        != np.array([[RAMP.index(c) for c in ln] for ln in learned_text.splitlines()])))
    frames_meta.append({"frame": fr, "cells_changed_fixed_vs_learned": changed})

    # --- Q1/Q2: quadrant tone sums ---
    qs = {
        "TL": int(y[:160, :148].sum()), "TR": int(y[:160, 148:].sum()),
        "BL": int(y[160:, :148].sum()), "BR": int(y[160:, 148:].sum()),
    }
    total = sum(qs.values())
    srt = sorted(qs.items(), key=lambda kv: -kv[1])
    q1 = srt[0][0]
    q1_margin = (srt[0][1] - srt[1][1]) / total
    srt2 = sorted(qs.items(), key=lambda kv: kv[1])
    q2 = srt2[0][0]
    q2_margin = (srt2[1][1] - srt2[0][1]) / total

    # --- Q3: tone centroid side ---
    xbar = float((y * xs).sum() / y.sum())  # pixel units, centre = 148
    q3 = "left" if xbar < W / 2 else "right"
    q3_margin_cells = abs(xbar - W / 2) / 2.0

    # --- Q4/Q5: Sobel edge counts ---
    m = sobel_mag(y)  # rows 1..318, cols 1..294 in original pixel coords
    edge = m > EDGE_THRESH
    # halves over interior cols: original col < 148 -> interior col index < 147
    left_count = int(edge[:, :147].sum())
    right_count = int(edge[:, 147:].sum())
    q4 = "left" if left_count >= right_count else "right"
    left_mass = int(m[:, :147].sum())
    right_mass = int(m[:, 147:].sum())
    q4_mass_side = "left" if left_mass >= right_mass else "right"
    # bands over interior rows: original row r -> interior r-1;
    # top rows [0,108) -> interior [0,107); middle [108,216) -> [107,215); bottom -> [215,318)
    band_counts = {
        "top": int(edge[:107].sum()),
        "middle": int(edge[107:215].sum()),
        "bottom": int(edge[215:].sum()),
    }
    band_mass = {
        "top": int(m[:107].sum()),
        "middle": int(m[107:215].sum()),
        "bottom": int(m[215:].sum()),
    }
    q5 = max(band_counts, key=band_counts.get)
    q5_mass = max(band_mass, key=band_mass.get)
    bc = sorted(band_counts.values(), reverse=True)
    q5_margin = (bc[0] - bc[1]) / max(1, bc[0])

    comp = {
        "quadrant_y_sums": qs, "q1_margin_frac": round(q1_margin, 4),
        "q2_margin_frac": round(q2_margin, 4),
        "centroid_x_px": round(xbar, 2), "centroid_x_cells": round(xbar / 2, 2),
        "q3_margin_cells": round(q3_margin_cells, 2),
        "edge_counts_half": {"left": left_count, "right": right_count},
        "edge_mass_half": {"left": left_mass, "right": right_mass},
        "q4_count_vs_mass_agree": q4 == q4_mass_side,
        "edge_counts_band": band_counts, "edge_mass_band": band_mass,
        "q5_count_vs_mass_agree": q5 == q5_mass,
        "q5_margin_frac": round(q5_margin, 4),
        "edge_threshold": EDGE_THRESH,
    }
    answers = {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "Q5": q5}
    questions.append({"frame": fr, "answers": answers, "computation": comp})
    print(f"frame {fr:3d}: Q1={q1} Q2={q2} Q3={q3} Q4={q4} Q5={q5} "
          f"(centroid {xbar / 2:.1f} cells, changed cells {changed})")

(HERE / "questions.jsonl").write_text(
    "".join(json.dumps(q) + "\n" for q in questions))
(HERE / "frames_sampled.json").write_text(json.dumps({
    "sample_rule": "40 frames, np.linspace(0, 296, 40) rounded, sorted unique",
    "frames": sample,
    "per_frame": frames_meta,
    "shipped_vs_corpus_tone_mismatches_on_sample": mismatch_cells,
}, indent=2) + "\n")
print(f"\nwrote {len(questions)} question records; "
      f"shipped-vs-corpus tone mismatches on sampled frames: {mismatch_cells}")

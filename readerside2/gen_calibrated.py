#!/usr/bin/env python3
"""Reader-side evaluation v2: ONE variable — the render mapping.

v1 (~/workspace/syzygy-forge/readerside/) rendered the learned micro
table by RANK: the 16 codewords ordered by centroid mean_luma and
squeezed uniformly onto the 10-step ramp (rank * 10 // 16). That
preserves brightness ORDER but destroys INTERVALS — and the interval
questions (Q1 quadrant mass, Q3 centroid balance) are exactly where
GLM's learned-plane reading collapsed in v1 (Q3 0.600 fixed -> 0.325
learned-rank; frame 83 the named case).

v2 adds one condition, LEARNED-CALIBRATED: the SAME per-cell codeword
assignments (biased argmax over the frozen micro table, verbatim from
the forge), but the character for codeword cw is chosen by the
kernel's own luma->ramp scale applied to the centroid's actual mean
luma:

    idx_cw = min( (L_cw * 9) // 255, 9 ),   L_cw = table rows[cw][0]

That is literally the base term of syz_fused_tone (syzygy
include/syz_fused.h: `idx = ((uint32_t)luma * 9u) / 255u + sel;`), with
luma := the codeword centroid's mean_luma (feature 0 — the same
quantity the v1 rank ordering sorted by; micro K=8 features are 0-7,
so cell_luma/feature 10 is not in the table's space). Codewords whose
centroid lumas fall within one ramp step (~28.3 luma units) share a
character; wider true gaps keep proportionally wider character gaps.
Intervals preserved; resolution is the ramp's own.

SANITY GATE (must pass before any reader runs):
  1. v1 learned-rank planes regenerated with the verbatim forge
     renderer == readerside/renders/learned/*.txt, byte for byte,
     all 40 sampled frames (plus the forge compare-panel check for
     frames 10/120, as in v1's gen_dataset.py).
  2. v1 fixed planes (readerside/renders/fixed/*.txt) == the shipped
     syzygy-video tone files with header stripped, all 40 frames.
  3. The grids embedded in v1 reader_outputs.jsonl prompts == the
     render files on disk, for every successful fixed/learned record —
     i.e. the scores being compared against are the scores of THESE
     bytes.
"""
import json
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1 = Path.home() / "workspace/syzygy-forge/readerside"
FORGE = Path.home() / "workspace/syzygy-forge"
VIDEO = Path.home() / "workspace/syzygy-video"
COLS, ROWS, NFRAMES = 148, 80, 297
RAMP = " .:-=+*#%@"

qrecs = [json.loads(l) for l in open(V1 / "questions.jsonl")]
FRAMES = [q["frame"] for q in qrecs]
micro = json.loads((FORGE / "tables/table_micro.json").read_text())

dt = np.dtype([("feat", "<i2", (16,)), ("sel", "u1"), ("tone", "u1")])
data = np.fromfile(FORGE / "corpus/features.bin", dtype=dt).reshape(NFRAMES, ROWS, COLS)
FEAT = data["feat"].astype(np.int64)


def assign_learned(fr, table):
    """Verbatim from syzygy-forge/src/eval.py."""
    X = FEAT[fr].reshape(-1, 16)[:, table["feature_indices"]]
    R = np.array(table["inference_rows"], dtype=np.int64)
    B = np.array(table["inference_bias"], dtype=np.int64)
    return np.argmax(X @ R.T + B, axis=1)


def rank_index(table):
    """v1 mapping, verbatim from src/eval.py learned_plane."""
    rows = np.array(table["rows"], dtype=np.int64)
    order = np.argsort(rows[:, 0], kind="stable")  # by centroid mean_luma
    rank = np.empty(len(rows), dtype=np.int64)
    rank[order] = np.arange(len(rows))
    return np.minimum(rank * 10 // len(rows), 9), rank


def cal_index(table):
    """v2 calibrated mapping: kernel luma scale on centroid mean_luma."""
    L = np.array(table["rows"], dtype=np.int64)[:, 0]
    return np.minimum(L * 9 // 255, 9), L


RANK_IDX, RANK = rank_index(micro)
CAL_IDX, CENT_LUMA = cal_index(micro)


def plane_text(fr, char_idx):
    a = assign_learned(fr, micro).reshape(ROWS, COLS)
    idx = char_idx[a]
    return "\n".join("".join(RAMP[int(v)] for v in row) for row in idx)


def grid_of(text):
    return np.array([[RAMP.index(c) for c in ln] for ln in text.splitlines()])


# ---------- sanity gate ----------
gate = {}

# (1a) forge compare panels, frames 10/120 (v1's own check)
for chk in (10, 120):
    shipped = (FORGE / f"render/frame_{chk:04d}_tone_compare.txt").read_text().splitlines()[1:]
    theirs = [ln.split(" | ")[1] for ln in shipped]
    assert plane_text(chk, RANK_IDX).splitlines() == theirs, f"panel mismatch frame {chk}"
gate["forge_compare_panels_10_120"] = "OK"

# (1b) regenerated rank planes == v1 render files, all sampled frames
bad = [fr for fr in FRAMES
       if plane_text(fr, RANK_IDX) != (V1 / f"renders/learned/frame_{fr:04d}.txt").read_text().rstrip("\n")]
assert not bad, f"learned-rank regeneration mismatch on {bad}"
gate["learned_rank_regen_vs_v1_files"] = f"OK ({len(FRAMES)}/{len(FRAMES)} byte-identical)"

# (2) v1 fixed files == shipped tone files (header stripped)
bad = []
for fr in FRAMES:
    raw = (VIDEO / f"out/docks-text/tone/frame_{fr:04d}.txt").read_text().splitlines()
    if raw and raw[0].startswith("#"):
        raw = raw[1:]
    if "\n".join(raw) != (V1 / f"renders/fixed/frame_{fr:04d}.txt").read_text().rstrip("\n"):
        bad.append(fr)
assert not bad, f"fixed mismatch on {bad}"
gate["fixed_vs_shipped_tone_files"] = f"OK ({len(FRAMES)}/{len(FRAMES)} byte-identical)"

# (3) grids embedded in v1 prompts == render files on disk
n_checked = 0
for l in open(V1 / "reader_outputs.jsonl"):
    r = json.loads(l)
    if r.get("error") is not None or not r.get("raw_output"):
        continue
    cond = r["condition"]
    if cond not in ("fixed", "learned"):
        continue
    grid = r["prompt"].split("Frame:\n```\n", 1)[1].split("\n```", 1)[0]
    disk = (V1 / f"renders/{cond}/frame_{r['frame']:04d}.txt").read_text().rstrip("\n")
    assert grid == disk, f"prompt grid mismatch: {cond} frame {r['frame']}"
    n_checked += 1
gate["v1_prompt_grids_vs_render_files"] = f"OK ({n_checked} records)"
print("SANITY GATE PASSED:", json.dumps(gate, indent=2))

# ---------- mapping record ----------
mapping = {
    "ramp": RAMP,
    "kernel_scale": "idx = min((luma * 9) // 255, 9) — base term of syz_fused_tone "
                    "(syzygy/include/syz_fused.h); fixed plane adds sel in {0,1,2} "
                    "from the fixed selector on top of this base.",
    "quantity_mapped": "centroid mean_luma = table_micro rows[cw][0] (feature 0; "
                       "micro table feature space is features 0-7, so this is the "
                       "centroid's own luma summary, and the same quantity the v1 "
                       "rank ordering sorted by).",
    "codewords": [
        {"cw": int(cw), "centroid_mean_luma": int(CENT_LUMA[cw]),
         "rank": int(RANK[cw]), "rank_char_idx": int(RANK_IDX[cw]),
         "cal_char_idx": int(CAL_IDX[cw]),
         "rank_char": RAMP[int(RANK_IDX[cw])], "cal_char": RAMP[int(CAL_IDX[cw])]}
        for cw in range(len(CENT_LUMA))
    ],
    "distinct_chars_rank": int(len(set(RANK_IDX.tolist()))),
    "distinct_chars_cal": int(len(set(CAL_IDX.tolist()))),
}
(HERE / "mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
print("\ncw  mean_luma  rank->char   cal->char")
for c in mapping["codewords"]:
    print(f"{c['cw']:3d}  {c['centroid_mean_luma']:9d}  {c['rank']:4d} -> {c['rank_char']!r}"
          f"   {c['cal_char_idx']} -> {c['cal_char']!r}")
print(f"distinct ramp chars used: rank {mapping['distinct_chars_rank']}/10, "
      f"calibrated {mapping['distinct_chars_cal']}/10")

# ---------- calibrated renders + reuse copies ----------
import shutil
for name in ("questions.jsonl", "frames_sampled.json", "GUIDE.md"):
    shutil.copyfile(V1 / name, HERE / name)
for cond in ("fixed", "learned"):
    (HERE / f"renders/{cond}").mkdir(parents=True, exist_ok=True)
    for fr in FRAMES:
        shutil.copyfile(V1 / f"renders/{cond}/frame_{fr:04d}.txt",
                        HERE / f"renders/{cond}/frame_{fr:04d}.txt")

meta = []
for fr in FRAMES:
    cal = plane_text(fr, CAL_IDX)
    (HERE / f"renders/learned_cal/frame_{fr:04d}.txt").write_text(cal + "\n")
    g_cal = grid_of(cal)
    g_rank = grid_of(plane_text(fr, RANK_IDX))
    g_fix = grid_of((V1 / f"renders/fixed/frame_{fr:04d}.txt").read_text().rstrip("\n"))
    meta.append({
        "frame": fr,
        "cells_cal_vs_rank": int(np.count_nonzero(g_cal != g_rank)),
        "cells_cal_vs_fixed": int(np.count_nonzero(g_cal != g_fix)),
    })
(HERE / "frames_meta_v2.json").write_text(json.dumps({
    "sanity_gate": gate, "per_frame": meta,
    "mean_cells_cal_vs_rank": float(np.mean([m["cells_cal_vs_rank"] for m in meta])),
    "mean_cells_cal_vs_fixed": float(np.mean([m["cells_cal_vs_fixed"] for m in meta])),
}, indent=2) + "\n")
print(f"\ncalibrated renders written for {len(FRAMES)} frames; "
      f"mean cells changed cal-vs-rank {np.mean([m['cells_cal_vs_rank'] for m in meta]):.0f}/11840, "
      f"cal-vs-fixed {np.mean([m['cells_cal_vs_fixed'] for m in meta]):.0f}/11840")

#!/usr/bin/env python3
"""Verify the forge extraction reproduces the shipped docks-text corpus
(kernel tone output) exactly, and write corpus metadata + the split manifest.

Split rule (by frame): test = frames with index % 5 == 4 (60 frames),
train = the other 237 frames. Deterministic, no RNG.
"""
import json
import numpy as np
from pathlib import Path

FORGE = Path.home() / "workspace/syzygy-forge"
TONE_DIR = Path.home() / "workspace/syzygy-video/out/docks-text/tone"
RAMP = " .:-=+*#%@"
COLS, ROWS, NFRAMES = 148, 80, 297

dt = np.dtype([("feat", "<i2", (16,)), ("sel", "u1"), ("tone", "u1")])
data = np.fromfile(FORGE / "corpus/features.bin", dtype=dt)
assert data.shape[0] == NFRAMES * ROWS * COLS, data.shape
data = data.reshape(NFRAMES, ROWS, COLS)

# --- verify tone bytes against the shipped text corpus -------------------
total_mismatch = 0
for fi in range(NFRAMES):
    lines = (TONE_DIR / f"frame_{fi:04d}.txt").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# frame"), lines[0]
    grid = lines[1:]
    assert len(grid) == ROWS, (fi, len(grid))
    ref = np.array([[RAMP.index(ch) for ch in row] for row in grid], dtype=np.uint8)
    assert ref.shape == (ROWS, COLS), (fi, ref.shape)
    mine = data[fi]["tone"]
    mism = int(np.count_nonzero(ref != mine))
    total_mismatch += mism
    if mism and fi < 3:
        print(f"frame {fi}: {mism} mismatched cells")
print(f"tone verification: {NFRAMES} frames, {total_mismatch} mismatched cells "
      f"of {NFRAMES * ROWS * COLS}")

# --- corpus metadata ------------------------------------------------------
feat_names = ["mean_luma", "min_luma", "max_luma", "top_mean", "bottom_mean",
              "left_mean", "right_mean", "green", "red", "blue", "cell_luma",
              "green_minus_rb", "grad_y", "grad_x", "diag_diff", "texture"]
sel_hist = np.bincount(data["sel"].ravel(), minlength=4).tolist()
meta = {
    "source_nv12": "~/workspace/syzygy-video/work/docks_base.nv12",
    "source_video": "~/workspace/user/media_library/video/5a/5acbb9b592d6c5b15d516318b6e1a5b534fc55c36b5c71656c79d8a9539475e0.mp4",
    "grid": {"cols": COLS, "rows": ROWS, "frames": NFRAMES,
             "cells_per_frame": COLS * ROWS, "total_cells": NFRAMES * COLS * ROWS},
    "extractor": "src/extract_features.c (kernel headers included read-only; "
                 "features replicate syz_fused's per-cell reads; fixed sel/tone "
                 "computed by the kernel's own syz_ste_select4 / syz_fused_tone)",
    "features": feat_names,
    "feature_notes": "int16; block stats over the cell's 8 luma taps (means "
                     "round-half-up); features 7,10 are the exact two values the "
                     "fixed tokenizer scores (green, cell_luma). Micro K=8 uses "
                     "features 0-7; full K=16 uses all.",
    "record_layout": "per cell: 16 x int16 features, uint8 fixed_sel, uint8 fixed_tone_idx; frame-major, row-major",
    "fixed_sel_histogram_all_frames": sel_hist,
    "tone_verification_vs_shipped_corpus": {
        "frames_checked": NFRAMES, "mismatched_cells": total_mismatch},
}
(FORGE / "corpus/features_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

# --- split manifest --------------------------------------------------------
test_frames = [i for i in range(NFRAMES) if i % 5 == 4]
train_frames = [i for i in range(NFRAMES) if i % 5 != 4]
manifest = {
    "rule": "frame index % 5 == 4 -> test; all other frames -> train",
    "train_frames": train_frames,
    "test_frames": test_frames,
    "train_frame_count": len(train_frames),
    "test_frame_count": len(test_frames),
    "train_cells": len(train_frames) * COLS * ROWS,
    "test_cells": len(test_frames) * COLS * ROWS,
}
(FORGE / "eval/split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"split: {len(train_frames)} train / {len(test_frames)} test frames")
print("sel histogram (all frames):", sel_hist)

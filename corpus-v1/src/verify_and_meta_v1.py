#!/usr/bin/env python3
"""Corpus v1 verification + metadata.

For each of the five stage-0 clips, verify the forge extraction against
stage 0's independent native implementation: FNV-1a/32 over each frame's
tone plane bytes must equal the `native.tone` hash recorded in stage 0's
hashes.jsonl. Two byte conventions are tested on the first frame of the
first clip (tone-index bytes vs ASCII ramp bytes); the matching one is
then applied to every frame of every clip.

Also writes, per clip, features_meta_<clip>.json and the frozen
split_manifest_v1.json (same rule as v0: frame index % 5 == 4 -> test).
"""
import json
import numpy as np
from pathlib import Path

FORGE = Path.home() / "workspace/syzygy-forge"
V1 = FORGE / "corpus-v1"
S0 = Path.home() / "workspace/syzygy-program/stage0"
RAMP = " .:-=+*#%@"
COLS, ROWS = 80, 24
CLIPS = ["docks", "arena", "chiaroscuro", "lighthouses", "judge"]
NV12 = {"docks": "docks.nv12", "arena": "arena.nv12",
        "chiaroscuro": "chiaroscuro.nv12", "lighthouses": "lighthouses.nv12",
        "judge": "judge.nv12"}

dt = np.dtype([("feat", "<i2", (16,)), ("sel", "u1"), ("tone", "u1")])

# stage-0 native tone hashes, grouped by clip in stream order
ref = {c: [] for c in CLIPS}
for line in (S0 / "hashes.jsonl").read_text().splitlines():
    rec = json.loads(line)
    clip = rec["frame_key"].split(":")[0]
    ref[clip].append(rec["native"]["tone"])


def fnv1a(data: bytes) -> int:
    h = 2166136261
    for b in data:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def load(clip):
    d = np.fromfile(V1 / "features" / f"features_{clip}.bin", dtype=dt)
    nframes = d.shape[0] // (ROWS * COLS)
    assert d.shape[0] == nframes * ROWS * COLS, (clip, d.shape)
    return d.reshape(nframes, ROWS, COLS)


# --- decide the tone byte convention on docks frame 0 ---------------------
RAMP_BYTES = np.array([ord(c) for c in RAMP], dtype=np.uint8)
probe = load("docks")
tone0 = probe[0]["tone"].ravel()
idx_hex = f"{fnv1a(tone0.tobytes()):08x}"
ascii_hex = f"{fnv1a(RAMP_BYTES[tone0].tobytes()):08x}"
print(f"docks:0 ref={ref['docks'][0]} idx={idx_hex} ascii={ascii_hex}")
if idx_hex == ref["docks"][0]:
    MODE = "index"
elif ascii_hex == ref["docks"][0]:
    MODE = "ascii"
else:
    raise SystemExit("NEITHER tone byte convention matches stage 0 — stop")
print(f"tone byte convention: {MODE}")

# --- full verification + per-clip meta ------------------------------------
split = {"rule": "frame index % 5 == 4 -> test; all other frames -> train "
                 "(same rule as v0, applied per clip)",
         "clips": {}}
summary = {}
for clip in CLIPS:
    d = load(clip)
    nframes = d.shape[0]
    assert nframes == len(ref[clip]), (clip, nframes, len(ref[clip]))
    bad = 0
    for fi in range(nframes):
        t = d[fi]["tone"].ravel()
        b = t.tobytes() if MODE == "index" else RAMP_BYTES[t].tobytes()
        if f"{fnv1a(b):08x}" != ref[clip][fi]:
            bad += 1
    sel_hist = np.bincount(d["sel"].ravel(), minlength=4).tolist()
    test_frames = [i for i in range(nframes) if i % 5 == 4]
    train_frames = [i for i in range(nframes) if i % 5 != 4]
    split["clips"][clip] = {
        "train_frames": train_frames, "test_frames": test_frames,
        "train_frame_count": len(train_frames),
        "test_frame_count": len(test_frames),
        "train_cells": len(train_frames) * COLS * ROWS,
        "test_cells": len(test_frames) * COLS * ROWS}
    meta = {
        "clip": clip,
        "source_nv12": f"~/workspace/syzygy-program/stage0/work/{NV12[clip]}",
        "source_provenance": "stage-0 extracted NV12 stream (see "
                             "syzygy-program/stage0/frames_manifest.jsonl)",
        "grid": {"cols": COLS, "rows": ROWS, "frames": nframes,
                 "cells_per_frame": COLS * ROWS,
                 "total_cells": nframes * COLS * ROWS},
        "extractor": "forge src/extract_features.c (v0 binary, unchanged; "
                     "kernel headers included read-only)",
        "record_layout": "per cell: 16 x int16 features, uint8 fixed_sel, "
                         "uint8 fixed_tone_idx; frame-major, row-major",
        "fixed_sel_histogram_all_frames": sel_hist,
        "tone_verification_vs_stage0_native": {
            "method": "FNV-1a/32 over per-frame tone plane bytes "
                      f"({MODE} bytes) vs stage0 hashes.jsonl native.tone",
            "frames_checked": nframes,
            "mismatched_frames": bad},
    }
    (V1 / "features" / f"features_meta_{clip}.json").write_text(
        json.dumps(meta, indent=2) + "\n")
    summary[clip] = {"frames": nframes, "tone_mismatch_frames": bad,
                     "sel_hist": sel_hist}
    print(f"{clip}: frames={nframes} tone_mismatches={bad} sel={sel_hist}")

(V1 / "split_manifest_v1.json").write_text(json.dumps(split, indent=2) + "\n")
print("split manifest written")
print(json.dumps(summary, indent=2))

#!/usr/bin/env python3
"""Evaluate the frozen codebooks against the fixed tokenizer, held out.

All learned-table assignments use the frozen inference form
(argmax of inference_rows . x + inference_bias) in pure int64 — the exact
computation a kernel with a bias term would run.

Metrics:
  * usage histograms + dead codewords on train and test (split by frame)
  * identity check: biased dot-product argmax == direct nearest-centroid
    argmin, cell for cell, on sample frames
  * held-out quantisation error, partition form: each method's codewords
    are represented by the mean 16-dim feature vector of their TRAIN
    cells (post-hoc centroids, identical treatment for every method —
    generous to the fixed tokenizer, which never earned 16-dim centroids);
    error = mean Euclidean distance of TEST cells to their assigned
    codeword's centroid. The collapsed dot-product tables reduce to one
    live codeword and are included as the floor.
  * native-space error: mean distance from a test cell to the winning
    table row in the table's own feature space (for the fixed tokenizer:
    distance to the winning basis row in its 2 active dims — the basis
    rows are not centroids, so this number is labelled as such).
  * tone-plane renders for 3 frames (fixed vs micro vs full).
"""
import json
import numpy as np
from pathlib import Path

FORGE = Path.home() / "workspace/syzygy-forge"
COLS, ROWS, NFRAMES = 148, 80, 297
RAMP = " .:-=+*#%@"

dt = np.dtype([("feat", "<i2", (16,)), ("sel", "u1"), ("tone", "u1")])
data = np.fromfile(FORGE / "corpus/features.bin", dtype=dt).reshape(NFRAMES, ROWS, COLS)
FEAT = data["feat"].astype(np.int64)
SEL = data["sel"]
TONE = data["tone"]
manifest = json.loads((FORGE / "eval/split_manifest.json").read_text())
train_frames, test_frames = manifest["train_frames"], manifest["test_frames"]

micro = json.loads((FORGE / "tables/table_micro.json").read_text())
full = json.loads((FORGE / "tables/table_full.json").read_text())
micro_c = json.loads((FORGE / "tables/table_micro_dotproduct_collapsed.json").read_text())
full_c = json.loads((FORGE / "tables/table_full_dotproduct_collapsed.json").read_text())
FIXED_BASIS = np.array([[32, 64, 64, 32], [64, -32, -32, 64],
                        [16, 96, 96, 16], [-64, 12, 12, -64]], dtype=np.int64)


def assign_learned(fr, table):
    X = FEAT[fr].reshape(-1, 16)[:, table["feature_indices"]]
    R = np.array(table["inference_rows"], dtype=np.int64)
    B = np.array(table["inference_bias"], dtype=np.int64)
    return np.argmax(X @ R.T + B, axis=1)


def assign_collapsed(fr, table):
    X = FEAT[fr].reshape(-1, 16)[:, table["feature_indices"]]
    R = np.array(table["rows"], dtype=np.int64)
    return np.argmax(X @ R.T, axis=1)


def assign_fixed(fr):
    return SEL[fr].ravel().astype(np.int64)


METHODS = {
    "fixed": (assign_fixed, 4),
    "micro": (lambda fr: assign_learned(fr, micro), 16),
    "full": (lambda fr: assign_learned(fr, full), 64),
    "micro_dotproduct_collapsed": (lambda fr: assign_collapsed(fr, micro_c), 16),
    "full_dotproduct_collapsed": (lambda fr: assign_collapsed(fr, full_c), 64),
}

results = {}

# --- identity check --------------------------------------------------------
identity = {}
for name, table in (("micro", micro), ("full", full)):
    C = np.array(table["rows"], dtype=np.int64)
    bad = 0
    for fr in (test_frames[0], test_frames[len(test_frames) // 2], test_frames[-1]):
        X = FEAT[fr].reshape(-1, 16)[:, table["feature_indices"]]
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        a_direct = np.argmin(d, axis=1)
        a_biased = assign_learned(fr, table)
        bad += int(np.count_nonzero(a_direct != a_biased))
    identity[name] = {"frames_checked": 3, "mismatched_cells": bad}
print("identity check:", identity)

# --- usage + post-hoc centroids (train) ------------------------------------
usage = {}
centroids = {}
for name, (fn, n) in METHODS.items():
    u_tr = np.zeros(n, dtype=np.int64)
    sums = np.zeros((n, 16), dtype=np.float64)
    for fr in train_frames:
        a = fn(fr)
        u_tr += np.bincount(a, minlength=n)
        X = FEAT[fr].reshape(-1, 16).astype(np.float64)
        np.add.at(sums, a, X)
    cen = np.zeros((n, 16))
    live = u_tr > 0
    cen[live] = sums[live] / u_tr[live, None]
    # codewords dead in train keep a zero centroid; they are never assigned
    # in test for any method here (checked in the results), so it is unused.
    usage[name] = {"train": u_tr.tolist(),
                   "train_dead": [int(j) for j in range(n) if u_tr[j] == 0]}
    centroids[name] = cen
print("train usage done")

# --- test: usage, partition quantisation error, native error ---------------
for name, (fn, n) in METHODS.items():
    u_te = np.zeros(n, dtype=np.int64)
    dist_sum, dist2_sum, cnt = 0.0, 0.0, 0
    for fr in test_frames:
        a = fn(fr)
        u_te += np.bincount(a, minlength=n)
        X = FEAT[fr].reshape(-1, 16).astype(np.float64)
        d = np.sqrt(((X - centroids[name][a]) ** 2).sum(axis=1))
        dist_sum += float(d.sum())
        dist2_sum += float((d * d).sum())
        cnt += len(d)
    results[name] = {
        "test_usage": u_te.tolist(),
        "test_dead": [int(j) for j in range(n) if u_te[j] == 0],
        "heldout_mean_euclid_to_train_centroid": dist_sum / cnt,
        "heldout_mean_sq_euclid_to_train_centroid": dist2_sum / cnt,
        "test_cells": cnt,
    }
    print(f"{name}: dead(test)={results[name]['test_dead']} "
          f"mean_err={dist_sum / cnt:.3f}")

# --- native-space errors -----------------------------------------------------
def native_error(table, collapsed=False):
    tot, cnt = 0.0, 0
    R = np.array(table["rows"], dtype=np.int64)
    for fr in test_frames:
        X = FEAT[fr].reshape(-1, 16)[:, table["feature_indices"]]
        a = assign_collapsed(fr, table) if collapsed else assign_learned(fr, table)
        d = np.sqrt(((X - R[a]) ** 2).sum(axis=1))
        tot += float(d.sum()); cnt += len(d)
    return tot / cnt

native = {
    "micro": native_error(micro),
    "full": native_error(full),
    "micro_dotproduct_collapsed": native_error(micro_c, collapsed=True),
    "full_dotproduct_collapsed": native_error(full_c, collapsed=True),
}
# fixed: distance to winning basis row over its 2 active dims (luma, green)
tot, cnt = 0.0, 0
for fr in test_frames:
    X = FEAT[fr].reshape(-1, 16)
    a = assign_fixed(fr)
    xy = np.stack([X[:, 10], X[:, 7]], axis=1)  # cell_luma, green
    d = np.sqrt(((xy - FIXED_BASIS[a][:, :2]) ** 2).sum(axis=1))
    tot += float(d.sum()); cnt += len(d)
native["fixed_basis_rows_2dim"] = tot / cnt
print("native errors:", native)

out = {
    "identity_check_biased_vs_direct_nearest_centroid": identity,
    "usage": usage,
    "methods": results,
    "native_space_mean_euclid_to_winning_row": native,
    "notes": [
        "Partition quantisation error uses post-hoc train centroids for every "
        "method, including the fixed tokenizer — generous to fixed.",
        "The fixed tokenizer's partition is its sel (argmax over the 4x4 "
        "basis on features cell_luma, green); its tone char additionally "
        "depends on luma directly, so its partition understates its tone "
        "resolution — see the renders.",
        "Collapsed tables have exactly one live codeword; their partition "
        "error is the distance-to-global-mean floor.",
    ],
}
(FORGE / "eval/eval_results.json").write_text(json.dumps(out, indent=2) + "\n")

# --- tone renders ------------------------------------------------------------
def learned_plane(fr, table):
    a = assign_learned(fr, table).reshape(ROWS, COLS)
    rows = np.array(table["rows"], dtype=np.int64)
    order = np.argsort(rows[:, 0], kind="stable")  # by row mean_luma, dark->bright
    rank = np.empty(len(rows), dtype=np.int64)
    rank[order] = np.arange(len(rows))
    idx = np.minimum(rank[a] * 10 // len(rows), 9)
    return idx

for fr in (10, 120, 260):
    fixed_plane = TONE[fr].astype(np.int64)
    mp = learned_plane(fr, micro)
    fp = learned_plane(fr, full)
    lines = []
    for r in range(ROWS):
        def rowstr(plane):
            return "".join(RAMP[v] for v in plane[r])
        lines.append(rowstr(fixed_plane) + " | " + rowstr(mp) + " | " + rowstr(fp))
    (FORGE / f"render/frame_{fr:04d}_tone_compare.txt").write_text(
        f"# frame {fr}: fixed kernel tone | learned micro (N=16) | learned full (N=64)\n"
        + "\n".join(lines) + "\n")
    # PNG: three grayscale panels side by side, 3px per cell
    from PIL import Image
    panels = []
    for plane in (fixed_plane, mp, fp):
        img = Image.fromarray((plane * 28).astype(np.uint8), mode="L")
        panels.append(img.resize((COLS * 3, ROWS * 3), Image.NEAREST))
    combo = Image.new("L", (COLS * 3 * 3 + 12, ROWS * 3), 128)
    for i, p in enumerate(panels):
        combo.paste(p, (i * (COLS * 3 + 6), 0))
    combo.save(FORGE / f"render/frame_{fr:04d}_tone_compare.png")
print("renders written")

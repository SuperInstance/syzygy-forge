#!/usr/bin/env python3
"""Corpus v1 evaluation: does the learned codebook generalise?

Protocol is v0's eval.py, generalised from one corpus to (method x clip):

  * Assignments for learned tables use the frozen inference form
    (argmax of inference_rows . x + inference_bias) in pure int64.
    The fixed table's assignment is the extractor's sel byte (the
    kernel's own syz_ste_select4 decision).
  * Partition quantisation error (the headline metric): each method's
    codewords are represented by the mean 16-dim feature vector of the
    TRAIN cells assigned to them (post-hoc centroids, identical
    treatment for every method — generous to the fixed tokenizer);
    error = mean Euclidean distance of a clip's TEST cells (per-clip
    split: frame % 5 == 4) to their assigned codeword's centroid.
  * Fitting corpora: fixed and the shipped docks tables are fitted on
    the v0 docks train set (exactly v0's eval); pooled tables on the
    pooled stage-0 train set; specialist tables on their own clip's
    train set.
  * Native-space error: mean distance from a test cell to the winning
    table row in the table's own feature space (fixed: the winning
    basis row over its 2 active dims, as in v0 — indicative only).
  * Usage histograms over test frames and over all frames of each clip.

Anchor gate: this same machinery, run on the v0 docks corpus, must
reproduce v0's published held-out errors (fixed 151.18, micro 44.63,
full 29.87) before any v1 number is trusted.
"""
import json
import numpy as np
from pathlib import Path

FORGE = Path.home() / "workspace/syzygy-forge"
V1 = FORGE / "corpus-v1"
CELLS = 80 * 24
CLIPS = ["docks", "arena", "chiaroscuro", "lighthouses", "judge"]
FIXED_BASIS = np.array([[32, 64, 64, 32], [64, -32, -32, 64],
                        [16, 96, 96, 16], [-64, 12, 12, -64]],
                       dtype=np.int64)

dt = np.dtype([("feat", "<i2", (16,)), ("sel", "u1"), ("tone", "u1")])
split = json.loads((V1 / "split_manifest_v1.json").read_text())["clips"]
v0split = json.loads((FORGE / "eval/split_manifest.json").read_text())

# --- sources: name -> (feat int16 flat, sel flat, cells/frame, frames) -----
SRC = {}
for clip in CLIPS:
    d = np.fromfile(V1 / "features" / f"features_{clip}.bin", dtype=dt)
    nf = d.shape[0] // CELLS
    SRC[clip] = (d["feat"].reshape(-1, 16), d["sel"].ravel(), CELLS, nf)
d0 = np.fromfile(FORGE / "corpus/features.bin", dtype=dt)
SRC["v0"] = (d0["feat"].reshape(-1, 16), d0["sel"].ravel(), 148 * 80, 297)

TRAIN_FRAMES = {c: split[c]["train_frames"] for c in CLIPS}
TEST_FRAMES = {c: split[c]["test_frames"] for c in CLIPS}
TRAIN_FRAMES["v0"] = v0split["train_frames"]
TEST_FRAMES["v0"] = v0split["test_frames"]


def load_table(path):
    return json.loads(Path(path).read_text())


ship_micro = load_table(FORGE / "tables/table_micro.json")
ship_full = load_table(FORGE / "tables/table_full.json")
pool_micro = load_table(V1 / "tables/table_pooled_micro.json")
pool_full = load_table(V1 / "tables/table_pooled_full.json")

METHODS = {
    "fixed": {"kind": "fixed", "n": 4, "fit": ["v0"]},
    "docks_micro": {"kind": "learned", "table": ship_micro, "n": 16,
                    "fit": ["v0"]},
    "docks_full": {"kind": "learned", "table": ship_full, "n": 64,
                   "fit": ["v0"]},
    "pooled_micro": {"kind": "learned", "table": pool_micro, "n": 16,
                     "fit": CLIPS},
    "pooled_full": {"kind": "learned", "table": pool_full, "n": 64,
                    "fit": CLIPS},
}
for clip in CLIPS:
    for prof in ("micro", "full"):
        t = load_table(V1 / "tables" / f"table_spec_{clip}_{prof}.json")
        METHODS[f"spec_{clip}_{prof}"] = {
            "kind": "learned", "table": t, "n": t["n"], "fit": [clip],
            "eval_clips": [clip]}


def frame_X(src, f):
    feat, _, cpf, _ = SRC[src]
    return feat[f * cpf:(f + 1) * cpf].astype(np.int64)


def assign_frame(method, src, f):
    """Assignment of one frame's cells under a method (int64, exact)."""
    m = METHODS[method]
    feat, sel, cpf, _ = SRC[src]
    if m["kind"] == "fixed":
        return sel[f * cpf:(f + 1) * cpf].astype(np.int64)
    t = m["table"]
    X = frame_X(src, f)[:, t["feature_indices"]]
    R = np.array(t["inference_rows"], dtype=np.int64)
    B = np.array(t["inference_bias"], dtype=np.int64)
    return np.argmax(X @ R.T + B, axis=1)


# --- compute assignments once per (method, source) -------------------------
ASSIGN = {}
for name, m in METHODS.items():
    srcs = set(m["fit"]) | set(m.get("eval_clips", CLIPS + ["v0"]))
    if m["kind"] != "fixed" and "eval_clips" not in m:
        srcs = set(m["fit"]) | set(CLIPS) | {"v0"}
    for src in srcs:
        _, _, _, nf = SRC[src]
        cpf = SRC[src][2]
        out = np.empty(nf * cpf, dtype=np.int64)
        for f in range(nf):
            out[f * cpf:(f + 1) * cpf] = assign_frame(name, src, f)
        ASSIGN[(name, src)] = out
    print(f"assignments done: {name}")

# --- fit post-hoc centroids per method -------------------------------------
CENT, FIT_USAGE = {}, {}
for name, m in METHODS.items():
    n = m["n"]
    counts = np.zeros(n, dtype=np.int64)
    sums = np.zeros((n, 16), dtype=np.float64)
    for src in m["fit"]:
        _, _, cpf, _ = SRC[src]
        A = ASSIGN[(name, src)]
        for f in TRAIN_FRAMES[src]:
            a = A[f * cpf:(f + 1) * cpf]
            X = frame_X(src, f).astype(np.float64)
            counts += np.bincount(a, minlength=n)
            for dim in range(16):
                sums[:, dim] += np.bincount(a, weights=X[:, dim],
                                            minlength=n)
    cen = np.zeros((n, 16))
    live = counts > 0
    cen[live] = sums[live] / counts[live, None]
    CENT[name] = cen
    FIT_USAGE[name] = counts
    print(f"fitted: {name} (dead in fit: "
          f"{[int(j) for j in range(n) if counts[j] == 0]})")


def evaluate(method, src):
    """Partition + native error on src's test frames; usage everywhere."""
    m = METHODS[method]
    n = m["n"]
    _, _, cpf, nf = SRC[src]
    A = ASSIGN[(method, src)]
    cen = CENT[method]
    u_all = np.bincount(A, minlength=n)
    dist_sum = dist2_sum = nat_sum = 0.0
    cnt = 0
    u_test = np.zeros(n, dtype=np.int64)
    to_dead = 0
    fit_dead = set(int(j) for j in range(n) if FIT_USAGE[method][j] == 0)
    for f in TEST_FRAMES[src]:
        a = A[f * cpf:(f + 1) * cpf]
        u_test += np.bincount(a, minlength=n)
        if fit_dead:
            to_dead += int(np.isin(a, list(fit_dead)).sum())
        X = frame_X(src, f).astype(np.float64)
        d = np.sqrt(((X - cen[a]) ** 2).sum(axis=1))
        dist_sum += float(d.sum())
        dist2_sum += float((d * d).sum())
        cnt += len(d)
        if m["kind"] == "fixed":
            xy = np.stack([X[:, 10], X[:, 7]], axis=1)
            dn = np.sqrt(((xy - FIXED_BASIS[a][:, :2]) ** 2).sum(axis=1))
        else:
            t = m["table"]
            R = np.array(t["rows"], dtype=np.int64)
            Xs = X[:, t["feature_indices"]]
            dn = np.sqrt(((Xs - R[a]) ** 2).sum(axis=1))
        nat_sum += float(dn.sum())
    return {
        "partition_err": dist_sum / cnt,
        "partition_err_meansq": dist2_sum / cnt,
        "native_err": nat_sum / cnt,
        "test_cells": cnt,
        "usage_test": u_test.tolist(),
        "usage_all": u_all.tolist(),
        "dead_test": [int(j) for j in range(n) if u_test[j] == 0],
        "dead_all": [int(j) for j in range(n) if u_all[j] == 0],
        "max_share_test": float(u_test.max() / u_test.sum()),
        "max_share_all": float(u_all.max() / u_all.sum()),
        "test_assignments_to_fitdead_codewords": to_dead,
    }


# --- anchor gate on the v0 corpus ------------------------------------------
anchor = {}
for name, expect in (("fixed", 151.18), ("docks_micro", 44.63),
                     ("docks_full", 29.87)):
    r = evaluate(name, "v0")
    anchor[name] = {"partition_err": r["partition_err"], "expected": expect,
                    "abs_diff": abs(r["partition_err"] - expect)}
    print(f"anchor {name}: got {r['partition_err']:.4f} "
          f"expected {expect} (diff {anchor[name]['abs_diff']:.4f})")
assert all(v["abs_diff"] < 0.01 for v in anchor.values()), anchor
print("ANCHOR GATE PASSED")

# --- identity check for the pooled tables -----------------------------------
identity = {}
for name in ("pooled_micro", "pooled_full"):
    t = METHODS[name]["table"]
    C = np.array(t["rows"], dtype=np.int64)
    bad = 0
    checked = 0
    for src, f in (("docks", TEST_FRAMES["docks"][0]),
                   ("arena", TEST_FRAMES["arena"][len(TEST_FRAMES["arena"]) // 2]),
                   ("judge", TEST_FRAMES["judge"][-1])):
        X = frame_X(src, f)[:, t["feature_indices"]]
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        bad += int(np.count_nonzero(np.argmin(d, axis=1)
                                    != ASSIGN[(name, src)][f * SRC[src][2]:(f + 1) * SRC[src][2]]))
        checked += len(X)
    identity[name] = {"cells_checked": checked, "mismatched_cells": bad}
print("identity:", identity)

# --- main evaluation ---------------------------------------------------------
per_clip = {}
for clip in CLIPS:
    per_clip[clip] = {}
    for name, m in METHODS.items():
        if "eval_clips" in m and clip not in m["eval_clips"]:
            continue
        per_clip[clip][name] = evaluate(name, clip)
    print(f"evaluated clip: {clip}")

out = {
    "anchor_gate_v0_docks": anchor,
    "identity_check_pooled_biased_vs_direct": identity,
    "fit_usage": {k: v.tolist() for k, v in FIT_USAGE.items()},
    "per_clip": per_clip,
    "notes": [
        "Partition error: post-hoc centroids fitted on each method's own "
        "training corpus (v0 docks train for fixed + shipped tables; "
        "pooled stage-0 train for pooled tables; own-clip train for "
        "specialists), evaluated on each clip's held-out test frames "
        "(frame % 5 == 4). Identical treatment for every method.",
        "Native error is in each table's own feature space and is NOT "
        "comparable across profiles with different K (micro 8-dim, full "
        "16-dim; fixed is distance to a basis row in 2 dims, not a "
        "centroid).",
        "Specialist tables are evaluated only on their own clip "
        "(the per-domain ceiling arm).",
    ],
}
(V1 / "eval" / "eval_v1.json").write_text(json.dumps(out, indent=2) + "\n")

print("\n=== per-clip held-out partition error (fixed | docks | pooled | spec) ===")
for clip in CLIPS:
    r = per_clip[clip]
    def g(k):
        return f"{r[k]['partition_err']:7.2f}" if k in r else "      -"
    print(f"{clip:12s} fixed {g('fixed')}  micro {g('docks_micro')}/"
          f"{g('pooled_micro')}/{g(f'spec_{clip}_micro')}  "
          f"full {g('docks_full')}/{g('pooled_full')}/{g(f'spec_{clip}_full')}")
print("\n=== max codeword share of test cells (collapse check) ===")
for clip in CLIPS:
    r = per_clip[clip]
    parts = []
    for name in METHODS:
        if name in r:
            parts.append(f"{name}={r[name]['max_share_test']:.3f}"
                         f"(dead {len(r[name]['dead_test'])})")
    print(f"{clip:12s} " + "  ".join(parts))
print("wrote eval/eval_v1.json")

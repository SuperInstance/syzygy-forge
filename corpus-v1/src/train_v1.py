#!/usr/bin/env python3
"""Corpus v1 training: pooled (all five clips) + per-clip specialist tables.

Same algorithm as v0's train_kmeans.py euclid path, copied deliberately:
integer Lloyd k-means, seeded init from real corpus cells, centroids
rounded back to integers every step (np.rint), dead codewords kept and
recorded, float64 scores during iteration (exact for these ranges).
Differences from v0, all documented in the tables' provenance blocks:
  * corpus = the five stage-0 clips at 80x24 (features_<clip>.bin),
    split per clip by the v0 rule (frame % 5 == 4 -> test);
  * each training run draws its own 200,000-cell subsample with a fresh
    rng seeded 20261011 (v0 used one shared stream); for the pooled run
    the subsample is drawn per clip with quotas proportional to each
    clip's train-cell count (uniform over the pooled train set);
  * only the euclid (biased) rule is trained — the dot-product collapse
    is already established and is not re-run.
Table ids are FNV-1a/32 over centroid rows as little-endian int32,
row-major (the v0 identity rule); the shipped v0 ids are recomputed
first as an anchor check.
"""
import json
import struct
import numpy as np
from pathlib import Path

FORGE = Path.home() / "workspace/syzygy-forge"
V1 = FORGE / "corpus-v1"
COLS, ROWS = 80, 24
CELLS = COLS * ROWS
SEED = 20261011
SUBSAMPLE = 200_000
MAX_ITERS = 100
CLIPS = ["docks", "arena", "chiaroscuro", "lighthouses", "judge"]

dt = np.dtype([("feat", "<i2", (16,)), ("sel", "u1"), ("tone", "u1")])
split = json.loads((V1 / "split_manifest_v1.json").read_text())["clips"]

FEAT, SEL, TRAIN_ROWS, TEST_FRAMES = {}, {}, {}, {}
for clip in CLIPS:
    d = np.fromfile(V1 / "features" / f"features_{clip}.bin", dtype=dt)
    nf = d.shape[0] // CELLS
    d = d.reshape(nf, CELLS)
    FEAT[clip] = d["feat"].reshape(nf * CELLS, 16)
    SEL[clip] = d["sel"].reshape(nf * CELLS)
    rows = np.concatenate([np.arange(f * CELLS, (f + 1) * CELLS)
                           for f in split[clip]["train_frames"]])
    TRAIN_ROWS[clip] = rows
    TEST_FRAMES[clip] = split[clip]["test_frames"]


def fnv1a(data: bytes) -> int:
    h = 2166136261
    for b in data:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def table_id(rows):
    return "0x%08x" % fnv1a(b"".join(struct.pack("<i", int(v))
                                     for row in rows for v in row))


# --- anchor: shipped v0 table ids must reproduce ---------------------------
for name, expect in (("micro", "0x7ccb88b5"), ("full", "0x6f81750a")):
    t = json.loads((FORGE / "tables" / f"table_{name}.json").read_text())
    got = table_id(t["rows"])
    assert got == expect, (name, got, expect)
    print(f"anchor ok: shipped {name} id {got}")


# --- v0 Lloyd machinery, copied --------------------------------------------
def assign_euclid(X, P):
    Xf = X.astype(np.float64)
    d = (Xf * Xf).sum(axis=1)[:, None] - 2.0 * (Xf @ P.astype(np.float64).T) \
        + (P * P).sum(axis=1)[None, :]
    return np.argmin(d, axis=1)


def lloyd(X, P):
    log = []
    for it in range(MAX_ITERS):
        a = assign_euclid(X, P)
        counts = np.bincount(a, minlength=len(P))
        newP = P.copy()
        for j in range(len(P)):
            if counts[j] > 0:
                newP[j] = np.rint(X[a == j].mean(axis=0)).astype(np.int64)
        changed = int(np.count_nonzero(np.any(newP != P, axis=1)))
        log.append({"iter": it, "rows_changed": changed,
                    "dead": int(np.count_nonzero(counts == 0)),
                    "min_count": int(counts.min()),
                    "max_count": int(counts.max())})
        P = newP
        if changed == 0:
            break
    return P, log


def full_train_usage(clips, feat_idx, R, B, n):
    """Usage over the method's full training corpus, biased int64 scoring,
    streamed per frame (v0's full_train_usage, generalised to clips)."""
    usage = np.zeros(n, dtype=np.int64)
    for clip in clips:
        F = FEAT[clip]
        for f in split[clip]["train_frames"]:
            X = F[f * CELLS:(f + 1) * CELLS][:, feat_idx].astype(np.int64)
            usage += np.bincount(np.argmax(X @ R.T + B, axis=1),
                                 minlength=n)
    return usage


def subsample(clips, rng):
    """200k cells drawn uniformly from the pooled train rows of `clips`
    (per-clip quotas proportional to train-cell counts)."""
    counts = {c: len(TRAIN_ROWS[c]) for c in clips}
    total = sum(counts.values())
    quotas, frac = {}, {}
    for c in clips:
        q = SUBSAMPLE * counts[c] / total
        quotas[c] = int(q)
        frac[c] = q - quotas[c]
    for c in sorted(clips, key=lambda c: -frac[c]):
        if sum(quotas.values()) >= SUBSAMPLE:
            break
        quotas[c] += 1
    parts = []
    for c in clips:
        idx = rng.choice(len(TRAIN_ROWS[c]), size=quotas[c], replace=False)
        rows = TRAIN_ROWS[c][idx]
        parts.append(FEAT[c][rows].astype(np.int64))
    return np.concatenate(parts)


def train(clips, feat_idx, n, name, fname):
    rng = np.random.default_rng(SEED)
    sub_all = subsample(clips, rng)
    X = sub_all[:, feat_idx]
    init_idx = rng.choice(len(X), size=n, replace=False)
    P, log = lloyd(X, X[init_idx].copy())
    inf_rows, inf_bias = 2 * P, -(P * P).sum(axis=1)
    usage = full_train_usage(clips, feat_idx, inf_rows, inf_bias, n)
    table = {
        "profile": name, "n": n, "k": len(feat_idx),
        "feature_indices": list(feat_idx), "rows": P.tolist(),
        "assignment_rule": "nearest centroid; equivalently dot-product "
                           "argmax of inference_rows with inference_bias "
                           "(exact identity)",
        "inference_rows": inf_rows.tolist(),
        "inference_bias": inf_bias.tolist(),
        "table_id_fnv1a": table_id(P.tolist()),
        "training": {"seed": SEED, "subsample_cells": SUBSAMPLE,
                     "iterations_run": len(log),
                     "converged": log[-1]["rows_changed"] == 0,
                     "iteration_log": log},
        "train_usage_full_set": usage.tolist(),
        "train_dead_codewords": [int(j) for j in range(n) if usage[j] == 0],
        "corpus_v1": {
            "trained_on_clips": clips,
            "raster": "stage-0 160x96 px -> 80x24 cells",
            "train_cells_total": int(sum(len(TRAIN_ROWS[c]) for c in clips)),
            "split_rule": "per clip: frame index % 5 == 4 -> test"},
    }
    (V1 / "tables" / fname).write_text(json.dumps(table, indent=2) + "\n")
    print(f"{fname}: id={table['table_id_fnv1a']} iters={len(log)} "
          f"converged={table['training']['converged']} "
          f"dead={table['train_dead_codewords']}")
    return table


index = {"format": "SYZT v0 (forge prototype) — corpus v1 tables",
         "tables": []}

# pooled
for feat_idx, n, prof in ((list(range(8)), 16, "micro"),
                          (list(range(16)), 64, "full")):
    t = train(CLIPS, feat_idx, n, f"pooled-{prof}", f"table_pooled_{prof}.json")
    index["tables"].append({"name": f"pooled_{prof}", "source_json":
                            f"tables/table_pooled_{prof}.json",
                            "table_id_fnv1a": t["table_id_fnv1a"],
                            "n": n, "k": len(feat_idx),
                            "trained_on": CLIPS})

# specialists
for clip in CLIPS:
    for feat_idx, n, prof in ((list(range(8)), 16, "micro"),
                              (list(range(16)), 64, "full")):
        t = train([clip], feat_idx, n, f"spec-{clip}-{prof}",
                  f"table_spec_{clip}_{prof}.json")
        index["tables"].append({"name": f"spec_{clip}_{prof}", "source_json":
                                f"tables/table_spec_{clip}_{prof}.json",
                                "table_id_fnv1a": t["table_id_fnv1a"],
                                "n": n, "k": len(feat_idx),
                                "trained_on": [clip]})

(V1 / "tables" / "tables_index_v1.json").write_text(
    json.dumps(index, indent=2) + "\n")
print("tables_index_v1.json written")

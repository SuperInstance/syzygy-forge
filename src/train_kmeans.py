#!/usr/bin/env python3
"""Train integer k-means codebooks (T1) for the Syzygy learned tokenizer.

Two assignment rules are trained and frozen, because the difference is the
v0 finding:

  * "dotproduct" (kept in tables/table_*_dotproduct_collapsed.json):
    assignment by raw dot-product argmax — the kernel selector's literal
    rule. On this corpus it COLLAPSES: one codeword takes all 2.8M train
    cells in both profiles. Features live in a narrow brightness cone, so
    the largest-norm direction wins everything. Same disease as the fixed
    basis (codeword 2 takes 99.94% of cells; codeword 3 is dead).

  * "euclid" (the primary tables/table_*.json): classic integer Lloyd
    k-means — assignment to the NEAREST centroid. Nearest-centroid selection
    is exactly a dot-product argmax with a per-codeword bias:
        argmin_j ||x - c_j||^2  ==  argmax_j ( 2<c_j, x> - ||c_j||^2 )
    so the frozen table carries both forms: `rows` (the centroids) and
    `inference_rows` = 2*rows with `inference_bias` = -||row||^2, which the
    kernel's dot-product argmax can score unchanged once shard 0010 adds a
    bias term (or a constant-1 feature column). Training forward pass ==
    inference forward pass holds for the biased form; eval.py verifies the
    identity cell-for-cell on sample frames.

Rules common to both: centroids rounded back to integers EVERY step
(np.rint); seeded init from real corpus cells; dead codewords keep their
row and are recorded, never silently resurrected.

Scores are computed in float64 during iterations: all quantities are
integers with |dot| < 2^21 and squared distances < 2^23, exactly
representable in float64, so assignments are bit-identical to int64.
Final usage histograms use pure int64 over the full train set.
"""
import json
import numpy as np
from pathlib import Path

FORGE = Path.home() / "workspace/syzygy-forge"
COLS, ROWS, NFRAMES = 148, 80, 297
SEED = 20261011
SUBSAMPLE = 200_000
MAX_ITERS = 100

dt = np.dtype([("feat", "<i2", (16,)), ("sel", "u1"), ("tone", "u1")])
data = np.fromfile(FORGE / "corpus/features.bin", dtype=dt).reshape(NFRAMES, ROWS, COLS)
manifest = json.loads((FORGE / "eval/split_manifest.json").read_text())
train_frames = manifest["train_frames"]

FEAT = data["feat"].astype(np.int64)
train_all = FEAT[train_frames].reshape(-1, 16)
print(f"train cells: {len(train_all)}")

rng = np.random.default_rng(SEED)
sub_idx = rng.choice(len(train_all), size=SUBSAMPLE, replace=False)
sub = train_all[sub_idx]


def lloyd(X, P, assign_fn):
    log = []
    for it in range(MAX_ITERS):
        a = assign_fn(X, P)
        counts = np.bincount(a, minlength=len(P))
        newP = P.copy()
        for j in range(len(P)):
            if counts[j] > 0:
                newP[j] = np.rint(X[a == j].mean(axis=0)).astype(np.int64)
        changed = int(np.count_nonzero(np.any(newP != P, axis=1)))
        log.append({"iter": it, "rows_changed": changed,
                    "dead": int(np.count_nonzero(counts == 0)),
                    "min_count": int(counts.min()), "max_count": int(counts.max())})
        P = newP
        if changed == 0:
            break
    return P, log


def assign_dot(X, P):
    return np.argmax(X.astype(np.float64) @ P.astype(np.float64).T, axis=1)


def assign_euclid(X, P):
    Xf = X.astype(np.float64)
    d = (Xf * Xf).sum(axis=1)[:, None] - 2.0 * (Xf @ P.astype(np.float64).T) \
        + (P * P).sum(axis=1)[None, :]
    return np.argmin(d, axis=1)


def full_train_usage(feat_idx, score_fn):
    n = score_fn.n
    usage = np.zeros(n, dtype=np.int64)
    for fr in train_frames:
        Xf = FEAT[fr].reshape(-1, 16)[:, feat_idx]
        usage += np.bincount(score_fn(Xf), minlength=n)
    return usage


def train(feat_idx, n, name, rule):
    X = sub[:, feat_idx]
    init_idx = rng.choice(len(X), size=n, replace=False)
    P0 = X[init_idx].copy()
    if rule == "dotproduct":
        P, log = lloyd(X, P0, assign_dot)

        def score(Xf, P=P):
            return np.argmax(Xf @ P.T, axis=1)
        score.n = n
        usage = full_train_usage(feat_idx, score)
        table = {
            "profile": name, "n": n, "k": len(feat_idx),
            "feature_indices": list(feat_idx), "rows": P.tolist(),
            "assignment_rule": "raw dot-product argmax (kernel selector's literal rule)",
            "outcome": "COLLAPSED — one codeword takes every train cell; kept as a negative result",
            "training": {"seed": SEED, "subsample_cells": SUBSAMPLE,
                         "iterations_run": len(log),
                         "converged": log[-1]["rows_changed"] == 0,
                         "iteration_log": log},
            "train_usage_full_set": usage.tolist(),
            "train_dead_codewords": [int(j) for j in range(n) if usage[j] == 0],
        }
        fname = f"table_{name}_dotproduct_collapsed.json"
    else:
        P, log = lloyd(X, P0, assign_euclid)
        inf_rows = (2 * P)
        inf_bias = -(P * P).sum(axis=1)

        def score(Xf, R=inf_rows, B=inf_bias):
            return np.argmax(Xf @ R.T + B, axis=1)
        score.n = n
        usage = full_train_usage(feat_idx, score)
        table = {
            "profile": name, "n": n, "k": len(feat_idx),
            "feature_indices": list(feat_idx), "rows": P.tolist(),
            "assignment_rule": "nearest centroid; equivalently dot-product argmax "
                               "of inference_rows with inference_bias (exact identity)",
            "inference_rows": inf_rows.tolist(),
            "inference_bias": inf_bias.tolist(),
            "training": {"seed": SEED, "subsample_cells": SUBSAMPLE,
                         "iterations_run": len(log),
                         "converged": log[-1]["rows_changed"] == 0,
                         "iteration_log": log},
            "train_usage_full_set": usage.tolist(),
            "train_dead_codewords": [int(j) for j in range(n) if usage[j] == 0],
        }
        fname = f"table_{name}.json"
    (FORGE / "tables" / fname).write_text(json.dumps(table, indent=2) + "\n")
    print(f"{name}/{rule}: iters={len(log)} converged={log[-1]['rows_changed'] == 0} "
          f"dead(full train)={table['train_dead_codewords']}")
    print(f"  train usage: {usage.tolist()}")
    return table


for rule in ("dotproduct", "euclid"):
    train(list(range(8)), 16, "micro", rule)
    train(list(range(16)), 64, "full", rule)

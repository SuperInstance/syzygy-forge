#!/usr/bin/env python3
"""Reader-side evaluation v3 — the DIGEST channel.

Computes, for the same 40 docks frames, a compact computed digest from the
SAME source the frozen truth was computed from (the NV12 Y plane in
syzygy-video/work/docks_base.nv12), using the SAME definitions as
readerside/gen_dataset.py (copied below). The digest is a perception
channel, not a leak: every number is a sum/mean/extremum of the Y plane
or of the Sobel field the truth definitions use — no answer, argmax or
comparison result is included.

Gates run before any reader sees a digest:
  A. Every region number in the digest equals the frozen computation
     recorded in questions.jsonl (which is copied byte-identical; the
     truth itself is never recomputed into a new file).
  B. Internal consistency: strip tallies add up to the half sums,
     quadrant sums and half edge counts they decompose.
  C. Oracle audit: a scripted reader that parses the digest TEXT (the
     exact string the models will see) and applies the frozen decision
     rules must reproduce all 200 frozen answers. A second oracle works
     from the strip tallies alone for Q1/Q2/Q4 (the v2 guide's procedure
     path); Q3 is excluded there by design — half totals do not
     determine a centroid — and the strip path's Q3 heuristic (larger
     half total) is reported informationally, not gated.

Writes digests.jsonl (frame -> digest text + structured fields),
digest_meta.json (gate results, sufficiency audit, size stats) and
digest_samples.md (three verbatim sample digests).
"""
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VIDEO = Path.home() / "workspace/syzygy-video"
NV12 = VIDEO / "work/docks_base.nv12"
W, H, NFRAMES = 296, 320, 297
EDGE_THRESH = 200

qrecs = [json.loads(l) for l in (HERE / "questions.jsonl").read_text().splitlines()]
truth = {q["frame"]: q["answers"] for q in qrecs}
fcomp = {q["frame"]: q["computation"] for q in qrecs}
frames = sorted(truth)

# ---------- source plane (identical loading to gen_dataset.py) ----------
nv12 = np.fromfile(NV12, dtype=np.uint8)
frame_bytes = W * H * 3 // 2
assert nv12.size == NFRAMES * frame_bytes
Y = nv12[: NFRAMES * W * H].reshape(NFRAMES, H, W).astype(np.int64)
xs = np.arange(W, dtype=np.int64)[None, :]
ys = np.arange(H, dtype=np.int64)[:, None]


def sobel_mag(y):
    """Verbatim from gen_dataset.py."""
    gx = (-y[:-2, :-2] - 2 * y[1:-1, :-2] - y[2:, :-2]
          + y[:-2, 2:] + 2 * y[1:-1, 2:] + y[2:, 2:])
    gy = (-y[:-2, :-2] - 2 * y[:-2, 1:-1] - y[:-2, 2:]
          + y[2:, :-2] + 2 * y[2:, 1:-1] + y[2:, 2:])
    return np.abs(gx) + np.abs(gy)


def compute_digest_fields(fr):
    """All digest quantities for one frame, gen_dataset definitions."""
    y = Y[fr]
    qs = {
        "TL": int(y[:160, :148].sum()), "TR": int(y[:160, 148:].sum()),
        "BL": int(y[160:, :148].sum()), "BR": int(y[160:, 148:].sum()),
    }
    total = int(y.sum())
    xbar_px = float((y * xs).sum() / y.sum())
    ybar_px = float((y * ys).sum() / y.sum())
    m = sobel_mag(y)
    edge = m > EDGE_THRESH
    ec_half = {"left": int(edge[:, :147].sum()), "right": int(edge[:, 147:].sum())}
    em_half = {"left": int(m[:, :147].sum()), "right": int(m[:, 147:].sum())}
    ec_band = {"top": int(edge[:107].sum()), "middle": int(edge[107:215].sum()),
               "bottom": int(edge[215:].sum())}
    em_band = {"top": int(m[:107].sum()), "middle": int(m[107:215].sum()),
               "bottom": int(m[215:].sum())}
    # strips: 8 strips of 10 cell rows = 40 pixel rows. Brightness over
    # pixel rows [40k, 40k+40) x pixel-col halves; edge counts over the
    # Sobel interior rows belonging to those original rows (interior row
    # i is original row i+1), halves as in the frozen counts.
    strips = []
    for k in range(8):
        b_l = int(y[40 * k:40 * k + 40, :148].sum())
        b_r = int(y[40 * k:40 * k + 40, 148:].sum())
        i_lo = max(0, 40 * k - 1)
        i_hi = min(H - 2, 40 * k + 39)
        e_l = int(edge[i_lo:i_hi, :147].sum())
        e_r = int(edge[i_lo:i_hi, 147:].sum())
        strips.append({"rows": f"{10 * k + 1}-{10 * k + 10}",
                       "brightness": {"left": b_l, "right": b_r},
                       "edges": {"left": e_l, "right": e_r}})
    return {
        "quadrant_sums": qs,
        "quadrant_means": {k: round(v / (148 * 160), 2) for k, v in qs.items()},
        "half_sums": {"left": qs["TL"] + qs["BL"], "right": qs["TR"] + qs["BR"]},
        "total": total, "mean_luma": round(total / (W * H), 2),
        "min_luma": int(y.min()), "max_luma": int(y.max()),
        "centroid_x_cells": round(xbar_px / 2, 2),
        "centroid_y_cells": round(ybar_px / 4, 2),
        "edge_counts_half": ec_half, "edge_mass_half": em_half,
        "edge_counts_band": ec_band, "edge_mass_band": em_band,
        "strips": strips,
    }


def render_digest(fr, f):
    qs, qm = f["quadrant_sums"], f["quadrant_means"]
    hs = f["half_sums"]
    ech, emh = f["edge_counts_half"], f["edge_mass_half"]
    ecb, emb = f["edge_counts_band"], f["edge_mass_band"]
    lines = [
        f"Frame digest — frame {fr}: exact values computed from the frame's "
        f"pixels (image 296x320 px; cell grid 148x80).",
        "",
        "BRIGHTNESS (pixel luma sums; every quadrant covers 148x160 px, "
        "so sums and means rank identically)",
        f"quadrant sums: TL={qs['TL']} TR={qs['TR']} BL={qs['BL']} BR={qs['BR']}",
        f"quadrant means: TL={qm['TL']:.2f} TR={qm['TR']:.2f} "
        f"BL={qm['BL']:.2f} BR={qm['BR']:.2f}",
        f"half sums: left={hs['left']} right={hs['right']}",
        f"frame luma: total={f['total']} mean={f['mean_luma']:.2f} "
        f"min={f['min_luma']} max={f['max_luma']}",
        f"brightness centroid: x={f['centroid_x_cells']:.2f} cells "
        f"(centre line x=74.00), y={f['centroid_y_cells']:.2f} cells "
        f"(centre y=40.00)",
        "",
        "EDGES (edge pixel = Sobel |gx|+|gy| > 200; count = number of edge "
        "pixels, mass = summed magnitude)",
        f"edge counts by half: left={ech['left']} right={ech['right']}",
        f"edge mass by half: left={emh['left']} right={emh['right']}",
        f"edge counts by band: top={ecb['top']} middle={ecb['middle']} "
        f"bottom={ecb['bottom']}",
        f"edge mass by band: top={emb['top']} middle={emb['middle']} "
        f"bottom={emb['bottom']}",
        "",
        "STRIP TALLIES (brightness sum and edge count per half, per 10-row "
        "strip of the cell grid; the strips add up to the half and quadrant "
        "totals above)",
    ]
    for s in f["strips"]:
        lines.append(
            f"strip rows {s['rows']}: brightness left={s['brightness']['left']} "
            f"right={s['brightness']['right']} | edges left={s['edges']['left']} "
            f"right={s['edges']['right']}")
    return "\n".join(lines)


# ---------- oracle: parse the digest TEXT, apply the frozen rules ----------
def _first_max(pairs):
    best_k, best_v = None, None
    for k, v in pairs:
        if best_v is None or v > best_v:
            best_k, best_v = k, v
    return best_k


def _first_min(pairs):
    best_k, best_v = None, None
    for k, v in pairs:
        if best_v is None or v < best_v:
            best_k, best_v = k, v
    return best_k


def oracle_from_text(text):
    g = re.search(r"quadrant sums: TL=(\d+) TR=(\d+) BL=(\d+) BR=(\d+)", text)
    quads = list(zip(["TL", "TR", "BL", "BR"], (int(v) for v in g.groups())))
    cx = float(re.search(r"brightness centroid: x=([\d.]+) cells", text).group(1))
    g = re.search(r"edge counts by half: left=(\d+) right=(\d+)", text)
    el, er = int(g.group(1)), int(g.group(2))
    g = re.search(r"edge counts by band: top=(\d+) middle=(\d+) bottom=(\d+)", text)
    bands = list(zip(["top", "middle", "bottom"], (int(v) for v in g.groups())))
    return {
        "Q1": _first_max(quads),
        "Q2": _first_min(quads),
        "Q3": "left" if cx < 74.0 else "right",
        "Q4": "left" if el >= er else "right",
        "Q5": _first_max(bands),
    }


def oracle_from_strips(text):
    """Q1/Q2/Q4 from the strip lines alone (the v2 procedure path), plus
    the half-total Q3 heuristic for information only."""
    strips = []
    for mm in re.finditer(
            r"strip rows \d+-\d+: brightness left=(\d+) right=(\d+) \| "
            r"edges left=(\d+) right=(\d+)", text):
        strips.append(tuple(int(v) for v in mm.groups()))
    assert len(strips) == 8
    quads = [("TL", sum(s[0] for s in strips[:4])),
             ("TR", sum(s[1] for s in strips[:4])),
             ("BL", sum(s[0] for s in strips[4:])),
             ("BR", sum(s[1] for s in strips[4:]))]
    el = sum(s[2] for s in strips)
    er = sum(s[3] for s in strips)
    hl = sum(s[0] for s in strips)
    hr = sum(s[1] for s in strips)
    return {
        "Q1": _first_max(quads),
        "Q2": _first_min(quads),
        "Q3_heuristic": "left" if hl > hr else "right",
        "Q4": "left" if el >= er else "right",
    }


# ---------- build + gates ----------
records, meta_frames = [], {}
gateA_fail, gateB_fail = [], []
oracle_ok = {q: 0 for q in ("Q1", "Q2", "Q3", "Q4", "Q5")}
oracle_strips_ok = {q: 0 for q in ("Q1", "Q2", "Q4")}
q3_heuristic_ok = 0

for fr in frames:
    f = compute_digest_fields(fr)
    c = fcomp[fr]
    # Gate A: digest numbers == frozen computation record
    checks = [
        f["quadrant_sums"] == c["quadrant_y_sums"],
        abs(f["centroid_x_cells"] - c["centroid_x_cells"]) < 1e-9,
        f["edge_counts_half"] == c["edge_counts_half"],
        f["edge_mass_half"] == c["edge_mass_half"],
        f["edge_counts_band"] == c["edge_counts_band"],
        f["edge_mass_band"] == c["edge_mass_band"],
    ]
    if not all(checks):
        gateA_fail.append(fr)
    # Gate B: strip consistency
    st = f["strips"]
    b_ok = (
        sum(s["brightness"]["left"] for s in st) == f["half_sums"]["left"]
        and sum(s["brightness"]["right"] for s in st) == f["half_sums"]["right"]
        and sum(s["edges"]["left"] for s in st) == f["edge_counts_half"]["left"]
        and sum(s["edges"]["right"] for s in st) == f["edge_counts_half"]["right"]
        and sum(s["brightness"]["left"] for s in st[:4]) == f["quadrant_sums"]["TL"]
        and sum(s["brightness"]["right"] for s in st[4:]) == f["quadrant_sums"]["BR"])
    if not b_ok:
        gateB_fail.append(fr)
    text = render_digest(fr, f)
    # Gate C: text oracle
    oa = oracle_from_text(text)
    for q in oracle_ok:
        oracle_ok[q] += int(oa[q] == truth[fr][q])
    os_ = oracle_from_strips(text)
    for q in oracle_strips_ok:
        oracle_strips_ok[q] += int(os_[q] == truth[fr][q])
    q3_heuristic_ok += int(os_["Q3_heuristic"] == truth[fr]["Q3"])
    records.append({"frame": fr, "digest": text, "fields": f})
    meta_frames[fr] = {"digest_chars": len(text),
                       "oracle_text": oa, "oracle_strips": os_}

n = len(frames)
print(f"Gate A (digest == frozen computation): "
      f"{'PASS all 40' if not gateA_fail else 'FAIL ' + str(gateA_fail)}")
print(f"Gate B (strip sums consistent): "
      f"{'PASS all 40' if not gateB_fail else 'FAIL ' + str(gateB_fail)}")
print(f"Gate C oracle (digest text): "
      + " ".join(f"{q} {oracle_ok[q]}/{n}" for q in oracle_ok)
      + f"  total {sum(oracle_ok.values())}/{5 * n}")
print(f"Strips oracle: "
      + " ".join(f"{q} {oracle_strips_ok[q]}/{n}" for q in oracle_strips_ok)
      + f"  Q3-by-half-totals heuristic {q3_heuristic_ok}/{n} (informational)")
assert not gateA_fail and not gateB_fail
assert sum(oracle_ok.values()) == 5 * n, "digest text oracle must be 200/200"

chars = [r["digest"] and len(r["digest"]) for r in records]
meta = {
    "gates": {"A_digest_equals_frozen_computation": not gateA_fail,
              "B_strip_consistency": not gateB_fail,
              "C_oracle_from_digest_text": {q: oracle_ok[q] for q in oracle_ok},
              "C_oracle_total": f"{sum(oracle_ok.values())}/{5 * n}",
              "strips_oracle": {q: oracle_strips_ok[q] for q in oracle_strips_ok},
              "q3_half_total_heuristic": f"{q3_heuristic_ok}/{n}"},
    "digest_chars": {"min": min(chars), "mean": sum(chars) / len(chars),
                     "max": max(chars)},
    "grid_chars": 148 * 80 + 79,  # 80 lines of 148 chars + 79 newlines
    "sufficiency_audit": {
        "Q1": "quadrant sums line (argmax) — also quadrant means, same order",
        "Q2": "quadrant sums line (argmin)",
        "Q3": "brightness centroid x line vs centre 74.00 (half sums do NOT "
              "determine this; strips-oracle Q3 heuristic reported separately)",
        "Q4": "edge counts by half line (>= rule matches frozen definition)",
        "Q5": "edge counts by band line (first-max in top/middle/bottom order)",
    },
    "per_frame": meta_frames,
}
(HERE / "digests.jsonl").write_text(
    "".join(json.dumps(r) + "\n" for r in records))
(HERE / "digest_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

sample_frames = [0, 23, 83]
samples = ["# Digest samples (verbatim, as embedded in the reader prompts)\n"]
for fr in sample_frames:
    rec = next(r for r in records if r["frame"] == fr)
    samples.append(f"\n## Frame {fr}\n\n```\n{rec['digest']}\n```\n")
(HERE / "digest_samples.md").write_text("\n".join(samples))
print(f"\nwrote digests.jsonl ({n} frames), digest_meta.json, digest_samples.md")
print(f"digest chars: min {min(chars)} mean {sum(chars)/len(chars):.0f} max {max(chars)}"
      f"  (grid text: {148*80+79} chars)")

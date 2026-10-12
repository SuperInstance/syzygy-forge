#!/usr/bin/env python3
"""Frozen-style scoring for reader-side evaluation v2. Pure functions
over reader_outputs.jsonl (first successful record per
reader/condition/frame wins — the stage-1 filter rule), three frame
conditions: fixed / learned (rank) / learned_cal (calibrated).

Blind is NOT re-run in v2: it is loaded from v1's reader_outputs.jsonl
and recomputed with this same code, and v1's fixed/learned/blind cells
are recomputed too and checked against v1's frozen scores.json — the
consistency gate proving this summariser is the v1 summariser.

Writes scores_v2.json and failures_v2.json; prints the tables that go
into RESULTS-V2.md."""
import json, os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(os.path.dirname(HERE), 'readerside')
QIDS = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
CONDS3 = ['fixed', 'learned', 'learned_cal']
READERS = ['minimax', 'glm']

qrecs = [json.loads(l) for l in open(os.path.join(HERE, 'questions.jsonl'))]
truth = {q['frame']: q['answers'] for q in qrecs}
comp = {q['frame']: q['computation'] for q in qrecs}
frames = sorted(truth)

const_base = {}
for qid in QIDS:
    dist = Counter(truth[f][qid] for f in frames)
    best, cnt = dist.most_common(1)[0]
    const_base[qid] = {'answer': best, 'accuracy': cnt / len(frames),
                       'distribution': dict(dist)}


def load_first_success(path, conditions):
    recs, n_fail = {}, 0
    for l in open(path):
        r = json.loads(l)
        if r.get('error') is not None or not r.get('raw_output'):
            n_fail += 1
            continue
        if r['condition'] not in conditions:
            continue
        key = (r['reader'], r['condition'], r['frame'])
        if key not in recs:
            recs[key] = r
    return recs, n_fail


def cell_stats(recs, rd, cond):
    per_q, tot_ok, tot, unparsed, parsed_ok, parsed_n = {}, 0, 0, 0, 0, 0
    for qid in QIDS:
        ok = n = unp = pok = pn = 0
        for f in frames:
            r = recs.get((rd, cond, f))
            if r is None:
                continue
            n += 1
            if r['parsed'][qid] is None:
                unp += 1
            else:
                pn += 1
                if r['correct'][qid]:
                    pok += 1
            if r['correct'][qid]:
                ok += 1
        per_q[qid] = {'acc': ok / n if n else None, 'n': n, 'unparsed': unp,
                      'parsed_only': pok / pn if pn else None, 'parsed_n': pn}
        tot_ok += ok; tot += n; unparsed += unp; parsed_ok += pok; parsed_n += pn
    return {'overall': tot_ok / tot if tot else None, 'n': tot,
            'unparsed': unparsed,
            'parsed_only': parsed_ok / parsed_n if parsed_n else None,
            'per_question': per_q}


def clear(f, qid):
    c = comp[f]
    if qid == 'Q1':
        return c['q1_margin_frac'] >= 0.02
    if qid == 'Q2':
        return c['q2_margin_frac'] >= 0.02
    if qid == 'Q3':
        return c['q3_margin_cells'] >= 2.0
    if qid == 'Q5':
        return c['q5_margin_frac'] >= 0.10
    return True


def clear_stats(recs, rd, cond):
    tot_ok = tot = 0
    per_q = {}
    for qid in QIDS:
        fs = [f for f in frames if clear(f, qid)]
        n_ok = sum(1 for f in fs
                   if recs.get((rd, cond, f)) and recs[(rd, cond, f)]['correct'][qid])
        per_q[qid] = {'acc': n_ok / len(fs) if fs else None, 'n': len(fs)}
        tot_ok += n_ok; tot += len(fs)
    return {'overall': tot_ok / tot if tot else None, 'per_question': per_q}


# ---------- v1 recomputation (consistency gate) ----------
v1recs, v1_fail = load_first_success(os.path.join(V1, 'reader_outputs.jsonl'),
                                     ['fixed', 'learned', 'blind'])
v1_frozen = json.load(open(os.path.join(V1, 'scores.json')))
mismatch = []
for rd in READERS:
    for cond in ('fixed', 'learned', 'blind'):
        mine = cell_stats(v1recs, rd, cond)['overall']
        theirs = v1_frozen['stats'][rd][cond]['overall']
        if abs(mine - theirs) > 1e-9:
            mismatch.append((rd, cond, mine, theirs))
print('v1 recomputation vs v1 scores.json:',
      'IDENTICAL' if not mismatch else f'MISMATCH {mismatch}')
assert not mismatch

blind = {(rd, 'blind'): cell_stats(v1recs, rd, 'blind') for rd in READERS}

# ---------- v2 stats ----------
recs, n_fail = load_first_success(os.path.join(HERE, 'reader_outputs.jsonl'),
                                  CONDS3)
missing = [(rd, c, f) for rd in READERS for c in CONDS3 for f in frames
           if (rd, c, f) not in recs]
print(f'v2 records: {len(recs)} / {len(READERS) * len(CONDS3) * len(frames)} '
      f'(failed attempts retained: {n_fail}; missing: {len(missing)})')
if missing:
    print('MISSING:', missing[:24])

stats = {}
for rd in READERS:
    stats[rd] = {c: cell_stats(recs, rd, c) for c in CONDS3}
    stats[rd]['blind'] = blind[(rd, 'blind')]
    for c in CONDS3:
        stats[rd][c]['lift_vs_v1blind'] = stats[rd][c]['overall'] - stats[rd]['blind']['overall']
    stats[rd]['cal_minus_rank'] = stats[rd]['learned_cal']['overall'] - stats[rd]['learned']['overall']
    stats[rd]['rank_minus_fixed'] = stats[rd]['learned']['overall'] - stats[rd]['fixed']['overall']
    stats[rd]['cal_minus_fixed'] = stats[rd]['learned_cal']['overall'] - stats[rd]['fixed']['overall']

clear2 = {rd: {c: clear_stats(recs, rd, c) for c in CONDS3} for rd in READERS}

# ---------- Q3 deep dive ----------
q3 = {}
for rd in READERS:
    d = {}
    for c in CONDS3:
        fs = frames
        d[c] = sum(1 for f in fs if recs.get((rd, c, f)) and recs[(rd, c, f)]['correct']['Q3']) / len(fs)
    # head-to-head frame counts rank vs cal, and each vs fixed
    def h2h(a, b):
        aw = bw = both = neither = 0
        for f in frames:
            ra, rb = recs.get((rd, a, f)), recs.get((rd, b, f))
            if not ra or not rb:
                continue
            ca, cb = ra['correct']['Q3'], rb['correct']['Q3']
            if ca and cb: both += 1
            elif ca: aw += 1
            elif cb: bw += 1
            else: neither += 1
        return {f'{a}_only': aw, f'{b}_only': bw, 'both': both, 'neither': neither}
    d['rank_vs_cal'] = h2h('learned', 'learned_cal')
    d['fixed_vs_cal'] = h2h('fixed', 'learned_cal')
    d['fixed_vs_rank'] = h2h('fixed', 'learned')
    q3[rd] = d

# ---------- frame 83 ----------
f83 = {}
for rd in READERS:
    f83[rd] = {}
    for c in CONDS3:
        r = recs.get((rd, c, 83))
        f83[rd][c] = {'parsed': r['parsed'] if r else None,
                      'correct': r['correct'] if r else None,
                      'raw_excerpt': (r['raw_output'] or '')[:1200] if r else None}
    r = v1recs.get((rd, 'learned', 83))
    f83[rd]['v1_learned_rank'] = {'parsed': r['parsed'] if r else None,
                                  'correct': r['correct'] if r else None,
                                  'raw_excerpt': (r['raw_output'] or '')[:1200] if r else None}
    r = v1recs.get((rd, 'fixed', 83))
    f83[rd]['v1_fixed'] = {'parsed': r['parsed'] if r else None,
                           'correct': r['correct'] if r else None}
f83['truth'] = truth[83]
f83['computation'] = comp[83]

# ---------- failures digest ----------
failures = []
for rd in READERS:
    for cond in CONDS3:
        for f in frames:
            r = recs.get((rd, cond, f))
            if r is None:
                continue
            for qid in QIDS:
                if not r['correct'][qid]:
                    failures.append({
                        'reader': rd, 'condition': cond, 'frame': f, 'qid': qid,
                        'truth': truth[f][qid], 'answered': r['parsed'][qid],
                        'raw_excerpt': (r['raw_output'] or '')[:1500]})

out = {'stats': stats, 'clear_subset': clear2, 'q3_dive': q3,
       'frame83': f83, 'constant_baselines': const_base,
       'v1_reference': {rd: {c: v1_frozen['stats'][rd][c]['overall']
                              for c in ('fixed', 'learned', 'blind')}
                        for rd in READERS},
       'n_failed_attempts': n_fail,
       'missing': [list(m) for m in missing]}
with open(os.path.join(HERE, 'scores_v2.json'), 'w') as fh:
    json.dump(out, fh, indent=2)
with open(os.path.join(HERE, 'failures_v2.json'), 'w') as fh:
    json.dump(failures, fh, indent=2)

print('\n=== v2 headline (fresh samples; blind = v1 reference) ===')
print(f"{'reader':9s} {'fixed':>7s} {'rank':>7s} {'cal':>7s} {'blind':>7s} "
      f"{'cal-rank':>9s} {'cal-fixed':>10s}")
for rd in READERS:
    s = stats[rd]
    print(f"{rd:9s} {s['fixed']['overall']:.3f}  {s['learned']['overall']:.3f}  "
          f"{s['learned_cal']['overall']:.3f}  {s['blind']['overall']:.3f}  "
          f"{s['cal_minus_rank']:+.3f}   {s['cal_minus_fixed']:+.3f}")
print('\n=== v1 frozen reference ===')
for rd in READERS:
    v = out['v1_reference'][rd]
    print(f"{rd:9s} fixed {v['fixed']:.3f}  learned {v['learned']:.3f}  blind {v['blind']:.3f}")
print('\n=== per-question (fixed | rank | cal) ===')
for rd in READERS:
    for qid in QIDS:
        s = stats[rd]
        print(f"{rd:9s} {qid}: {s['fixed']['per_question'][qid]['acc']:.3f} | "
              f"{s['learned']['per_question'][qid]['acc']:.3f} | "
              f"{s['learned_cal']['per_question'][qid]['acc']:.3f}   "
              f"const={const_base[qid]['answer']}@{const_base[qid]['accuracy']:.2f} "
              f"unp F/R/C={s['fixed']['per_question'][qid]['unparsed']}/"
              f"{s['learned']['per_question'][qid]['unparsed']}/"
              f"{s['learned_cal']['per_question'][qid]['unparsed']}")
print('\n=== Q3 dive ===')
for rd in READERS:
    d = q3[rd]
    print(f"{rd}: fixed {d['fixed']:.3f} rank {d['learned']:.3f} cal {d['learned_cal']:.3f} "
          f"| rank-vs-cal {d['rank_vs_cal']} | fixed-vs-cal {d['fixed_vs_cal']}")
print('\n=== frame 83 (truth Q1=TR wide margin) ===')
for rd in READERS:
    for c in CONDS3 + ['v1_learned_rank', 'v1_fixed']:
        e = f83[rd][c]
        print(f"{rd:9s} {c:16s} Q1={e['parsed']['Q1'] if e['parsed'] else None} "
              f"Q3={e['parsed']['Q3'] if e['parsed'] else None} "
              f"correct={e['correct']}")
print('\n=== parsed-only overall ===')
for rd in READERS:
    s = stats[rd]
    print(f"{rd:9s} fixed {s['fixed']['parsed_only']:.3f} rank {s['learned']['parsed_only']:.3f} "
          f"cal {s['learned_cal']['parsed_only']:.3f} blind(v1) {s['blind']['parsed_only']:.3f}")
print('\n=== clear-margin subset ===')
for rd in READERS:
    for c in CONDS3:
        print(f"{rd:9s} {c:12s} {clear2[rd][c]['overall']:.3f}")
print(f'\nfailures recorded: {len(failures)}')

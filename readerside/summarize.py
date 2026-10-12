#!/usr/bin/env python3
"""Frozen-style scoring for the reader-side evaluation. Pure functions over
reader_outputs.jsonl (first successful record per reader/condition/frame
wins — the stage-1 filter rule). Writes scores.json, failures.json (every
wrong answer in the two frame conditions, with ground-truth computation
refs) and prints the headline tables."""
import json, os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
QIDS = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
CONDS = ['fixed', 'learned', 'blind']
READERS = ['minimax', 'glm']

qrecs = [json.loads(l) for l in open(os.path.join(HERE, 'questions.jsonl'))]
truth = {q['frame']: q['answers'] for q in qrecs}
comp = {q['frame']: q['computation'] for q in qrecs}
frames = sorted(truth)

# constant-answer baselines from the truth distribution
const_base = {}
for qid in QIDS:
    dist = Counter(truth[f][qid] for f in frames)
    best, cnt = dist.most_common(1)[0]
    const_base[qid] = {'answer': best, 'accuracy': cnt / len(frames),
                       'distribution': dict(dist)}

# first-success filter
recs = {}
n_fail_attempts = 0
for l in open(os.path.join(HERE, 'reader_outputs.jsonl')):
    r = json.loads(l)
    if r.get('error') is not None or not r.get('raw_output'):
        n_fail_attempts += 1
        continue
    key = (r['reader'], r['condition'], r['frame'])
    if key not in recs:
        recs[key] = r

missing = [(rd, c, f) for rd in READERS for c in CONDS for f in frames
           if (rd, c, f) not in recs]
print(f'records: {len(recs)} / {len(READERS) * len(CONDS) * len(frames)} '
      f'(failed attempts retained: {n_fail_attempts}; missing: {len(missing)})')
if missing:
    print('MISSING:', missing[:20])

stats = {}
for rd in READERS:
    stats[rd] = {}
    for cond in CONDS:
        per_q = {}
        tot_ok = tot = unparsed = 0
        for qid in QIDS:
            ok = n = unp = 0
            for f in frames:
                r = recs.get((rd, cond, f))
                if r is None:
                    continue
                n += 1
                if r['parsed'][qid] is None:
                    unp += 1
                if r['correct'][qid]:
                    ok += 1
            per_q[qid] = {'acc': ok / n if n else None, 'n': n, 'unparsed': unp}
            tot_ok += ok; tot += n; unparsed += unp
        stats[rd][cond] = {'overall': tot_ok / tot if tot else None,
                           'n': tot, 'unparsed': unparsed, 'per_question': per_q}
    for cond in CONDS:
        stats[rd][cond]['lift'] = (stats[rd][cond]['overall'] - stats[rd]['blind']['overall']
                                   if cond != 'blind' else 0.0)
    stats[rd]['learned_minus_fixed'] = stats[rd]['learned']['overall'] - stats[rd]['fixed']['overall']

# clear-margin subsets (sensitivity): Q1/Q2 quadrant-sum margin >= 2% of
# total; Q3 centroid margin >= 2 cells; Q5 band margin >= 10% of winner.
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
    return True  # Q4: no margin recorded per-side beyond counts; use all

clear_stats = {}
for rd in READERS:
    clear_stats[rd] = {}
    for cond in ('fixed', 'learned'):
        per_q = {}
        tot_ok = tot = 0
        for qid in QIDS:
            fs = [f for f in frames if clear(f, qid)]
            ok = sum(1 for f in fs if recs.get((rd, cond, f)) and recs[(rd, cond, f)]['correct'][qid])
            per_q[qid] = {'acc': ok / len(fs) if fs else None, 'n': len(fs)}
            tot_ok += ok; tot += len(fs)
        clear_stats[rd][cond] = {'overall': tot_ok / tot if tot else None, 'per_question': per_q}

# failures digest for the write-up
failures = []
for rd in READERS:
    for cond in ('fixed', 'learned'):
        for f in frames:
            r = recs.get((rd, cond, f))
            if r is None:
                continue
            for qid in QIDS:
                if not r['correct'][qid]:
                    failures.append({
                        'reader': rd, 'condition': cond, 'frame': f, 'qid': qid,
                        'truth': truth[f][qid], 'answered': r['parsed'][qid],
                        'raw_excerpt': (r['raw_output'] or '')[:1500],
                    })

out = {'stats': stats, 'clear_subset': clear_stats,
       'constant_baselines': const_base, 'n_failed_attempts': n_fail_attempts,
       'missing': [list(m) for m in missing]}
with open(os.path.join(HERE, 'scores.json'), 'w') as fh:
    json.dump(out, fh, indent=2)
with open(os.path.join(HERE, 'failures.json'), 'w') as fh:
    json.dump(failures, fh, indent=2)

print('\n=== headline (accuracy over 200 answers per cell) ===')
print(f"{'reader':9s} {'fixed':>7s} {'learned':>8s} {'blind':>7s} {'liftF':>7s} {'liftL':>7s} {'L-F':>7s}")
for rd in READERS:
    s = stats[rd]
    print(f"{rd:9s} {s['fixed']['overall']:.3f}  {s['learned']['overall']:.3f}  "
          f"{s['blind']['overall']:.3f}  {s['fixed']['lift']:+.3f}  {s['learned']['lift']:+.3f}  "
          f"{s['learned_minus_fixed']:+.3f}")
print('\n=== per-question (fixed | learned | blind) ===')
for rd in READERS:
    for qid in QIDS:
        s = stats[rd]
        print(f"{rd:9s} {qid}: {s['fixed']['per_question'][qid]['acc']:.3f} | "
              f"{s['learned']['per_question'][qid]['acc']:.3f} | "
              f"{s['blind']['per_question'][qid]['acc']:.3f}   "
              f"const={const_base[qid]['answer']}@{const_base[qid]['accuracy']:.2f} "
              f"unp F/L/B={s['fixed']['per_question'][qid]['unparsed']}/"
              f"{s['learned']['per_question'][qid]['unparsed']}/"
              f"{s['blind']['per_question'][qid]['unparsed']}")
print('\n=== clear-margin subset (fixed | learned) ===')
for rd in READERS:
    for cond in ('fixed', 'learned'):
        print(f"{rd:9s} {cond:8s} overall {clear_stats[rd][cond]['overall']:.3f}")
print(f'\nfailures recorded: {len(failures)}')

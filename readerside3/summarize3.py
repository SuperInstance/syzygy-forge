#!/usr/bin/env python3
"""Frozen-style scoring for reader-side evaluation v3 (digest channel).
Pure functions over reader_outputs.jsonl (first successful record per
reader/condition/frame wins — the stage-1 filter rule), conditions:
digest / learned_cal.

Blind is NOT re-run in v3: loaded from v1's reader_outputs.jsonl and
recomputed with this code; v1's cells are checked against v1's frozen
scores.json (the consistency gate, as in v2). v2's learned_cal cells are
loaded as a cross-batch reference for the same condition.

Writes scores_v3.json and failures_v3.json; prints the RESULTS-V3 tables.
"""
import json, os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(os.path.dirname(HERE), 'readerside')
V2 = os.path.join(os.path.dirname(HERE), 'readerside2')
QIDS = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
CONDS = ['digest', 'learned_cal']
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
const_composite = sum(const_base[q]['accuracy'] for q in QIDS) / 5


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


def h2h(recs, rd, a, b, qid=None):
    aw = bw = both = neither = 0
    for f in frames:
        ra, rb = recs.get((rd, a, f)), recs.get((rd, b, f))
        if not ra or not rb:
            continue
        if qid is None:
            ca = all(ra['correct'][q] for q in QIDS)
            cb = all(rb['correct'][q] for q in QIDS)
            # per-answer discordance is the meaningful unit; count slots
            for q in QIDS:
                x, y = ra['correct'][q], rb['correct'][q]
                if x and y: both += 1
                elif x: aw += 1
                elif y: bw += 1
                else: neither += 1
            continue
        ca, cb = ra['correct'][qid], rb['correct'][qid]
        if ca and cb: both += 1
        elif ca: aw += 1
        elif cb: bw += 1
        else: neither += 1
    return {f'{a}_only': aw, f'{b}_only': bw, 'both': both, 'neither': neither}


# ---------- v1 recomputation (consistency gate, as in v2) ----------
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

# v2 learned_cal reference (cross-batch, same condition, same prompts)
v2recs, v2_fail = load_first_success(os.path.join(V2, 'reader_outputs.jsonl'),
                                     ['learned_cal'])
v2_cal = {rd: cell_stats(v2recs, rd, 'learned_cal') for rd in READERS}

# ---------- v3 stats ----------
recs, n_fail = load_first_success(os.path.join(HERE, 'reader_outputs.jsonl'),
                                  CONDS)
missing = [(rd, c, f) for rd in READERS for c in CONDS for f in frames
           if (rd, c, f) not in recs]
print(f'v3 records: {len(recs)} / {len(READERS) * len(CONDS) * len(frames)} '
      f'(failed attempts retained: {n_fail}; missing: {len(missing)})')
if missing:
    print('MISSING:', missing[:24])

stats = {}
for rd in READERS:
    stats[rd] = {c: cell_stats(recs, rd, c) for c in CONDS}
    stats[rd]['blind_v1'] = blind[(rd, 'blind')]
    stats[rd]['digest_minus_cal'] = (stats[rd]['digest']['overall']
                                     - stats[rd]['learned_cal']['overall'])
    stats[rd]['digest_minus_ceiling'] = (stats[rd]['digest']['overall']
                                         - const_composite)
    stats[rd]['cal_v2_reference'] = v2_cal[rd]['overall']

clear3 = {rd: {c: clear_stats(recs, rd, c) for c in CONDS} for rd in READERS}
paired = {rd: {'overall': h2h(recs, rd, 'digest', 'learned_cal'),
               'per_question': {q: h2h(recs, rd, 'digest', 'learned_cal', q)
                                for q in QIDS}}
          for rd in READERS}

# prompt sizes + latencies from the records themselves
sizes = {}
for c in CONDS:
    ps = [len(r['prompt']) for (rd, cc, f), r in recs.items() if cc == c]
    sizes[c] = {'n': len(ps), 'mean_prompt_chars': sum(ps) / len(ps),
                'min_prompt_chars': min(ps), 'max_prompt_chars': max(ps)}
lat = {}
for rd in READERS:
    lat[rd] = {}
    for c in CONDS:
        ls = [r['latency_s'] for (rr, cc, f), r in recs.items()
              if rr == rd and cc == c]
        lat[rd][c] = {'mean_s': sum(ls) / len(ls), 'max_s': max(ls)} if ls else None

# count-vs-mass disagreement frames: does the digest reader follow counts?
disagree = {}
for qid, flag in (('Q4', 'q4_count_vs_mass_agree'), ('Q5', 'q5_count_vs_mass_agree')):
    fs = [f for f in frames if not comp[f][flag]]
    disagree[qid] = {
        'frames': fs,
        'digest_correct': {rd: sum(1 for f in fs
                                    if recs.get((rd, 'digest', f))
                                    and recs[(rd, 'digest', f)]['correct'][qid])
                           for rd in READERS},
        'cal_correct': {rd: sum(1 for f in fs
                                 if recs.get((rd, 'learned_cal', f))
                                 and recs[(rd, 'learned_cal', f)]['correct'][qid])
                        for rd in READERS},
    }

# frame 83 digest record (the v1/v2 named case)
f83 = {}
for rd in READERS:
    f83[rd] = {}
    for c in CONDS:
        r = recs.get((rd, c, 83))
        f83[rd][c] = {'parsed': r['parsed'] if r else None,
                      'correct': r['correct'] if r else None,
                      'raw_excerpt': (r['raw_output'] or '')[:1200] if r else None}
f83['truth'] = truth[83]
f83['computation'] = comp[83]

# ---------- failures digest ----------
failures = []
for rd in READERS:
    for cond in CONDS:
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

meta = json.load(open(os.path.join(HERE, 'digest_meta.json')))
out = {'stats': stats, 'clear_subset': clear3, 'paired_digest_vs_cal': paired,
       'constant_baselines': const_base, 'constant_composite': const_composite,
       'prompt_sizes': sizes, 'latencies': lat,
       'count_vs_mass_disagreement': disagree, 'frame83': f83,
       'oracle': meta['gates'], 'digest_chars': meta['digest_chars'],
       'v2_cal_reference': {rd: v2_cal[rd]['overall'] for rd in READERS},
       'v1_reference': {rd: {c: v1_frozen['stats'][rd][c]['overall']
                              for c in ('fixed', 'learned', 'blind')}
                        for rd in READERS},
       'n_failed_attempts': n_fail,
       'missing': [list(m) for m in missing]}
with open(os.path.join(HERE, 'scores_v3.json'), 'w') as fh:
    json.dump(out, fh, indent=2)
with open(os.path.join(HERE, 'failures_v3.json'), 'w') as fh:
    json.dump(failures, fh, indent=2)

print('\n=== v3 headline (strict; digest vs SAME-BATCH learned_cal) ===')
print(f"{'reader':9s} {'digest':>7s} {'cal':>7s} {'d-cal':>7s} {'blind1':>7s} "
      f"{'ceiling':>8s} {'cal@v2':>7s}")
for rd in READERS:
    s = stats[rd]
    print(f"{rd:9s} {s['digest']['overall']:.3f}  {s['learned_cal']['overall']:.3f}  "
          f"{s['digest_minus_cal']:+.3f}  {s['blind_v1']['overall']:.3f}  "
          f"{const_composite:.3f}   {s['cal_v2_reference']:.3f}")
print('\n=== per-question (digest | cal), const baseline ===')
for rd in READERS:
    for qid in QIDS:
        s = stats[rd]
        print(f"{rd:9s} {qid}: {s['digest']['per_question'][qid]['acc']:.3f} | "
              f"{s['learned_cal']['per_question'][qid]['acc']:.3f}   "
              f"const={const_base[qid]['answer']}@{const_base[qid]['accuracy']:.2f} "
              f"unp D/C={s['digest']['per_question'][qid]['unparsed']}/"
              f"{s['learned_cal']['per_question'][qid]['unparsed']}")
print('\n=== parsed-only overall ===')
for rd in READERS:
    s = stats[rd]
    print(f"{rd:9s} digest {s['digest']['parsed_only']:.3f} "
          f"cal {s['learned_cal']['parsed_only']:.3f} "
          f"blind(v1) {s['blind_v1']['parsed_only']:.3f} "
          f"(unparsed D/C = {s['digest']['unparsed']}/{s['learned_cal']['unparsed']} of 200)")
print('\n=== clear-margin subset ===')
for rd in READERS:
    for c in CONDS:
        print(f"{rd:9s} {c:12s} {clear3[rd][c]['overall']:.3f}")
print('\n=== paired head-to-head digest vs cal (answer slots of 200) ===')
for rd in READERS:
    print(f"{rd}: {paired[rd]['overall']}")
    for qid in QIDS:
        print(f"   {qid}: {paired[rd]['per_question'][qid]}")
print('\n=== prompt sizes (chars, from records) ===')
for c in CONDS:
    print(f"{c:12s} mean {sizes[c]['mean_prompt_chars']:.0f} "
          f"min {sizes[c]['min_prompt_chars']} max {sizes[c]['max_prompt_chars']}")
print(f"digest block chars: mean {meta['digest_chars']['mean']:.0f} "
      f"(grid block: {meta['grid_chars']})")
print('\n=== latencies (mean s) ===')
for rd in READERS:
    print(f"{rd:9s} digest {lat[rd]['digest']['mean_s']:.1f} "
          f"cal {lat[rd]['learned_cal']['mean_s']:.1f}")
print('\n=== count-vs-mass disagreement frames ===')
for qid in ('Q4', 'Q5'):
    d = disagree[qid]
    print(f"{qid}: frames {d['frames']} digest-correct {d['digest_correct']} "
          f"cal-correct {d['cal_correct']}")
print('\n=== frame 83 ===')
for rd in READERS:
    for c in CONDS:
        e = f83[rd][c]
        print(f"{rd:9s} {c:12s} parsed={e['parsed']} correct={e['correct']}")
print(f'\nfailures recorded: {len(failures)}')

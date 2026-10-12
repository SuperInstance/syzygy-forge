#!/usr/bin/env python3
"""Reader-side evaluation v3 runner: the DIGEST channel vs a same-batch
raw-plane control. Two fleet text models answer the same 5 frozen,
programmatically-grounded region questions about the same 40 docks frames
under 2 conditions:

  digest      — NO grid. The reader receives a compact computed digest of
                the frame (gen_digest.py; GUIDE-DIGEST.md). The digest is
                derived only from the source NV12 Y plane, with the same
                definitions as the frozen truth — a perception channel,
                audited by a scripted text-parsing oracle at 200/200.
  learned_cal — the v2 adopted mapping (learned micro table, calibrated
                render), re-run FRESH in this batch so digest-vs-plane is
                a same-batch comparison (v2's lesson: the batch is a
                variable). Prompt construction is byte-identical to v2's
                (gate-checked against v2's recorded prompts).

Parser, normalizers, scorer and reader callers are copied verbatim from
run_readers2.py; gate_check.py proves the frozen parts are identical and
that the learned_cal prompts reproduce v2's byte for byte. One batched
call per (reader, condition, frame); resumable; failed attempts recorded.
Temperature is the callers' default (0.7), as in v1 and v2.
"""
import argparse, json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
READERS = {
    'minimax': ('~/workspace/skills/minimax/bin/chat.py', 'MiniMax-Text-01'),
    'glm': ('~/workspace/syzygy-program/reading-experiment/glm_call.py',
            'glm-4.5-flash (thinking disabled)'),
}
GUIDE = open(os.path.join(HERE, 'GUIDE.md')).read().split('\n', 2)[2].strip()
GUIDE_DIGEST = open(os.path.join(HERE, 'GUIDE-DIGEST.md')).read().split('\n', 2)[2].strip()

QUESTIONS = [
    ("Q1", "Which quadrant of the frame has the greatest total brightness? Answer TL, TR, BL or BR (T/B = top/bottom half, L/R = left/right half)."),
    ("Q2", "Which quadrant of the frame has the least total brightness? Answer TL, TR, BL or BR."),
    ("Q3", "Considering the whole frame, on which side of the vertical centre line (between columns 74 and 75) does the brightness centroid — the balance point of all the brightness — lie? Answer left or right."),
    ("Q4", "Which half of the frame, left or right, contains more edge activity — more sharp brightness changes between neighbouring cells? Answer left or right."),
    ("Q5", "Which horizontal band of the frame — top (rows 1-27), middle (rows 28-54) or bottom (rows 55-80) — contains the most edge activity? Answer top, middle or bottom."),
]

ANSWER_TAIL = """Answer in this form: first your Work lines for each question, then the 5 answer lines:
Q1: <answer>
Q2: <answer>
Q3: <answer>
Q4: <answer>
Q5: <answer>
If you cannot determine an answer, give your best guess in the same form."""


def build_prompt(frame_text):
    parts = [GUIDE, '']
    if frame_text is not None:
        parts += ['Frame:', '```', frame_text, '```', '']
    parts.append('Questions about the frame:')
    for qid, text in QUESTIONS:
        parts.append(f"{qid}. {text}")
    parts += ['', ANSWER_TAIL]
    return '\n'.join(parts)


def build_prompt_digest(digest_text):
    parts = [GUIDE_DIGEST, '']
    parts += ['Frame digest:', '```', digest_text, '```', '']
    parts.append('Questions about the frame:')
    for qid, text in QUESTIONS:
        parts.append(f"{qid}. {text}")
    parts += ['', ANSWER_TAIL]
    return '\n'.join(parts)


# ---------- parsing / scoring (pure functions) ----------
def norm_quad(text):
    m = re.search(r'\b(TL|TR|BL|BR)\b', text.upper())
    if m:
        return m.group(1)
    t = ' ' + text.lower() + ' '
    pairs = [(('top', 'left'), 'TL'), (('upper', 'left'), 'TL'),
             (('north', 'west'), 'TL'), (('northwest',), 'TL'),
             (('top', 'right'), 'TR'), (('upper', 'right'), 'TR'),
             (('north', 'east'), 'TR'), (('northeast',), 'TR'),
             (('bottom', 'left'), 'BL'), (('lower', 'left'), 'BL'),
             (('south', 'west'), 'BL'), (('southwest',), 'BL'),
             (('bottom', 'right'), 'BR'), (('lower', 'right'), 'BR'),
             (('south', 'east'), 'BR'), (('southeast',), 'BR')]
    best, best_pos = None, None
    for words, tok in pairs:
        pos = min((t.find(' ' + w) for w in words), default=-1)
        if all((' ' + w) in t for w in words):
            p = max(t.find(' ' + w) for w in words)
            if best_pos is None or p < best_pos:
                best, best_pos = tok, p
    return best


def norm_side(text):
    m = re.search(r'\b(left|right)\b', text.lower())
    return m.group(1) if m else None


def norm_band(text):
    m = re.search(r'\b(top|middle|bottom)\b', text.lower())
    if m:
        return m.group(1)
    if re.search(r'\b(mid|center|centre)\b', text.lower()):
        return 'middle'
    return None


NORMALIZERS = {'Q1': norm_quad, 'Q2': norm_quad, 'Q3': norm_side,
               'Q4': norm_side, 'Q5': norm_band}


def parse_answers(raw):
    """Extract {qid: normalized answer or None}; same line discipline as
    the stage-1 frozen parser (first 'Qn:' line wins; 'Work Qn:' lines do
    not match the anchored pattern)."""
    segs = {}
    for line in raw.splitlines():
        m = re.match(r'\s*\**Q\s*([1-5])\s*[:\.\)]\**\s*(.*)', line, re.IGNORECASE)
        if m and m.group(1) not in segs:
            segs[m.group(1)] = m.group(2).strip()
    return {f'Q{n}': (NORMALIZERS[f'Q{n}'](segs[n]) if n in segs else None)
            for n in '12345'}


def call_reader(cli, prompt, timeout=300, attempts=5):
    """Retries with patient backoff (ZAI has hit account-wide 429s today).
    Empty stdout counts as failure, as in the stage-1 harness."""
    total_dt, last_err = 0.0, None
    backoffs = [15, 45, 120, 240]
    for attempt in range(attempts):
        if attempt:
            time.sleep(backoffs[min(attempt - 1, len(backoffs) - 1)])
        t0 = time.time()
        try:
            r = subprocess.run([os.path.expanduser(cli), prompt, '--max-tokens', '3000'],
                               capture_output=True, text=True, timeout=timeout)
            total_dt += time.time() - t0
            if r.returncode != 0:
                last_err = f'exit {r.returncode}: {r.stderr.strip()[:300]}'
                continue
            out = r.stdout.strip()
            if not out:
                last_err = 'empty output (exit 0)'
                continue
            return out, None, total_dt
        except subprocess.TimeoutExpired:
            total_dt += time.time() - t0
            last_err = f'timeout after {timeout}s'
    return None, last_err, total_dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--readers', default='minimax,glm')
    ap.add_argument('--conditions', default='digest,learned_cal')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', default=os.path.join(HERE, 'reader_outputs.jsonl'))
    args = ap.parse_args()

    qrecs = [json.loads(l) for l in open(os.path.join(HERE, 'questions.jsonl'))]
    truth = {q['frame']: q['answers'] for q in qrecs}
    renders = {'learned_cal': {q['frame']: open(os.path.join(
        HERE, 'renders', 'learned_cal', f"frame_{q['frame']:04d}.txt")).read().rstrip('\n')
        for q in qrecs}}
    digests = {r['frame']: r['digest'] for r in
               (json.loads(l) for l in open(os.path.join(HERE, 'digests.jsonl')))}

    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try:
                r = json.loads(l)
                if r.get('error') is None and r.get('raw_output'):
                    done.add((r['reader'], r['condition'], r['frame']))
            except Exception:
                pass

    jobs = []
    for q in qrecs:
        fr = q['frame']
        for reader in args.readers.split(','):
            cli, model = READERS[reader]
            for cond in args.conditions.split(','):
                if (reader, cond, fr) in done:
                    continue
                if cond == 'digest':
                    prompt = build_prompt_digest(digests[fr])
                else:
                    prompt = build_prompt(renders[cond][fr])
                jobs.append((reader, model, cli, cond, fr, prompt))
    print(f'{len(jobs)} calls to make ({len(done)} already done)', flush=True)

    import threading
    write_lock = threading.Lock()

    def do(job):
        reader, model, cli, cond, fr, prompt = job
        raw, err, dt = call_reader(cli, prompt)
        rec = {'reader': reader, 'model': model, 'condition': cond, 'frame': fr,
               'prompt': prompt, 'raw_output': raw, 'error': err,
               'latency_s': round(dt, 2)}
        if raw is not None:
            parsed = parse_answers(raw)
            rec['parsed'] = parsed
            rec['correct'] = {qid: (parsed[qid] is not None and parsed[qid] == truth[fr][qid])
                              for qid in parsed}
        with write_lock:
            with open(args.out, 'a') as f:
                f.write(json.dumps(rec) + '\n')
        status = 'ERR' if raw is None else f"{sum(rec.get('correct', {}).values())}/5"
        print(f'{reader:8s} {cond:8s} frame {fr:3d} -> {status} ({dt:.0f}s)', flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(do, j) for j in jobs]
        for _ in as_completed(futs):
            pass
    print('done')


if __name__ == '__main__':
    main()

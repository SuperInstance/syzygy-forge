#!/usr/bin/env python3
"""Pre-run gates for v3, mirroring v2's sanity-gate discipline:

  1. Frozen-code identity: parser / normalizers / caller / prompt builder
     / QUESTIONS / ANSWER_TAIL / GUIDE in run_readers3.py are identical to
     run_readers2.py (source-compared via inspect).
  2. Prompt reproduction: the learned_cal prompts v3 builds are
     byte-identical to the prompts recorded in v2's reader_outputs.jsonl
     for all 40 frames.
  3. Digest prompt hygiene: the digest prompt embeds the digest text
     verbatim and contains no grid (no 148-char frame line).
"""
import importlib.util
import inspect
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(os.path.dirname(HERE), 'readerside2')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


r2 = load('run_readers2', os.path.join(V2, 'run_readers2.py'))
r3 = load('run_readers3', os.path.join(HERE, 'run_readers3.py'))

ok = True
for fn in ('norm_quad', 'norm_side', 'norm_band', 'parse_answers',
           'call_reader', 'build_prompt'):
    same = inspect.getsource(getattr(r2, fn)) == inspect.getsource(getattr(r3, fn))
    print(f'frozen function {fn}: {"IDENTICAL" if same else "DIFFERS"}')
    ok &= same
for name in ('QUESTIONS', 'ANSWER_TAIL', 'GUIDE', 'READERS'):
    same = getattr(r2, name) == getattr(r3, name)
    print(f'frozen constant {name}: {"IDENTICAL" if same else "DIFFERS"}')
    ok &= same

# prompt reproduction vs v2 records
v2_prompts = {}
for l in open(os.path.join(V2, 'reader_outputs.jsonl')):
    r = json.loads(l)
    if r['condition'] == 'learned_cal' and r.get('error') is None and r.get('raw_output'):
        v2_prompts.setdefault(r['frame'], r['prompt'])
qrecs = [json.loads(l) for l in open(os.path.join(HERE, 'questions.jsonl'))]
mismatch = []
for q in qrecs:
    fr = q['frame']
    grid = open(os.path.join(HERE, 'renders', 'learned_cal',
                             f'frame_{fr:04d}.txt')).read().rstrip('\n')
    if r3.build_prompt(grid) != v2_prompts.get(fr):
        mismatch.append(fr)
print(f'learned_cal prompt reproduction vs v2: '
      f'{"40/40 byte-identical" if not mismatch else "MISMATCH " + str(mismatch)}')
ok &= not mismatch and len(v2_prompts) == 40

# digest prompt hygiene
digests = {r['frame']: r['digest'] for r in
           (json.loads(l) for l in open(os.path.join(HERE, 'digests.jsonl')))}
bad = []
for fr, d in digests.items():
    p = r3.build_prompt_digest(d)
    if d not in p or any(len(ln) == 148 for ln in p.splitlines()):
        bad.append(fr)
print(f'digest prompt hygiene: {"PASS all 40" if not bad else "FAIL " + str(bad)}')
ok &= not bad

sizes = {}
for fr in list(digests)[:1]:
    sizes['grid_prompt_chars'] = len(r3.build_prompt(open(os.path.join(
        HERE, 'renders', 'learned_cal', f'frame_{fr:04d}.txt')).read().rstrip('\n')))
    sizes['digest_prompt_chars'] = len(r3.build_prompt_digest(digests[fr]))
print('sample prompt sizes (frame 0):', sizes)
print('\nALL GATES PASS' if ok else '\nGATE FAILURE')
raise SystemExit(0 if ok else 1)

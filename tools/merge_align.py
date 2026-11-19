import json, re, unicodedata, difflib
A = json.load(open('align_out.json')); SECS = json.load(open('sections.json'))
def norm(w):
    w = unicodedata.normalize('NFKD', w).encode('ascii','ignore').decode().lower()
    return re.sub(r"[^a-z]", '', w)
def sim(a, b):
    if not a or not b: return 0
    return difflib.SequenceMatcher(None, a, b).ratio()
report = {}
for s in SECS:
    k = s['id']
    if k not in A: continue
    v = A[k]
    sw, own = [], []
    for i, l in enumerate(s['lines']):
        for w in l['text'].split():
            n = norm(w)
            if n: sw.append(n); own.append(i)
    # whisper words, split glued tokens like '42.6-apa'
    ww = []
    for st, en, w in v['whisper']:
        for p in re.split(r'[-\s]+', w):
            n = norm(p)
            if n: ww.append((st, en, n))
    n, m = len(sw), len(ww); G = -0.4
    D = [[0.0]*(m+1) for _ in range(n+1)]; P = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1): D[i][0] = D[i-1][0] + G; P[i][0] = 1
    for j in range(1, m+1): D[0][j] = D[0][j-1] + G; P[0][j] = 2
    for i in range(1, n+1):
        for j in range(1, m+1):
            sc = sim(sw[i-1], ww[j-1][2]); ms = sc*2-0.8 if sc >= 0.5 else -1.5
            best = (D[i-1][j-1]+ms, 0); 
            if D[i-1][j]+G > best[0]: best = (D[i-1][j]+G, 1)
            if D[i][j-1]+G > best[0]: best = (D[i][j-1]+G, 2)
            D[i][j], P[i][j] = best
    i, j, match = n, m, {}
    while i > 0 or j > 0:
        p = P[i][j]
        if i > 0 and j > 0 and p == 0:
            if sim(sw[i-1], ww[j-1][2]) >= 0.5: match[i-1] = j-1
            i -= 1; j -= 1
        elif i > 0 and (j == 0 or p == 1): i -= 1
        else: j -= 1
    out = []
    for li, l in enumerate(s['lines']):
        idx = [x for x in range(n) if own[x] == li]
        mt = [match[x] for x in idx if x in match]
        frac = len(mt)/max(1, len(idx))
        mm = v['lines'][li]
        wt = ww[mt[0]][0] if mt else None
        if frac < 0.34:
            t, src = None, 'missing'
        elif mm['score'] >= 0.5 and wt is not None and abs(mm['t'] - wt) <= 1.2:
            t, src = mm['t'], 'mms'
        else:
            # whisper start, shifted back to first script word if first words unmatched
            t, src = wt, 'whisper'
        out.append((t, src, round(frac, 2)))
    report[k] = out
json.dump(report, open('merge_report.json', 'w'))
for s in SECS:
    k = s['id']
    if k not in report: continue
    print('##', k)
    for l, (t, src, f) in zip(s['lines'], report[k]):
        if src != 'mms': print('  ', t, src, f, l['text'][:60])

# ---- final timings
final = json.loads(json.dumps(SECS))
for s in final:
    k = s['id']
    if k not in report: continue
    for i, (l, (t, src, f)) in enumerate(zip(s['lines'], report[k])):
        mm = A[k]['lines'][i]
        last = i == len(s['lines']) - 1
        if src == 'missing' and (mm['score'] >= 0.6 or (last and mm['score'] >= 0.3)):
            t = mm['t']
        l['t'] = t; l.pop('e', None)
        if t is None: l['skip'] = True
    s['estimated'] = False
# l1: the second recorded chorus belongs to Reff 2 (right after Verse 2), not the last Reff
L = next(s for s in final if s['id'] == 'l1')['lines']
L[18]['t'], L[19]['t'] = L[22]['t'], L[23]['t']; L[18].pop('skip', None); L[19].pop('skip', None)
L[22]['t'] = L[23]['t'] = None; L[22]['skip'] = L[23]['skip'] = True
for s in final:
    ts = [l['t'] for l in s['lines'] if l.get('t') is not None]
    assert ts == sorted(ts), s['id']
json.dump(final, open('sections_aligned.json', 'w'), ensure_ascii=False)
for s in final:
    sk = [l['text'][:45] for l in s['lines'] if l.get('skip')]
    if sk and s.get('audio'): print(s['id'], sk)

"""Forced-align script text to audio (runs on GitHub Actions)."""
import json, os, re, subprocess, unicodedata, sys
import torch, torchaudio

AUD = 'app/src/main/assets/www/audio/'
secs = json.load(open('tools/sections.json'))
os.makedirs('work', exist_ok=True)

def norm(w):
    w = unicodedata.normalize('NFKD', w).encode('ascii', 'ignore').decode().lower()
    return re.sub(r"[^a-z']", '', w).strip("'")

bundle = torchaudio.pipelines.MMS_FA
model = bundle.get_model(with_star=False).eval()
tok, aligner = bundle.get_tokenizer(), bundle.get_aligner()
SR = bundle.sample_rate

from faster_whisper import WhisperModel
wm = WhisperModel('medium', device='cpu', compute_type='int8')

out = {}
for s in secs:
    if not s.get('audio'): continue
    sid = s['id']; src = AUD + sid + '.m4a'
    wav = f'work/{sid}.wav'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', src, '-ac', '1', '-ar', '44100', wav], check=True)
    voc = wav
    if s['kind'] == 'song':
        subprocess.run([sys.executable, '-m', 'demucs', '--two-stems=vocals', '-n', 'htdemucs', '-o', 'work/sep', wav], check=True)
        voc = f'work/sep/htdemucs/{sid}/vocals.wav'
    w16 = f'work/{sid}_16k.wav'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', voc, '-ac', '1', '-ar', str(SR), w16], check=True)

    # whisper transcript (for review)
    segs, _ = wm.transcribe(w16, language='id', word_timestamps=True, beam_size=5)
    wh = [[round(w.start, 2), round(w.end, 2), w.word] for sg in segs for w in sg.words]

    # chunked CTC emissions
    x, _ = torchaudio.load(w16); x = x.mean(0, keepdim=True); N = x.shape[1]
    CH, CX, HOP = SR * 20, SR * 2, 320
    ems = []
    with torch.inference_mode():
        for st in range(0, N, CH):
            a, b = max(0, st - CX), min(N, st + CH + CX)
            em, _ = model(x[:, a:b])
            fs = (st - a) // HOP; fe = fs + (min(st + CH, N) - st) // HOP
            ems.append(em[:, fs:fe])
    emission = torch.cat(ems, 1)

    words, owner = [], []
    for i, l in enumerate(s['lines']):
        for w in l['text'].split():
            n = norm(w)
            if n: words.append(n); owner.append(i)
    spans = aligner(emission[0], tok(words))
    fr = HOP / SR
    wl = [[round(sp[0].start * fr, 2), round(sp[-1].end * fr, 2), words[k], round(sum(t.score for t in sp) / len(sp), 3), owner[k]] for k, sp in enumerate(spans)]
    lines = []
    for i in range(len(s['lines'])):
        ws = [w for w in wl if w[4] == i]
        lines.append({'t': ws[0][0], 'e': ws[-1][1], 'score': round(sum(w[3] for w in ws) / len(ws), 3)})
    out[sid] = {'lines': lines, 'words': wl, 'whisper': wh}
    print(sid, 'done', [l['t'] for l in lines][:8], flush=True)

json.dump(out, open('align_out.json', 'w'), ensure_ascii=False)

"""Baselines for the drift measure (bge cosine between a generated answer and its root thread's human answer):
floor = the same answer against the human answer of a random other thread in the same forum; reference = two human
answers of threads in the same duplicate family where available. Writes results/drift_floor.json.
  GEN_TAG=_az python drift_floor.py android,unix,webmasters,mathematica"""
import json, os, sys, numpy as np
from common import load
from prep_gen import G, TAG, load_gen, kb_ids, parse_ev, st
out = {}
rng = np.random.default_rng(0)
for sub in sys.argv[1].split(","):
    kb, kfam, ask, afam = load(sub); ids = kb_ids(sub); ans = json.load(open(os.path.join(G, f"{sub}_answers.json")))
    z = np.load(os.path.join(G, f"{sub}_entries{TAG}.npz")); keys = z["keys"].tolist(); depth = z["depth"]; root = z["root"]
    E = z["E_gen_bge"].astype(np.float32); d = z["drift_ans"]
    ok = np.array([bool(ans.get(ids[r])) for r in root]); sel = np.where((depth == 0) & ok)[0]
    sel = rng.choice(sel, min(2000, len(sel)), replace=False)
    roots = np.unique(root[ok]); emb = {}
    R = st("bge").encode([ans[ids[r]]["body"] for r in roots], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    for r, e in zip(roots, R): emb[int(r)] = e
    other = rng.choice(roots, len(sel))
    floor = float(np.mean([E[i] @ emb[int(o)] for i, o in zip(sel, other) if int(o) != int(root[i])]))
    out[sub] = dict(d0_mean=float(d[sel].mean()), floor=floor, n=int(len(sel)))
    print(sub, out[sub], flush=True)
json.dump(out, open("results/drift_floor.json", "w"), indent=1)

"""Version-3 caches (v1/v2 caches untouched).

  python prep_v3.py cepairs [FORUMS]
      Cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2, max_length 256, as in v2) score of every
      (ask, shown thread) pair needed by the Doc2Query-- style write-back filter in sim_v3.py.
      Pairs = union over retrievers of (ask, top-5 base candidates), the (ask, thread) pairs of the v2
      ask+thread caches, and any pairs in cache/v3/missing_ce_{sub}.json (written by sim_v3.py), so that
      repeated calls make the lookup exact.  -> cache/v3/{sub}_cepairs.npz (keys = ask*nkb + thread, sc)
"""
import os, sys, json, time
import numpy as np
from common import *
V2 = os.path.join(CACHE, "v2"); V3 = os.path.join(CACHE, "v3"); os.makedirs(V3, exist_ok=True)

def cepairs(subs):
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="mps", max_length=256)
    for sub in subs:
        kb, kfam, asks, afam = load(sub); nkb = len(kb); n = len(asks)
        keys = set()
        for r in RETRIEVERS:
            ci = np.load(os.path.join(CACHE, f"{sub}_{r}.npz"))["cand_idx"][:, :5]
            keys |= set((np.repeat(np.arange(n), 5).astype(np.int64) * nkb + ci.ravel()).tolist())
            p = os.path.join(V2, f"{sub}_{r}_askdoc.npz")
            if os.path.exists(p): keys |= set(np.load(p)["keys"].tolist())
        mf = os.path.join(V3, f"missing_ce_{sub}.json")
        if os.path.exists(mf): keys |= set(json.load(open(mf)))
        keys = np.array(sorted(keys), dtype=np.int64)
        out = os.path.join(V3, f"{sub}_cepairs.npz"); old = {}
        if os.path.exists(out):
            z = np.load(out); old = dict(zip(z["keys"].tolist(), z["sc"].tolist()))
        need = [k for k in keys.tolist() if k not in old]
        if not need and len(old) == len(keys): continue
        t0 = time.time()
        if need:
            sc = ce.predict([(asks[k // nkb], kb[k % nkb]) for k in need], batch_size=256, show_progress_bar=False)
            for k, s in zip(need, sc): old[k] = float(s)
        np.savez(out, keys=keys, sc=np.array([old[k] for k in keys.tolist()], np.float32))
        print(f"cepairs {sub} pairs={len(keys)} new={len(need)} {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    if sys.argv[1] == "cepairs": cepairs(sys.argv[2].split(",") if len(sys.argv) > 2 else FORUMS)

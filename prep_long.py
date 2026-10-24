"""Long-stream pseudo-forums for the horizon check: {sub}L = the forum's asks followed by one paraphrase of each
(cache/gen/{sub}_paraphrase.jsonl), same families, same knowledge base. Builds cache/{sub}L_{ret}.npz exactly as prep.py
does for real forums, checks that the original asks' candidate lists match the cached ones, writes the ask embeddings
for sim_gen and links the answers file. Seeds the generation and judge caches with the main-run answers for the original
asks (their prompts are identical: same ask text, same evidence).
  GEN_TAG=_az python prep_long.py android,unix,webmasters,mathematica
"""
import os, sys, json, shutil, time
import numpy as np
from common import load, CACHE, RETRIEVERS
import prep
from prep_gen import G, TAG, DENSE, st
for sub in sys.argv[1].split(","):
    L = sub + "L"; kb, kfam, asks, afam = load(L); n = len(asks) // 2; t0 = time.time()
    for r in RETRIEVERS:
        out = os.path.join(CACHE, f"{L}_{r}.npz")
        if os.path.exists(out): continue
        if r == "bm25": base, S = prep.bm25(kb, asks)
        else:
            m = st(r); kv = m.encode(kb, batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
            av = m.encode(asks, batch_size=256, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
            base = av @ kv.T; S = av @ av.T
            np.save(os.path.join(G, f"{L}_{r}_asks.npy"), av)
        ci, cs = prep.topk(base, prep.K); np.fill_diagonal(S, -np.inf)
        old = np.load(os.path.join(CACHE, f"{sub}_{r}.npz"))
        same = float(np.mean(old["cand_idx"][:, 0] == ci[:n, 0]))
        print(f"{L} {r}: original asks' top-1 identical for {100*same:.2f}%", flush=True)
        np.savez(out, cand_idx=ci, cand_sc=cs, S=S.astype(np.float32), kfam=kfam, afam=afam)
    for name in ("answers.json",):
        dst = os.path.join(G, f"{L}_{name}")
        if not os.path.exists(dst): shutil.copy(os.path.join(G, f"{sub}_{name}"), dst)
    for kind in ("gen", "judge"):
        src = os.path.join(G, f"{sub}_{kind}{TAG}.jsonl"); dst = os.path.join(G, f"{L}_{kind}{TAG}.jsonl")
        if os.path.exists(src) and not os.path.exists(dst): shutil.copy(src, dst)
    print(f"{L} ready in {time.time()-t0:.0f}s", flush=True)

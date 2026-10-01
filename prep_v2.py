"""Version-2 caches (v1 caches in cache/*.npz are left untouched).

  python prep_v2.py askdoc [bge,minilm,bm25]   scores of every ask against entries whose text is
        "ask text + shown thread text" for pairs (ask j, thread t). Pairs = (j, top-5 base candidates of j)
        plus any pairs listed in cache/v2/missing_{sub}_{ret}.json (written by sim_v2.py), so repeated
        calls make the lookup exact.  -> cache/v2/{sub}_{ret}_askdoc.npz (keys, SC)
  python prep_v2.py rerank FORUMS               cross-encoder (ms-marco-MiniLM-L-6-v2) scores over the
        bge first stage: ask x top-30 base candidates and ask x top-300 other asks (by bge).
        -> cache/v2/{sub}_bgece.npz
"""
import os, sys, json, time
import numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from common import *
V2 = os.path.join(CACHE, "v2"); os.makedirs(V2, exist_ok=True)
MODELS = {"bge": ("BAAI/bge-small-en-v1.5", 512), "minilm": ("sentence-transformers/all-MiniLM-L6-v2", 256)}
TOPT = 5
_ST = {}

def st(r):
    from sentence_transformers import SentenceTransformer
    if r not in _ST:
        _ST[r] = SentenceTransformer(MODELS[r][0], device="mps"); _ST[r].max_seq_length = MODELS[r][1]
    return _ST[r]

def pairs_for(sub, r, nkb):
    z = np.load(os.path.join(CACHE, f"{sub}_{r}.npz"))
    ci = z["cand_idx"][:, :TOPT]; n = ci.shape[0]
    keys = set((np.repeat(np.arange(n), TOPT).astype(np.int64) * nkb + ci.ravel()).tolist())
    mf = os.path.join(V2, f"missing_{sub}_{r}.json")
    if os.path.exists(mf): keys |= set(json.load(open(mf)))
    return np.array(sorted(keys), dtype=np.int64)

def bm25_pairs(kb, asks, keys, nkb, k1=1.2, b=0.75):
    cv = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w\w+\b", stop_words=list(ENGLISH_STOP_WORDS), dtype=np.float32)
    D = cv.fit_transform(kb).tocsr(); N = D.shape[0]
    df = np.asarray((D > 0).sum(0)).ravel(); idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
    avg = np.asarray(D.sum(1)).ravel().mean()
    A = cv.transform(asks).tocsr()
    j, t = keys // nkb, keys % nkb
    M = (A[j] + D[t]).tocsr()
    l = np.asarray(M.sum(1)).ravel(); rows = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
    tf = M.data; M.data = tf * (k1 + 1) / (tf + k1 * (1 - b + b * l[rows] / avg)) * idf[M.indices]
    Q = A.copy(); Q.data[:] = 1.0
    return (Q @ M.T).toarray().astype(np.float32)

def askdoc(rets):
    for sub in FORUMS:
        kb, kfam, asks, afam = load(sub); nkb = len(kb)
        for r in rets:
            out = os.path.join(V2, f"{sub}_{r}_askdoc.npz"); keys = pairs_for(sub, r, nkb)
            if os.path.exists(out) and np.array_equal(np.load(out)["keys"], keys): continue
            t0 = time.time(); j, t = keys // nkb, keys % nkb
            if r == "bm25":
                SC = bm25_pairs(kb, asks, keys, nkb)
            else:
                old = {}
                if os.path.exists(out):     # reuse embeddings of pairs already done
                    z = np.load(out); old = dict(zip(z["keys"].tolist(), z["E"]))
                need = [k for k in keys.tolist() if k not in old]
                txt = [asks[k // nkb] + "\n" + kb[k % nkb] for k in need]
                m = st(r)
                E_new = m.encode(txt, batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32) if txt else np.zeros((0, 384), np.float32)
                for k, e in zip(need, E_new): old[k] = e
                E = np.stack([old[k] for k in keys.tolist()])
                av = m.encode(asks, batch_size=256, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
                SC = av @ E.T
            # an ask never retrieves its own entry
            SC[j, np.arange(len(keys))] = -np.inf
            save = dict(keys=keys, SC=SC.astype(np.float32))
            if r != "bm25": save["E"] = E.astype(np.float16)
            np.savez(out, **save)
            print(f"askdoc {sub} {r} pairs={len(keys)} {time.time()-t0:.0f}s", flush=True)

def rerank(subs, KB_K=30, ASK_K=300):
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="mps", max_length=256)
    for sub in subs:
        out = os.path.join(V2, f"{sub}_bgece.npz")
        if os.path.exists(out): continue
        t0 = time.time(); kb, kfam, asks, afam = load(sub)
        z = np.load(os.path.join(CACHE, f"{sub}_bge.npz")); n = len(asks)
        bci = z["cand_idx"][:, :KB_K]; bcs = z["cand_sc"][:, :KB_K]
        S = z["S"]; aci = np.argsort(-S, 1)[:, :ASK_K].astype(np.int32); acs = np.take_along_axis(S, aci, 1)
        p1 = [(asks[i], kb[t]) for i in range(n) for t in bci[i]]
        p2 = [(asks[i], asks[j]) for i in range(n) for j in aci[i]]
        c1 = ce.predict(p1, batch_size=128, show_progress_bar=False).reshape(n, KB_K).astype(np.float32)
        c2 = ce.predict(p2, batch_size=512, show_progress_bar=False).reshape(n, ASK_K).astype(np.float32)
        np.savez(out, bci=bci, bcs=bcs, bce=c1, aci=aci, acs=acs, ace=c2, kfam=z["kfam"], afam=z["afam"])
        print(f"rerank {sub} n={n} {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    if sys.argv[1] == "askdoc": askdoc(sys.argv[2].split(",") if len(sys.argv) > 2 else RETRIEVERS)
    else: rerank(sys.argv[2].split(","))

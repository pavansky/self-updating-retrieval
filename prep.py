"""Embed / index each forum with each retriever and cache what the simulation needs:
  cand_idx, cand_sc : top-K base-KB candidates for every ask (K=200)
  S                 : ask-by-ask score matrix (score of ask j's self-written entry for query ask i)
Self-written entries hold the ask text, so their score is computed from the ask text (as in the pilot).
BM25 scores for self-written entries use the base-corpus IDF and average length (stated approximation)."""
import os, sys, time, re, json
import numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from common import *
K = 200
MODELS = {"bge": ("BAAI/bge-small-en-v1.5", 512), "minilm": ("sentence-transformers/all-MiniLM-L6-v2", 256)}

def topk(scores, k):
    idx = np.argpartition(-scores, k, axis=1)[:, :k]
    sc = np.take_along_axis(scores, idx, 1); o = np.argsort(-sc, 1)
    return np.take_along_axis(idx, o, 1).astype(np.int32), np.take_along_axis(sc, o, 1).astype(np.float32)

def bm25(kb, asks, k1=1.2, b=0.75):
    cv = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w\w+\b", stop_words=list(ENGLISH_STOP_WORDS), dtype=np.float32)
    D = cv.fit_transform(kb).tocsr(); N = D.shape[0]
    df = np.asarray((D > 0).sum(0)).ravel(); idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
    dl = np.asarray(D.sum(1)).ravel(); avg = dl.mean()
    def weight(M):
        M = M.tocsr().copy(); l = np.asarray(M.sum(1)).ravel()
        rows = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
        tf = M.data; M.data = tf * (k1 + 1) / (tf + k1 * (1 - b + b * l[rows] / avg)) * idf[M.indices]; return M
    W = weight(D); A = cv.transform(asks); WA = weight(A)
    Q = A.copy(); Q.data[:] = 1.0                       # query: each distinct term once
    base = (Q @ W.T).toarray().astype(np.float32)        # asks x kb
    S = (Q @ WA.T).toarray().astype(np.float32)          # asks x asks
    return base, S

if __name__ == "__main__":
    which = sys.argv[1:] or RETRIEVERS
    os.makedirs(CACHE, exist_ok=True); meta = {}
    st = {}
    for sub in FORUMS:
        kb, kfam, asks, afam = load(sub)
        for r in which:
            out = os.path.join(CACHE, f"{sub}_{r}.npz")
            if os.path.exists(out): continue
            t = time.time()
            if r == "bm25":
                base, S = bm25(kb, asks)
            else:
                from sentence_transformers import SentenceTransformer
                if r not in st:
                    st[r] = SentenceTransformer(MODELS[r][0], device="mps"); st[r].max_seq_length = MODELS[r][1]
                m = st[r]
                kv = m.encode(kb, batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
                av = m.encode(asks, batch_size=256, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
                base = av @ kv.T; S = av @ av.T
            ci, cs = topk(base, K)
            np.fill_diagonal(S, -np.inf)                    # an ask never retrieves its own entry (it is written after)
            np.savez(out, cand_idx=ci, cand_sc=cs, S=S.astype(np.float32), kfam=kfam, afam=afam)
            print(f"{sub} {r} kb={len(kb)} asks={len(asks)} {time.time()-t:.0f}s", flush=True)

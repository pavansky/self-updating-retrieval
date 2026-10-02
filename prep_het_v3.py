"""Version-3 heterogeneous corpus: real repeated Bing queries against long, non-question passages (MS MARCO).

  python prep_het_v3.py build      -> data_het/msmarcoh/{kb,asks}.jsonl + stats.json
  python prep_het_v3.py index      -> cache/msmarcoh_{bge,minilm,bm25}.npz   (same format as prep.py)
  then: prep_v2.py-style ask+thread caches via  python prep_het_v3.py askdoc  (and after sim runs, again until
  no pair is missing), and cross-encoder pairs via  python prep_het_v3.py cepairs.

Families. In the MS MARCO passage-ranking train qrels, a passage judged relevant to two or more distinct queries
defines a family: those queries are real, independently issued user queries with the same relevant passage. We keep
only queries whose single judged-relevant passage is that passage, drop exact-duplicate query strings (after
lower-casing and whitespace normalisation), keep families with >= 2 distinct queries, and sample N_FAM families.
The passage is the canonical document; every query of the family is an ask.
Knowledge base. Canonical passages, plus for each canonical passage its NEIGH nearest neighbours by passage id on
each side (MS MARCO passages were collected in per-query blocks of about ten, so neighbouring ids are mostly other
passages retrieved for the same original query: topical hard distractors; checked in stats.json), plus N_BG passages
drawn uniformly at random. Neighbours and background passages are unjudged for our asks.
Download limit: the full 3.4 GB collection could not be downloaded in the time available (the transfer slowed to
~0.25 MB/s), so we use the first PID_MAX passage ids (the collection is stored in id order; ids carry no topical
order beyond the per-query blocks). Families are restricted to canonical passages with id < PID_MAX - NEIGH and the
background is drawn from ids < PID_MAX. Raw files: data_het/raw/queries.jsonl, data_het/raw/corpus.jsonl.part
(https://huggingface.co/datasets/mteb/msmarco, partial download, complete lines only).
"""
import os, sys, json, csv, collections, re
import numpy as np
from common import *
SUB = "msmarcoh"; OUT = os.path.join(HERE, "data_het", SUB); RAW = os.path.join(HERE, "data_het", "raw")
N_FAM = 2500; NEIGH = 5; N_BG = 60000; PID_MAX = 4300000

def hf(fn):
    from huggingface_hub import hf_hub_download
    return hf_hub_download("mteb/msmarco", fn, repo_type="dataset")

def build():
    os.makedirs(OUT, exist_ok=True)
    q2d = collections.defaultdict(set); d2q = collections.defaultdict(set)
    for r in csv.DictReader(open(hf("qrels/train.tsv")), delimiter="\t"):
        if int(float(r["score"])) > 0: q2d[r["query-id"]].add(r["corpus-id"]); d2q[r["corpus-id"]].add(r["query-id"])
    cand = {d: sorted(q for q in qs if len(q2d[q]) == 1) for d, qs in d2q.items() if len(qs) >= 2}
    need_q = set(q for qs in cand.values() for q in qs); qtext = {}
    for l in open(os.path.join(RAW, "queries.jsonl")):
        d = json.loads(l)
        if d["_id"] in need_q: qtext[d["_id"]] = d["text"]
    norm = lambda t: re.sub(r"\s+", " ", t.lower()).strip(" ?")
    fams = {}
    for d, qs in sorted(cand.items(), key=lambda x: int(x[0])):
        seen, keep = set(), []
        for q in qs:
            k = norm(qtext[q])
            if k not in seen: seen.add(k); keep.append(q)
        if len(keep) >= 2 and int(d) < PID_MAX - NEIGH: fams[d] = keep
    rng = np.random.default_rng(0); allf = sorted(fams, key=int)
    pick = sorted(rng.choice(len(allf), N_FAM, replace=False)); fdocs = [allf[i] for i in pick]
    canon = {d: f for f, d in enumerate(fdocs)}
    neigh = set()
    for d in fdocs:
        for o in range(-NEIGH, NEIGH + 1):
            p = int(d) + o
            if o and 0 <= p < PID_MAX: neigh.add(str(p))
    bg = set(str(x) for x in rng.choice(PID_MAX, N_BG, replace=False))
    want = set(fdocs) | neigh | bg; text = {}
    for l in open(os.path.join(RAW, "corpus.jsonl.part")):
        if not l.endswith("\n"): break
        d = json.loads(l)
        if d["_id"] in want: text[d["_id"]] = ((d.get("title") or "") + "\n" + d["text"]).strip()[:MAX_CHARS]
    ids = sorted(want & set(text), key=int)
    with open(os.path.join(OUT, "kb.jsonl"), "w") as f:
        for d in ids: f.write(json.dumps(dict(id=d, text=text[d], fam=canon.get(d, -1), kind="canon" if d in canon else ("neigh" if d in neigh else "bg"))) + "\n")
    n_ask = 0
    with open(os.path.join(OUT, "asks.jsonl"), "w") as f:
        for d in fdocs:
            for q in fams[d]: f.write(json.dumps(dict(id=q, text=qtext[q], fam=canon[d])) + "\n"); n_ask += 1
    st = dict(families_available=len(fams), families=N_FAM, asks=n_ask, kb=len(ids), neighbours=len(neigh & set(text)), background=len(bg & set(text)),
              asks_per_family=collections.Counter(len(fams[d]) for d in fdocs), mean_ask_words=float(np.mean([len(qtext[q].split()) for d in fdocs for q in fams[d]])),
              mean_kb_words=float(np.mean([len(text[d].split()) for d in ids])))
    json.dump(st, open(os.path.join(OUT, "stats.json"), "w"), indent=1); print(st)

def index(rets):
    import prep
    kb, kfam, asks, afam = load(SUB); st = {}
    for r in rets:
        out = os.path.join(CACHE, f"{SUB}_{r}.npz")
        if os.path.exists(out): continue
        if r == "bm25": base, S = prep.bm25(kb, asks)
        else:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(prep.MODELS[r][0], device="mps"); m.max_seq_length = prep.MODELS[r][1]
            kv = m.encode(kb, batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
            av = m.encode(asks, batch_size=256, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
            base = av @ kv.T; S = av @ av.T
            if r == "bge":   # check that id-neighbours are topical distractors: similarity to their canonical passage
                kinds = [json.loads(l)["kind"] for l in open(os.path.join(OUT, "kb.jsonl"))]; ids = [int(json.loads(l)["id"]) for l in open(os.path.join(OUT, "kb.jsonl"))]
                pos = {d: i for i, d in enumerate(ids)}; cs = [i for i, k in enumerate(kinds) if k == "canon"]
                nb = [float(kv[i] @ kv[pos[ids[i] + o]]) for i in cs for o in (-1, 1) if ids[i] + o in pos and kinds[pos[ids[i] + o]] == "neigh"]
                rr = np.random.default_rng(1); bgi = [i for i, k in enumerate(kinds) if k == "bg"]
                rb = [float(kv[i] @ kv[bgi[j]]) for i, j in zip(cs, rr.integers(0, len(bgi), len(cs)))]
                s = json.load(open(os.path.join(OUT, "stats.json"))); s["bge_cos_canon_to_id_neighbour"] = float(np.mean(nb)); s["bge_cos_canon_to_random_bg"] = float(np.mean(rb))
                json.dump(s, open(os.path.join(OUT, "stats.json"), "w"), indent=1); print("neighbour check", np.mean(nb), np.mean(rb))
        ci, cs_ = prep.topk(base, prep.K); np.fill_diagonal(S, -np.inf)
        np.savez(out, cand_idx=ci, cand_sc=cs_, S=S.astype(np.float32), kfam=kfam, afam=afam); print(SUB, r, len(kb), len(asks), flush=True)

if __name__ == "__main__":
    a = sys.argv[1]; rets = sys.argv[2].split(",") if len(sys.argv) > 2 else RETRIEVERS
    if a == "build": build()
    elif a == "index": index(rets)
    elif a == "askdoc":
        import prep_v2; prep_v2.FORUMS = [SUB]; prep_v2.askdoc(rets)
    elif a == "cepairs":
        import prep_v3; prep_v3.RETRIEVERS = rets; prep_v3.cepairs([SUB])

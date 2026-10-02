"""Shared data loading. Family construction identical to the pilot (../pilot_real.py)."""
import os, csv, json, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data"); CACHE = os.path.join(HERE, "cache"); RES = os.path.join(HERE, "results")
FORUMS = ["android","english","gaming","gis","mathematica","physics","programmers","stats","tex","unix","webmasters","wordpress"]
RETRIEVERS = ["bge", "minilm", "bm25"]
MAX_CHARS = 1200

def load(sub):
    if sub.startswith("msmarco"):        # v3 heterogeneous corpus built by prep_het_v3.py
        kb = [json.loads(l) for l in open(os.path.join(HERE, "data_het", sub, "kb.jsonl"))]
        ak = [json.loads(l) for l in open(os.path.join(HERE, "data_het", sub, "asks.jsonl"))]
        return [d["text"] for d in kb], np.array([d["fam"] for d in kb]), [d["text"] for d in ak], np.array([d["fam"] for d in ak])
    corpus = {}
    for line in open(os.path.join(DATA, sub, "corpus.jsonl")):
        d = json.loads(line); corpus[d["_id"]] = (d.get("title", ""), d.get("text", ""))
    queries = {}
    for l in open(os.path.join(DATA, sub, "queries.jsonl")):
        d = json.loads(l); queries[d["_id"]] = d["text"]
    rel = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(DATA, sub, "qrels", "test.tsv")), delimiter="\t"):
        if int(r["score"]) > 0: rel[r["query-id"]].append(r["corpus-id"])
    asks_txt, asks_fam, held, canon = [], [], set(), {}
    for fam, (qid, docs) in enumerate(sorted(rel.items())):
        docs = sorted(docs, key=int)
        canon.setdefault(docs[0], fam)
        asks_txt.append(queries[qid]); asks_fam.append(fam)
        for d in docs[1:]:
            asks_txt.append(corpus[d][0]); asks_fam.append(fam); held.add(d)
    kb_txt, kb_fam = [], []
    for cid, (title, text) in corpus.items():
        if cid in held: continue
        kb_txt.append((title + "\n" + text)[:MAX_CHARS]); kb_fam.append(canon.get(cid, -1))
    return kb_txt, np.array(kb_fam), asks_txt, np.array(asks_fam)

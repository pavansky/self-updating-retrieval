"""Paper 2 ("generated answers as evidence") caches, all under cache/gen/. v1-v3 caches are untouched.

  python prep_gen.py answers FORUMS        accepted (else top-voted) answer of every base post that can be shown as
        top-1 evidence (cand_idx[:, 0] of every retriever, plus cache/gen/missing_posts_{sub}.json written by
        sim_gen.py), fetched from the public Stack Exchange API by post id.  -> cache/gen/{sub}_answers.json
        Keyless quota is 300 calls/day; set SE_KEY=<stackapps key> for 10,000/day. Resumable.
  python prep_gen.py asks FORUMS           ask embeddings (bge, minilm) -> cache/gen/{sub}_{ret}_asks.npy
  python prep_gen.py gen FORUMS            generate the system's answer for every key in cache/gen/missing_{sub}.json
        (written by sim_gen.py) with the local model (LM Studio, OpenAI-compatible), temperature 0.
        -> cache/gen/{sub}_gen.jsonl (one {key, answer, model} per line; append-only)
  python prep_gen.py embed FORUMS          embeddings (bge, minilm) and BM25 scores of every entry text variant for every
        generated key, plus drift of each generated answer to the root thread's accepted answer.
        -> cache/gen/{sub}_entries.npz

Generation key: "{j}|{ev}" where j is the ask index and ev is the evidence key: "t:{t}" for base thread t, or
"e:{src}|{ev_src}" for a stored entry written from ask src that was answered with evidence ev_src. The text of a stored
entry is the src ask plus the generated answer for key "{src}|{ev_src}". Root thread of an evidence key = the base
thread at the end of the chain; depth = number of "e:" levels.
"""
import os, sys, json, time, re, html, collections, random
import numpy as np
import urllib.request, urllib.parse
from common import *
G = os.path.join(CACHE, "gen"); os.makedirs(G, exist_ok=True)
MODELS = {"bge": ("BAAI/bge-small-en-v1.5", 512), "minilm": ("sentence-transformers/all-MiniLM-L6-v2", 256)}
DENSE = ["bge", "minilm"]
VARIANTS = ["gen", "askgen", "genthr"]
BACKEND = os.environ.get("GEN_BACKEND", "local")          # local (LM Studio) | azure (Azure AI Foundry, OpenAI deployment)
TAG = os.environ.get("GEN_TAG", "")                       # suffix of the generator's cache files ("" = local gemma, "_az" = azure)
GEN_MODEL = os.environ.get("GEN_MODEL", "google/gemma-4-26b-a4b-qat" if BACKEND == "local" else "gpt-5.4-mini")
GEN_URL = os.environ.get("GEN_URL", "http://localhost:1234/v1/chat/completions")
AZ_EP = os.environ.get("AZURE_AI_ENDPOINT", "")   # your Azure AI Foundry endpoint
AZ_KEY_FILE = os.path.expanduser(os.environ.get("AZURE_AI_KEY_FILE", "~/.config/azure_ai.key"))
ANS_CHARS = 1500; GEN_WORDS = 120
SYSTEM = ("You are a technical support assistant. Answer the user's question using only the evidence provided. "
          "If the evidence does not fully answer the question, say what it does cover and give the closest guidance it supports. "
          f"At most {GEN_WORDS} words. Plain text, no headings, no mention of the evidence or of these instructions.")
_ST = {}

def st(r):
    from sentence_transformers import SentenceTransformer
    if r not in _ST:
        _ST[r] = SentenceTransformer(MODELS[r][0], device="mps"); _ST[r].max_seq_length = MODELS[r][1]
    return _ST[r]

def kb_ids(sub):
    """corpus ids in the same order as common.load() builds the KB."""
    import csv
    corpus = [json.loads(l)["_id"] for l in open(os.path.join(DATA, sub, "corpus.jsonl"))]
    rel = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(DATA, sub, "qrels", "test.tsv")), delimiter="\t"):
        if int(r["score"]) > 0: rel[r["query-id"]].append(r["corpus-id"])
    held = set()
    for qid, docs in rel.items():
        for d in sorted(docs, key=int)[1:]: held.add(d)
    return [c for c in corpus if c not in held]

def shown_posts(sub):
    s = set()
    for r in RETRIEVERS:
        s |= set(np.load(os.path.join(CACHE, f"{sub}_{r}.npz"))["cand_idx"][:, 0].tolist())
    mf = os.path.join(G, f"missing_posts_{sub}.json")
    if os.path.exists(mf): s |= set(json.load(open(mf)))
    return sorted(s)

def strip_html(s):
    s = re.sub(r"<pre><code>(.*?)</code></pre>", lambda m: "\n" + m.group(1) + "\n", s, flags=re.S)
    s = re.sub(r"</(p|li|div|br|h\d)>", "\n", s); s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"\n{3,}", "\n\n", html.unescape(s)).strip()

def answers(subs):
    key = os.environ.get("SE_KEY", "")
    kf = os.path.expanduser(os.environ.get("SE_KEY_FILE", "~/.config/se_key.env"))
    if not key and os.path.exists(kf):
        key = [l.split("=", 1)[1].strip() for l in open(kf) if l.startswith("SE_KEY=")][0]
    for sub in subs:
        out = os.path.join(G, f"{sub}_answers.json")
        got = json.load(open(out)) if os.path.exists(out) else {}
        ids = kb_ids(sub); need = [ids[t] for t in shown_posts(sub) if ids[t] not in got]
        print(f"{sub}: {len(need)} posts to fetch, {len(got)} cached", flush=True)
        B = 100
        for b in range(0, len(need), B):
            chunk = need[b:b + B]; found = {}
            page = 1
            while True:
                q = dict(site=sub if sub != "programmers" else "softwareengineering", filter="withbody", sort="votes", pagesize=100, page=page)
                if key: q["key"] = key
                url = f"https://api.stackexchange.com/2.3/questions/{';'.join(chunk)}/answers?" + urllib.parse.urlencode(q)
                req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
                import gzip
                try:
                    with urllib.request.urlopen(req, timeout=60) as resp: d = json.loads(gzip.decompress(resp.read()))
                except urllib.error.HTTPError as e:
                    body = e.read()
                    try: d = json.loads(gzip.decompress(body))
                    except Exception: d = {"error_message": body[:200]}
                    print("HTTP", e.code, d.get("error_message"), flush=True)
                    if e.code in (400, 502, 503) and "throttle" in str(d.get("error_message", "")).lower():
                        time.sleep(60); continue
                    json.dump(got, open(out, "w")); return
                for a in d.get("items", []):
                    qid = str(a["question_id"]); cur = found.get(qid)
                    better = cur is None or (a["is_accepted"] and not cur["accepted"]) or (a["is_accepted"] == cur["accepted"] and a["score"] > cur["score"])
                    if better: found[qid] = dict(accepted=bool(a["is_accepted"]), score=int(a["score"]), body=strip_html(a["body"])[:ANS_CHARS])
                if d.get("backoff"): time.sleep(int(d["backoff"]) + 1)
                if not d.get("has_more"): break
                page += 1
                if d.get("quota_remaining", 1) < 2: break
            for qid in chunk: got[qid] = found.get(qid)    # None = no answers (or post gone)
            json.dump(got, open(out, "w"))
            print(f"  {sub} {b + len(chunk)}/{len(need)} quota_remaining={d.get('quota_remaining')}", flush=True)
            if d.get("quota_remaining", 1) < 3: print("quota exhausted; rerun tomorrow or set SE_KEY"); return
            time.sleep(0.1)

def asks(subs):
    for sub in subs:
        kb, kfam, ask, afam = load(sub)
        for r in DENSE:
            out = os.path.join(G, f"{sub}_{r}_asks.npy")
            if os.path.exists(out): continue
            np.save(out, st(r).encode(ask, batch_size=256, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32))
            print("asks", sub, r, flush=True)

# ---- generation -------------------------------------------------------------------------------------------------
def parse_ev(ev):
    """evidence key -> (root thread, depth, chain of src asks outermost first)."""
    chain = []
    while ev.startswith("e:"):
        src, ev = ev[2:].split("|", 1); chain.append(int(src))
    assert ev.startswith("t:"), ev
    return int(ev[2:]), len(chain), chain

def load_gen(sub):
    p = os.path.join(G, f"{sub}_gen{TAG}.jsonl"); g = {}
    if os.path.exists(p):
        for l in open(p):
            d = json.loads(l); g[d["key"]] = d["answer"]
    return g

def evidence_text(ev, kb, ask, ans, ids, gen):
    """what the generator sees for evidence key ev; None if a needed generated answer is missing."""
    if ev.startswith("t:"):
        t = int(ev[2:]); a = ans.get(ids[t])
        s = "Resolved thread.\n" + kb[t]
        if a: s += "\n\nAccepted answer:\n" + a["body"] if a["accepted"] else "\n\nTop answer:\n" + a["body"]
        else: s += "\n\n(The thread has no answer on record.)"
        return s
    src, ev_src = ev[2:].split("|", 1); k = f"{src}|{ev_src}"
    if k not in gen: return None
    return "Previously resolved case.\nQuestion: " + ask[int(src)] + "\nAnswer: " + gen[k]

def _local_call(msgs):
    body = {"model": "google/gemma-4-26b-a4b-qat", "temperature": 0, "max_tokens": 260, "reasoning_effort": "none", "seed": 0, "messages": msgs}
    req = urllib.request.Request(GEN_URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    for attempt in range(8):
        try: return json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"].strip()
        except Exception as e: print("local error", e, flush=True); time.sleep(5 * (attempt + 1))
    raise RuntimeError("local generator unavailable")

def call_llm(question, evidence):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Question: " + question + "\n\nEvidence:\n" + evidence}]
    if BACKEND == "azure":
        url = f"{AZ_EP}/openai/deployments/{GEN_MODEL}/chat/completions?api-version=2024-12-01-preview"
        body = {"messages": msgs, "max_completion_tokens": 400, "reasoning_effort": "minimal", "seed": 0}
        hdr = {"Content-Type": "application/json", "api-key": open(AZ_KEY_FILE).read().strip()}
    else:
        url = GEN_URL; hdr = {"Content-Type": "application/json"}
        body = {"model": GEN_MODEL, "temperature": 0, "max_tokens": 260, "reasoning_effort": "none", "seed": 0, "messages": msgs}
    req = urllib.request.Request(url, json.dumps(body).encode(), hdr)
    attempt = 0
    while attempt < 8:
        try:
            d = json.load(urllib.request.urlopen(req, timeout=600)); return d["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            body = e.read() if e.code != 429 else b""
            if e.code == 400 and b"content management policy" in body:      # Azure prompt filter: this key is generated locally
                print("content filter -> local model", flush=True); return _local_call(msgs) + "\x00local"
            if e.code == 429:            # rate limit: wait as told (capped), never give up
                time.sleep(min(float(e.headers.get("Retry-After", 0) or 0) or 5, 30) + random.random()); continue
            print("llm http", e.code, body[:200], flush=True); attempt += 1; time.sleep(3 * attempt)
        except Exception as e:
            print("llm error", e, flush=True); attempt += 1; time.sleep(5 * attempt)
    raise RuntimeError("generator unavailable")

def gen(subs, workers=int(os.environ.get("GEN_WORKERS", 4))):
    import concurrent.futures as cf
    for sub in subs:
        kb, kfam, ask, afam = load(sub); ids = kb_ids(sub)
        ans = json.load(open(os.path.join(G, f"{sub}_answers.json")))
        g = load_gen(sub); mf = os.path.join(G, f"missing{TAG}_{sub}.json")
        need = [k for k in (json.load(open(mf)) if os.path.exists(mf) else []) if k not in g]
        # dependency order: a key whose evidence is an entry needs that entry's answer first
        need.sort(key=lambda k: parse_ev(k.split("|", 1)[1])[1])
        print(f"{sub}: {len(need)} answers to generate, {len(g)} cached", flush=True)
        out = open(os.path.join(G, f"{sub}_gen{TAG}.jsonl"), "a"); t0 = time.time(); done = 0; deferred = 0
        by_depth = collections.defaultdict(list)
        for k in need: by_depth[parse_ev(k.split("|", 1)[1])[1]].append(k)
        for depth in sorted(by_depth):
            keys = by_depth[depth]
            def work(k):
                j, ev = k.split("|", 1); e = evidence_text(ev, kb, ask, ans, ids, g)
                return k, (call_llm(ask[int(j)], e) if e is not None else None)
            with cf.ThreadPoolExecutor(workers) as ex:
                for k, a in (f.result() for f in cf.as_completed([ex.submit(work, k) for k in keys])):
                    if a is None: deferred += 1; continue
                    model = f"{BACKEND}/{GEN_MODEL}"
                    if a.endswith("\x00local"): a = a[:-6]; model = "local/google/gemma-4-26b-a4b-qat (azure content filter)"
                    g[k] = a; out.write(json.dumps(dict(key=k, answer=a, model=model)) + "\n"); out.flush(); done += 1
                    if done % 200 == 0: print(f"  {sub} {done}/{len(need)} {time.time()-t0:.0f}s", flush=True)
        print(f"{sub}: generated {done}, deferred {deferred} (evidence not yet generated), {time.time()-t0:.0f}s", flush=True)

# ---- entry texts, embeddings, BM25, drift ---------------------------------------------------------------------
def entry_text(variant, k, kb, ask, g):
    j, ev = k.split("|", 1); a = g[k]
    if variant == "gen": return a
    if variant == "askgen": return ask[int(j)] + "\n" + a
    root, _, _ = parse_ev(ev); return a + "\n" + kb[root]        # genthr: answer + cited root thread text

def bm25_matrix(kb, asks_txt, entries, k1=1.2, b=0.75):
    """BM25 score of every ask against every entry text, with document statistics of the base KB (as prep_v2)."""
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    cv = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w\w+\b", stop_words=list(ENGLISH_STOP_WORDS), dtype=np.float32)
    D = cv.fit_transform(kb).tocsr(); N = D.shape[0]
    df = np.asarray((D > 0).sum(0)).ravel(); idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
    avg = np.asarray(D.sum(1)).ravel().mean()
    M = cv.transform(entries).tocsr(); l = np.asarray(M.sum(1)).ravel(); rows = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
    tf = M.data; M.data = tf * (k1 + 1) / (tf + k1 * (1 - b + b * l[rows] / avg)) * idf[M.indices]
    Q = cv.transform(asks_txt).tocsr(); Q.data[:] = 1.0
    return (Q @ M.T).toarray().astype(np.float16)

def embed(subs):
    for sub in subs:
        kb, kfam, ask, afam = load(sub); ids = kb_ids(sub); g = load_gen(sub)
        ans = json.load(open(os.path.join(G, f"{sub}_answers.json")))
        keys = sorted(g); out = os.path.join(G, f"{sub}_entries{TAG}.npz")
        if os.path.exists(out) and np.load(out)["keys"].tolist() == keys: continue
        t0 = time.time(); save = dict(keys=np.array(keys))
        root = np.array([parse_ev(k.split("|", 1)[1])[0] for k in keys]); depth = np.array([parse_ev(k.split("|", 1)[1])[1] for k in keys])
        save["root"] = root; save["depth"] = depth
        old = {}; oz = None
        if os.path.exists(out):      # incremental: reuse embeddings and drift of keys already present
            oz = np.load(out); old = {k: i for i, k in enumerate(oz["keys"].tolist())}
        new_i = [i for i, k in enumerate(keys) if k not in old]; old_i = [i for i, k in enumerate(keys) if k in old]
        old_rows = np.array([old[keys[i]] for i in old_i], int)
        def merged(name, enc_new, dim):
            M = np.zeros((len(keys), dim), np.float16)
            if old_i: M[old_i] = oz[name][old_rows]
            if new_i: M[new_i] = enc_new()
            return M
        for v in VARIANTS:
            txt = [entry_text(v, k, kb, ask, g) for k in keys]; tnew = [txt[i] for i in new_i]
            for r in DENSE:
                save[f"E_{v}_{r}"] = merged(f"E_{v}_{r}", lambda: st(r).encode(tnew, batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype(np.float16), 384)
            save[f"BM_{v}"] = bm25_matrix(kb, ask, txt)
        # drift: similarity of the generated answer to the root thread's accepted answer (bge)
        d = np.zeros(len(keys), np.float32)
        if old_i: d[old_i] = oz["drift_ans"][old_rows]
        if new_i:
            ref = [(ans.get(ids[root[i]]) or {}).get("body") or kb[root[i]] for i in new_i]
            ea = st("bge").encode([g[keys[i]] for i in new_i], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
            er = st("bge").encode(ref, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
            d[new_i] = (ea * er).sum(1)
        save["drift_ans"] = d
        print(f"embed {sub}: {len(new_i)} new keys, {len(old_i)} reused", flush=True)
        np.savez(out, **save); print(f"embed {sub} keys={len(keys)} {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    cmd = sys.argv[1]; subs = sys.argv[2].split(",") if len(sys.argv) > 2 else FORUMS
    {"answers": answers, "asks": asks, "gen": gen, "embed": embed}[cmd](subs)

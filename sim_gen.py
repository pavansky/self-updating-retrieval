"""Paper-2 simulation: the stored entry carries a generated answer. Same loop, acceptance model and random streams
as sim_v2.run (cond static/naive/reopen); with every entry inactive it is sim_v2 static line for line.

Entry e (index in creation order): src ask src[e], evidence key evk[e] (what src's answer was generated from), gen key
"{src}|{evk}", root thread thr[e] (the base thread at the end of the evidence chain; defines correctness exactly as in
paper 1), depth dep[e] = 1 + depth of the evidence. Entry text by variant (prep_gen.VARIANTS):
    gen     : the generated answer only
    askgen  : src ask + generated answer (the realistic self-written Q&A entry)
    genthr  : generated answer + the root thread's text (the "cite the source" countermeasure)
Entry score for ask i: dense = ask embedding . entry embedding (cache/gen/{sub}_entries.npz); bm25 = BM_v[i, entry].
An entry whose generated answer is not in the cache yet is written but never active; its key is recorded in
cache/gen/missing_{sub}.json (run prep_gen.py gen + embed, then rerun; iterate until missing = 0). Base posts with no
fetched answer are recorded in cache/gen/missing_posts_{sub}.json for prep_gen.py answers.
Conditions: static, naive, reopen (as v2), prov = naive with provenance down-weighting: every entry score is reduced by
    lam * IQR of the static top-1 scores (label-free, retriever-scale free), lam in PROV_LAM.
Diagnostics per round (beyond v2's): entries by depth, top-1 self evidence of depth >= 2, wrong answers attributed to
    depth-1 vs deeper entries, and drift (bge similarity of the served generated answer to the root thread's accepted
    answer) by evidence depth 0 / 1 / 2+.
"""
import os, sys, json, time, collections
import numpy as np
from multiprocessing import Pool
from common import *
import sim_v2
from sim_v2 import retrieve, user_pw, PW_GRID
from prep_gen import G, TAG, VARIANTS, DENSE, parse_ev, kb_ids
PROV_LAM = [0.5, 1.0]
_C = {}

def getg(sub, r):
    if (sub, r) in _C: return _C[(sub, r)]
    _C.clear()
    c = sim_v2.get(sub, r)
    c = dict(c)
    if r in DENSE: c["A"] = np.load(os.path.join(G, f"{sub}_{r}_asks.npy"))
    p = os.path.join(G, f"{sub}_entries{TAG}.npz")
    if os.path.exists(p):
        z = np.load(p); c["gkeys"] = {k: i for i, k in enumerate(z["keys"].tolist())}
        for v in VARIANTS:
            c[f"E_{v}"] = z[f"E_{v}_{r}"].astype(np.float32) if r in DENSE else z[f"BM_{v}"].astype(np.float32)
        c["drift"] = z["drift_ans"]
    else:
        c["gkeys"] = {}
    ap = os.path.join(G, f"{sub}_answers.json")
    c["has_ans"] = set(json.load(open(ap))) if os.path.exists(ap) else set()
    c["ids"] = kb_ids(sub)
    s1 = c["cand_sc"][:, 0]; c["iqr"] = float(np.quantile(s1, 0.75) - np.quantile(s1, 0.25))
    _C[(sub, r)] = c; return c

class GRet:
    """sim_v2.Ret interface over generated entries."""
    def __init__(self, c, r, variant, n, pen=0.0):
        self.c, self.r, self.v, self.pen = c, r, variant, pen
        self.E = np.zeros((n, c["A"].shape[1]), np.float32) if r in DENSE else None
        self.idx = np.zeros(n, int)          # column in BM for bm25
        self.missing = set()
    def base(self, i, base_on):
        c = self.c; ci, cs = c["cand_idx"][i], c["cand_sc"][i]; m = base_on[ci]; return ci[m], cs[m], None
    def selfsc(self, i, src, thr):
        ns = len(src)
        if self.r in DENSE: s = self.E[:ns] @ self.c["A"][i]
        elif f"E_{self.v}" in self.c: s = self.c[f"E_{self.v}"][i, self.idx[:ns]]
        else: s = np.full(ns, -np.inf, np.float32)
        return (s - self.pen).astype(np.float32), None

def run(sub, r, variant, order, cond, p_w, rounds, seed, lam=0.0, p_c=0.9, r_reopen=0.5):
    c = getg(sub, r); kfam, afam, canon = c["kfam"], c["afam"], c["canon"]; nkb = len(kfam); n = len(afam)
    R = GRet(c, r, variant, n, pen=lam * c["iqr"] if cond == "prov" else 0.0)
    pw = user_pw(c, p_w, "const") if cond != "static" else np.zeros(n)
    rng = np.random.default_rng(seed * 7919 + int(p_w * 1000))
    base_on = np.ones(nkb, bool)
    src = np.zeros(n, int); thr = np.zeros(n, int); dep = np.zeros(n, int); evk = [None] * n; gk = [None] * n
    self_on = np.zeros(n, bool); has = np.zeros(n, bool); ns = 0
    per = int(np.ceil(len(order) / rounds)); pending = []; out = []; fresh = []
    miss_keys, miss_posts = set(), set()
    for rd in range(rounds):
        for e in fresh: self_on[e] = has[e]
        fresh = []
        for e in [e for due, e in pending if due == rd]: self_on[e] = False
        pending = [p for p in pending if p[0] != rd]
        m = collections.Counter(); drift = collections.defaultdict(float); dn = collections.Counter()
        for i in order[rd * per:(rd + 1) * per]:
            threads, isself, ent, sc, sall = retrieve(R, i, base_on, src[:ns], thr[:ns], self_on[:ns])
            t1 = threads[0]; f1 = kfam[t1]; ok = f1 == afam[i] and f1 >= 0
            hit5 = any(kfam[t] == afam[i] for t in threads) and afam[i] >= 0
            m["n"] += 1; m["c"] += ok; m["h5"] += hit5; m["self"] += isself[0]
            # evidence key of the answer shown to ask i, and its depth
            if isself[0]: e0 = ent[0]; ev = f"e:{src[e0]}|{evk[e0]}"; d = dep[e0]
            else: ev = f"t:{t1}"; d = 0
            key = f"{i}|{ev}"; gi = c["gkeys"].get(key)
            if d == 0 and c["ids"][t1] not in c["has_ans"]: miss_posts.add(int(t1))
            if gi is not None and "drift" in c: drift[min(d, 2)] += float(c["drift"][gi]); dn[min(d, 2)] += 1
            m["self_d2"] += isself[0] and d >= 2
            if isself[0]:
                sok = c["static_ok"][i]; e_ok = kfam[thr[e0]] == afam[src[e0]] and kfam[thr[e0]] >= 0
                m["eff_c" if e_ok else "eff_w"] += int(ok) - int(sok)
                m[f"eff_{'c' if e_ok else 'w'}_d{min(d, 2)}"] += int(ok) - int(sok); m[f"served_d{min(d, 2)}"] += 1
                if not ok: m["wrong_self_d1" if d == 1 else "wrong_self_d2"] += 1
                # capture diagnostics as in v2
            cf = canon[afam[i]] if afam[i] >= 0 else -1
            if cf >= 0 and ns and self_on[:ns].any():
                cpos = np.where(c["cand_idx"][i] == cf)[0]
                if len(cpos):
                    csc = c["cand_sc"][i][cpos[0]]; act = np.where(self_on[:ns] & np.isfinite(sall))[0]; m["cov"] += 1
                    if len(act):
                        good = kfam[thr[act]] == afam[i]
                        m["capture"] += bool((sall[act] > csc).any()); m["capture_w"] += bool((sall[act][~good] > csc).any())
                        if c["static_ok"][i]: m["cov_ok"] += 1; m["steal"] += bool((sall[act][~good] > csc).any())
            acc = rng.random() < (p_c if ok else pw[i]); m["accepted"] += acc; m["acc_w"] += acc and not ok; m["wrong"] += not ok
            if acc and cond != "static":
                src[ns] = i; thr[ns] = t1; dep[ns] = d + 1; evk[ns] = ev; gk[ns] = key
                if gi is None: miss_keys.add(key); has[ns] = False; m["ent_missing"] += 1
                else:
                    has[ns] = True
                    if r in DENSE: R.E[ns] = c[f"E_{variant}"][gi]
                    else: R.idx[ns] = gi
                self_on[ns] = False; fresh.append(ns); ns += 1
                m["ent_c" if ok else "ent_w"] += 1; m[f"ent_d{min(d + 1, 3)}"] += 1
                if not ok and cond == "reopen" and rng.random() < r_reopen: pending.append((rd + 1, ns - 1))
        out.append(dict(n=m["n"], acc=m["c"] / m["n"], hit5=m["h5"] / m["n"], self_share=m["self"] / m["n"], self_d2=m["self_d2"] / m["n"],
                        accept_rate=m["accepted"] / m["n"], acc_w=m["acc_w"], wrong=m["wrong"], wrong_self_d1=m["wrong_self_d1"], wrong_self_d2=m["wrong_self_d2"],
                        cov=m["cov"], capture=m["capture"], capture_w=m["capture_w"], cov_ok=m["cov_ok"], steal=m["steal"],
                        entries=int(ns), ent_c=m["ent_c"], ent_w=m["ent_w"], ent_d1=m["ent_d1"], ent_d2=m["ent_d2"], ent_d3=m["ent_d3"], ent_missing=m["ent_missing"],
                        eff_c=m["eff_c"], eff_w=m["eff_w"], active=int(self_on[:ns].sum()),
                        eff_c_d1=m["eff_c_d1"], eff_w_d1=m["eff_w_d1"], eff_c_d2=m["eff_c_d2"], eff_w_d2=m["eff_w_d2"], served_d1=m["served_d1"], served_d2=m["served_d2"],
                        drift=[(drift[d] / dn[d]) if dn[d] else None for d in (0, 1, 2)], drift_n=[dn[0], dn[1], dn[2]]))
    return out, miss_keys, miss_posts

def plan():
    P = [("askgen", "static", 0.0, 0.0)]
    for v in VARIANTS:
        for pw in PW_GRID: P += [(v, "naive", pw, 0.0), (v, "reopen", pw, 0.0)]
    for lam in PROV_LAM:
        for pw in PW_GRID: P.append(("askgen", "prov", pw, lam))
    ov = os.environ.get("ONLY_VARIANT")
    return [x for x in P if x[0] == ov] if ov else P

def task(args):
    sub, r, seed, rounds = args
    c = getg(sub, r); n = len(c["afam"])
    order = np.random.default_rng(1000 + seed).permutation(n)
    res, mk, mp = [], set(), set()
    for v, cnd, pw, lam in plan():
        cu, a, b = run(sub, r, v, order, cnd, pw, rounds, seed, lam); mk |= a; mp |= b
        res.append(dict(sub=sub, ret=r, seed=seed, rounds=rounds, variant=v, user="const", cond=cnd, p_w=None if cnd == "static" else pw, lam=lam, curve=cu))
    return sub, r, res, mk, mp

if __name__ == "__main__":
    rounds = int(sys.argv[1]); seeds = int(sys.argv[2]); subs = sys.argv[3].split(","); rets = sys.argv[4].split(","); tag = sys.argv[5]
    tasks = [(s, r, sd, rounds) for s in subs for r in rets for sd in range(seeds)]
    t = time.time(); allres = []; MK = collections.defaultdict(set); MP = collections.defaultdict(set)
    with Pool(int(os.environ.get("NPROC", 12))) as p:
        for k, (s, r, res, mk, mp) in enumerate(p.imap_unordered(task, tasks)):
            allres.extend(res); MK[s] |= mk; MP[s] |= mp
            if k % 10 == 0: print(k, len(tasks), f"{time.time()-t:.0f}s", flush=True)
    nk = np = 0
    for s in subs:
        for name, ms in ((f"missing{TAG}_{s}.json", MK[s]), (f"missing_posts_{s}.json", MP[s])):
            mf = os.path.join(G, name); old = set(json.load(open(mf))) if os.path.exists(mf) else set()
            json.dump(sorted(old | ms), open(mf, "w"))
        nk += len(MK[s]); np += len(MP[s])
    fn = os.path.join(RES, f"sim_gen_{tag}.json")
    json.dump(dict(runtime_s=time.time() - t, missing_keys=nk, missing_posts=np, runs=allres), open(fn, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote", fn, f"{time.time()-t:.0f}s", "missing gen keys:", nk, "missing posts:", np)

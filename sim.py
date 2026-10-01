"""Feedback-loop simulation over cached retrieval scores (see prep.py).

Same loop as the pilot (../pilot_real.py): asks arrive in random order in R rounds; the top-1
result is shown; a simulated user accepts a correct answer w.p. p_c=0.9 and a wrong one w.p. p_w;
accepted answers are written back as new entries (ask text -> thread it was given).
Conditions: static (no write-back), naive, reopen (demotion: half of accepted wrong entries are
reopened next round and removed), reopen+quarantine (also withdraw any entry rejected twice).
Extra over the pilot: top-5 hit, 3 retrievers, 12 forums, label-free monitors:
  ind_dis   : share of asks where the deployed top-1 thread differs from a static, independent
              retriever's top-1 thread (BM25 for dense retrievers, bge for BM25)
  bank_drift: share of a fixed 200-ask probe bank whose top-1 thread changed since round 0 (read-only probes)
"""
import os, sys, json, time, itertools, collections
import numpy as np
from multiprocessing import Pool
from common import *

IND = {"bge": "bm25", "minilm": "bm25", "bm25": "bge"}
_C = {}
def cache(sub, r):
    if (sub, r) not in _C:
        z = np.load(os.path.join(CACHE, f"{sub}_{r}.npz")); _C[(sub, r)] = {k: z[k] for k in z.files}
    return _C[(sub, r)]

def retrieve(i, c, base_on, src, thr, self_on, k=5):
    ci, cs = c["cand_idx"][i], c["cand_sc"][i]
    m = base_on[ci]; bi, bs = ci[m][:k], cs[m][:k]
    if len(src):
        s = c["S"][i, src]; s = np.where(self_on, s, -np.inf)
        if len(s) > k:
            top = np.argpartition(-s, k)[:k]
        else:
            top = np.arange(len(s))
        ss = s[top]; keep = np.isfinite(ss); top, ss = top[keep], ss[keep]
    else:
        top, ss = np.zeros(0, int), np.zeros(0, np.float32)
    # merge: entries are (score, is_self, entry index)
    sc = np.concatenate([bs, ss]); isself = np.r_[np.zeros(len(bs), bool), np.ones(len(ss), bool)]
    ent = np.r_[bi, top]; o = np.argsort(-sc, kind="stable")[:k]
    threads = ent[o].copy(); so = isself[o]
    if so.any(): threads[so] = thr[ent[o][so]]
    return threads, isself[o], ent[o]

def run(sub, r, order, cond, p_w, rounds, seed, bank, p_c=0.9, r_reopen=0.5, q_thresh=2):
    c = cache(sub, r); ind = cache(sub, IND[r])["cand_idx"][:, 0]
    kfam, afam = c["kfam"], c["afam"]; nkb = len(kfam); n = len(afam)
    rng = np.random.default_rng(seed * 7919 + int(p_w * 1000))
    base_on = np.ones(nkb, bool); base_bad = np.zeros(nkb, int)
    src = np.zeros(n, int); thr = np.zeros(n, int); self_on = np.zeros(n, bool); self_bad = np.zeros(n, int); ns = 0
    seen = collections.Counter(); per = int(np.ceil(len(order) / rounds)); pending = []; out = []
    def probe():
        return np.array([retrieve(i, c, base_on, src[:ns], thr[:ns], self_on[:ns], k=1)[0][0] for i in bank])
    bank0 = probe()
    fresh = []
    for rd in range(rounds):
        for e in fresh: self_on[e] = True           # batch re-index at round start (pilot semantics)
        fresh = []
        for e in [e for due, e in pending if due == rd]: self_on[e] = False
        pending = [p for p in pending if p[0] != rd]
        m = collections.Counter()
        for i in order[rd * per:(rd + 1) * per]:
            threads, isself, ent = retrieve(i, c, base_on, src[:ns], thr[:ns], self_on[:ns])
            t1 = threads[0]; f1 = kfam[t1]; ok = f1 == afam[i] and f1 >= 0
            hit5 = any(kfam[t] == afam[i] for t in threads) and afam[i] >= 0
            rep = seen[afam[i]] > 0; seen[afam[i]] += 1
            m["n"] += 1; m["c"] += ok; m["h5"] += hit5; m["rep"] += rep; m["rc"] += ok and rep; m["rh5"] += hit5 and rep
            m["self"] += isself[0]; m["ind_dis"] += t1 != ind[i]
            m["ind_dis_fam"] += kfam[t1] != kfam[ind[i]] or kfam[t1] < 0
            if isself[0]:
                b0 = c["cand_idx"][i][0]; sok = kfam[b0] == afam[i] and kfam[b0] >= 0
                e_ok = kfam[thr[ent[0]]] == afam[src[ent[0]]] and kfam[thr[ent[0]]] >= 0
                m["eff_c" if e_ok else "eff_w"] += int(ok) - int(sok)
            if isself[0]:   # pilot's failed signal: does the best human-written entry point elsewhere?
                hb = threads[~isself]
                hb = hb[0] if len(hb) else c["cand_idx"][i][base_on[c["cand_idx"][i]]][0]
                m["selfdis"] += hb != t1
            acc = rng.random() < (p_c if ok else p_w); m["accepted"] += acc
            if not acc:
                if isself[0]:
                    self_bad[ent[0]] += 1
                    if cond == "reopen+quarantine" and self_bad[ent[0]] >= q_thresh: self_on[ent[0]] = False
                else:
                    base_bad[ent[0]] += 1
                    if cond == "reopen+quarantine" and base_bad[ent[0]] >= q_thresh: base_on[ent[0]] = False
            elif cond != "static":
                src[ns] = i; thr[ns] = t1; self_on[ns] = False; fresh.append(ns); ns += 1
                m["ent_c" if ok else "ent_w"] += 1
                if not ok and cond in ("reopen", "reopen+quarantine") and rng.random() < r_reopen:
                    pending.append((rd + 1, ns - 1))
        b = probe()
        out.append(dict(n=m["n"], acc=m["c"] / m["n"], hit5=m["h5"] / m["n"], rep=m["rep"],
                        rep_acc=m["rc"] / max(m["rep"], 1), rep_hit5=m["rh5"] / max(m["rep"], 1),
                        self_share=m["self"] / m["n"], accept_rate=m["accepted"] / m["n"], selfdis=m["selfdis"] / m["n"],
                        ind_dis=m["ind_dis"] / m["n"], ind_dis_fam=m["ind_dis_fam"] / m["n"],
                        bank_drift=float(np.mean(b != bank0)),
                        bank_acc=float(np.mean([(kfam[t] == afam[i]) and kfam[t] >= 0 for t, i in zip(b, bank)])),
                        bank_ind_dis=float(np.mean(b != ind[bank])), entries=int(ns), ent_c=m["ent_c"], ent_w=m["ent_w"], eff_c=m["eff_c"], eff_w=m["eff_w"], active=int(self_on[:ns].sum())))
    return out

PW_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
PW_SAFE = [0.3, 0.6, 0.8, 0.95]
def task(args):
    sub, r, seed, rounds = args
    c = cache(sub, r); n = len(c["afam"])
    order = np.random.default_rng(1000 + seed).permutation(n)
    bank = np.random.default_rng(7).choice(n, size=min(200, n), replace=False)
    res = [dict(sub=sub, ret=r, seed=seed, rounds=rounds, p_w=None, cond="static", curve=run(sub, r, order, "static", 0.0, rounds, seed, bank))]
    for pw in PW_GRID:
        res.append(dict(sub=sub, ret=r, seed=seed, rounds=rounds, p_w=pw, cond="naive", curve=run(sub, r, order, "naive", pw, rounds, seed, bank)))
    for pw in PW_SAFE:
        for cond in ("reopen", "reopen+quarantine"):
            res.append(dict(sub=sub, ret=r, seed=seed, rounds=rounds, p_w=pw, cond=cond, curve=run(sub, r, order, cond, pw, rounds, seed, bank)))
    return res

if __name__ == "__main__":
    rounds = int(sys.argv[1]); seeds = int(sys.argv[2]); subs = sys.argv[3].split(",") if len(sys.argv) > 3 else FORUMS
    tasks = [(s, r, sd, rounds) for s in subs for r in RETRIEVERS for sd in range(seeds)]
    t = time.time(); allres = []
    with Pool(14) as p:
        for k, res in enumerate(p.imap_unordered(task, tasks)):
            allres.extend(res)
            if k % 20 == 0: print(k, len(tasks), f"{time.time()-t:.0f}s", flush=True)
    meta = {}
    for s in subs:
        z = cache(s, "bm25"); fam = collections.Counter(z["afam"].tolist())
        meta[s] = dict(kb=int(len(z["kfam"])), asks=int(len(z["afam"])), families=len(fam), repeat_asks=int(sum(v - 1 for v in fam.values())))
    fn = os.path.join(RES, f"sim_r{rounds}_{'all' if len(subs)==12 else '_'.join(subs)}.json")
    json.dump(dict(meta=meta, runtime_s=time.time() - t, runs=allres), open(fn, "w"))
    print("wrote", fn, f"{time.time()-t:.0f}s")

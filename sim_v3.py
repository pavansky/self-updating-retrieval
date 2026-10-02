"""Version-3 simulation: two further baselines on top of sim_v2.py (same loop, same random streams).

  filter (Doc2Query-- style): before an accepted (ask, shown thread) pair is written back, it is scored by the
      cross-encoder already used in v2 (cross-encoder/ms-marco-MiniLM-L-6-v2; cache/v3/{sub}_cepairs.npz) and dropped
      if its score is below a label-free threshold: the q-quantile (q in FILTER_Q) of the cross-encoder scores of the
      pairs accepted in round 1. The threshold is fixed after round 1 and round-1 entries are filtered with it
      (as for the v2 confidence gate). Run for ask entries and ask+thread entries.
  cache (semantic cache): stored entries are question entries used as a cache. For an incoming ask, if the highest
      similarity (deployed retriever score) between the ask and any stored question is >= tau, the stored thread is
      shown (cache hit); otherwise the base retriever answers from the original KB only. As in a semantic cache,
      only accepted answers to cache misses are stored. tau = q-quantile (q in CACHE_Q) over asks of the similarity to
      the nearest other ask (label-free).
  naive (sanity check): must reproduce sim_v2 naive exactly.
With an unchanged "cond", run3() is sim_v2.run() line for line.
"""
import os, sys, json, time, collections
import numpy as np
from multiprocessing import Pool
from common import *
import sim_v2
from sim_v2 import get, Ret, retrieve, user_pw, PW_GRID
V3 = os.path.join(CACHE, "v3")
FILTER_Q = [0.25, 0.5, 0.75]
CACHE_Q = [0.5, 0.75, 0.9, 0.95]
_CE = {}

def ce_lookup(sub):
    if sub not in _CE:
        _CE.clear(); z = np.load(os.path.join(V3, f"{sub}_cepairs.npz")); _CE[sub] = (z["keys"], z["sc"])
    return _CE[sub]

def run3(sub, r, variant, user, order, cond, p_w, rounds, seed, param=None, p_c=0.9):
    c = get(sub, r); R = Ret(c, r, variant)
    kfam, afam, canon = c["kfam"], c["afam"], c["canon"]; nkb = len(kfam); n = len(afam)
    pw = user_pw(c, p_w, user)
    rng = np.random.default_rng(seed * 7919 + int(p_w * 1000))
    base_on = np.ones(nkb, bool)
    src = np.zeros(n, int); thr = np.zeros(n, int); self_on = np.zeros(n, bool); cesc = np.zeros(n); eok = np.zeros(n, bool); ns = 0; drop = (0, 0)
    per = int(np.ceil(len(order) / rounds)); out = []; fresh = []; thr_f = None; missing = set()
    if cond == "filter": CK, CS = ce_lookup(sub)
    if cond == "cache":
        Sf = np.where(np.isfinite(c["S"]), c["S"], -np.inf); tau = float(np.quantile(Sf.max(1), param))
    for rd in range(rounds):
        if cond == "filter" and rd == 1:
            thr_f = float(np.quantile(r0ce, param)) if len(r0ce) else -np.inf
            dr = [e for e in fresh if cesc[e] < thr_f]; drop = (sum(eok[e] for e in dr), sum(not eok[e] for e in dr))
            fresh = [e for e in fresh if cesc[e] >= thr_f]
        for e in fresh: self_on[e] = True
        fresh = []
        m = collections.Counter(); r0ce = []
        for i in order[rd * per:(rd + 1) * per]:
            if cond == "cache":
                bi, bs, _ = R.base(i, base_on); act = np.where(self_on[:ns])[0]; hit = False
                if len(act):
                    sa = c["S"][i, src[act]]; j = int(np.argmax(sa)); hit = bool(sa[j] >= tau)
                if hit:
                    e = act[j]; threads = np.r_[thr[e], bi[:4]]; isself = np.r_[True, np.zeros(4, bool)]; ent = np.r_[e, bi[:4]]
                else:
                    threads = bi[:5].copy(); isself = np.zeros(len(threads), bool); ent = bi[:5]
                m["hits"] += hit
            else:
                threads, isself, ent, sc, sall = retrieve(R, i, base_on, src[:ns], thr[:ns], self_on[:ns])
            t1 = threads[0]; f1 = kfam[t1]; ok = f1 == afam[i] and f1 >= 0
            hit5 = any(kfam[t] == afam[i] for t in threads) and afam[i] >= 0
            m["n"] += 1; m["c"] += ok; m["h5"] += hit5; m["self"] += isself[0]
            if isself[0]:
                sok = c["static_ok"][i]
                e_ok = kfam[thr[ent[0]]] == afam[src[ent[0]]] and kfam[thr[ent[0]]] >= 0
                m["eff_c" if e_ok else "eff_w"] += int(ok) - int(sok)
                if cond == "cache": m["hit_ok"] += ok
            acc = rng.random() < (p_c if ok else pw[i]); m["accepted"] += acc; m["acc_w"] += acc and not ok; m["wrong"] += not ok
            if acc and cond != "static":
                if cond == "cache" and isself[0]: continue
                if cond == "filter":
                    key = np.int64(i) * nkb + t1; pos = min(np.searchsorted(CK, key), len(CK) - 1)
                    if CK[pos] != key: missing.add(int(key)); s_ce = np.inf
                    else: s_ce = float(CS[pos])
                    if rd == 0: r0ce.append(s_ce)
                    if thr_f is not None and s_ce < thr_f: m["filtered"] += 1; continue
                    cesc[ns] = s_ce
                src[ns] = i; thr[ns] = t1; eok[ns] = ok; self_on[ns] = False; fresh.append(ns); ns += 1
                m["ent_c" if ok else "ent_w"] += 1
        out.append(dict(n=m["n"], acc=m["c"] / m["n"], hit5=m["h5"] / m["n"], self_share=m["self"] / m["n"],
                        accept_rate=m["accepted"] / m["n"], acc_w=m["acc_w"], wrong=m["wrong"], entries=int(ns) - sum(drop),
                        ent_c=m["ent_c"] - (drop[0] if rd == 1 else 0), ent_w=m["ent_w"] - (drop[1] if rd == 1 else 0), eff_c=m["eff_c"], eff_w=m["eff_w"], filtered=m["filtered"],
                        hits=m["hits"], hit_ok=m["hit_ok"], active=int(self_on[:ns].sum())))
    return out, R.missing, missing

def plan(r):
    P = [("ask", "const", "naive", pw, None) for pw in (0.3, 0.95)]       # sanity: identical to sim_v2
    for v in ("ask", "askdoc"):
        for q in FILTER_Q: P += [(v, "const", "filter", pw, q) for pw in PW_GRID]
    for q in CACHE_Q: P += [("ask", "const", "cache", pw, q) for pw in PW_GRID]
    return P

def task(args):
    sub, r, seed, rounds = args
    c = get(sub, r); n = len(c["afam"])
    order = np.random.default_rng(1000 + seed).permutation(n)
    res, miss_ad, miss_ce = [], set(), set()
    for v, u, cnd, pw, q in plan(r):
        cu, ma, mc = run3(sub, r, v, u, order, cnd, pw, rounds, seed, q); miss_ad |= ma; miss_ce |= mc
        res.append(dict(sub=sub, ret=r, seed=seed, rounds=rounds, variant=v, user=u, cond=cnd, p_w=pw, param=q, curve=cu))
    return sub, r, res, miss_ad, miss_ce

if __name__ == "__main__":
    rounds = int(sys.argv[1]); seeds = int(sys.argv[2]); subs = sys.argv[3].split(","); rets = sys.argv[4].split(","); tag = sys.argv[5]
    tasks = [(s, r, sd, rounds) for s in subs for r in rets for sd in range(seeds)]
    t = time.time(); allres = []; MA = collections.defaultdict(set); MC = collections.defaultdict(set)
    with Pool(int(os.environ.get("NPROC", 12))) as p:
        for k, (s, r, res, ma, mc) in enumerate(p.imap_unordered(task, tasks)):
            allres.extend(res); MA[(s, r)] |= ma; MC[s] |= mc
            if k % 20 == 0: print(k, len(tasks), f"{time.time()-t:.0f}s", flush=True)
    n_ad = sum(len(v) for v in MA.values()); n_ce = 0
    for s, ms in MC.items():
        if ms:
            mf = os.path.join(V3, f"missing_ce_{s}.json"); old = set(json.load(open(mf))) if os.path.exists(mf) else set()
            json.dump(sorted(old | ms), open(mf, "w")); n_ce += len(ms)
    for (s, r), ms in MA.items():       # ask+thread pairs never embedded: same mechanism as sim_v2 (prep_v2.py askdoc)
        if ms:
            mf = os.path.join(sim_v2.V2, f"missing_{s}_{r}.json"); old = set(json.load(open(mf))) if os.path.exists(mf) else set()
            json.dump(sorted(old | ms), open(mf, "w"))
    fn = os.path.join(RES, f"sim_v3_{tag}.json")
    json.dump(dict(runtime_s=time.time() - t, missing_ce_pairs=n_ce, missing_askdoc_pairs=n_ad, runs=allres), open(fn, "w"),
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote", fn, f"{time.time()-t:.0f}s", "missing CE pairs:", n_ce, "missing askdoc pairs:", n_ad)

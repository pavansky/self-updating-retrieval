"""Version-2 simulation. Same loop as sim.py (v1); with variant="ask", user="const" and the v1 conditions
it reproduces sim.py's top-1 results exactly (same random streams). Additions:

  entry content (variant):
    ask    : the ask text (v1)
    doc    : the shown thread's own text (title + body prefix), i.e. a duplicate of the shown post with a new id;
             its score for any query equals the base score of that post
    askdoc : ask text + "\n" + shown thread text (embedded / BM25-scored as one entry; cache/v2/*_askdoc.npz)
  user model:
    const  : wrong answers accepted with probability p_w (v1)
    diff   : p_w(i) = sigmoid(a + b*(q_i - 0.5)), q_i = percentile of ask i's no-write-back top-1 score
             within the forum/retriever, b = 6; a is solved so that the mean of p_w(i) over the asks whose
             no-write-back top-1 answer is wrong equals the nominal p_w
  conditions: static, naive, reopen (demotion), reopen+quarantine (v1), and
    gate      : confidence-gated write-back: an accepted answer is written back only if its top-1 score is at least
                the median top-1 score of the round-1 asks (threshold fixed after round 1; round-1 entries filtered)
    footprint : CBR-style competence rule: a new entry is kept only if no existing entry for the same thread has
                ask-ask similarity >= tau (tau = median over asks of the nearest-other-ask similarity; label-free).
                For variant doc all entries for one thread are identical, so only the first is kept.
  retriever "ce": bge first stage over base posts (top-30) and written-back entries, top-20 merged, reranked by
                cross-encoder/ms-marco-MiniLM-L-6-v2 (cache/v2/*_bgece.npz). Variant ask only.
  diagnostics per round: capture = share of asks (canonical post among the cached candidates) for which some
                written-back entry outscores the canonical post; steal = same, restricted to entries pointing to a
                wrong thread and asks whose no-write-back top-1 is correct.
"""
import os, sys, json, time, collections
import numpy as np
from multiprocessing import Pool
from common import *
V2 = os.path.join(CACHE, "v2")
_C = {}

def get(sub, r):
    if (sub, r) in _C: return _C[(sub, r)]
    _C.clear()
    if r == "ce":
        z = np.load(os.path.join(V2, f"{sub}_bgece.npz")); c = {k: z[k] for k in z.files}
        c["S"] = np.load(os.path.join(CACHE, f"{sub}_bge.npz"))["S"]
        n = len(c["afam"])
        # static ranking: bge top-20 base posts reranked by CE
        o = np.argsort(-c["bce"][:, :20], 1, kind="stable")
        c["cand_idx"] = np.take_along_axis(c["bci"][:, :20], o, 1); c["cand_sc"] = np.take_along_axis(c["bce"][:, :20], o, 1)
    else:
        z = np.load(os.path.join(CACHE, f"{sub}_{r}.npz")); c = {k: z[k] for k in z.files}
        p = os.path.join(V2, f"{sub}_{r}_askdoc.npz")
        if os.path.exists(p):
            z = np.load(p); c["ad_keys"] = z["keys"]; c["ad_SC"] = z["SC"]
    kfam = c["kfam"]; nf = int(max(c["afam"].max(), kfam.max())) + 1
    canon = np.full(nf, -1); ok = kfam >= 0; canon[kfam[ok]] = np.where(ok)[0]; c["canon"] = canon
    S = c["S"]; Sf = np.where(np.isfinite(S), S, -np.inf)
    c["tau"] = float(np.median(Sf.max(1)))
    t1 = c["cand_sc"][:, 0]; c["q"] = (np.argsort(np.argsort(t1)) + 0.5) / len(t1)
    c["static_ok"] = (kfam[c["cand_idx"][:, 0]] == c["afam"]) & (kfam[c["cand_idx"][:, 0]] >= 0)
    _C[(sub, r)] = c; return c

def user_pw(c, p_w, user, b=6.0):
    n = len(c["afam"])
    if user == "const": return np.full(n, p_w)
    x = b * (c["q"] - 0.5); w = ~c["static_ok"]
    lo, hi = -30.0, 30.0
    for _ in range(100):
        a = (lo + hi) / 2
        if np.mean(1 / (1 + np.exp(-(a + x[w])))) < p_w: lo = a
        else: hi = a
    return 1 / (1 + np.exp(-(a + x)))

class Ret:
    def __init__(self, c, r, variant):
        self.c, self.r, self.v = c, r, variant; self.nkb = len(c["kfam"]); self.missing = set()
    def base(self, i, base_on):
        c = self.c
        if self.r == "ce":
            ci, s1, s2 = c["bci"][i], c["bcs"][i], c["bce"][i]; m = base_on[ci]; return ci[m], s1[m], s2[m]
        ci, cs = c["cand_idx"][i], c["cand_sc"][i]; m = base_on[ci]; return ci[m], cs[m], None
    def selfsc(self, i, src, thr):
        """first-stage score of each entry for query i (and CE score for r=ce)."""
        c = self.c
        if self.r == "ce":
            n = len(c["afam"]); a1 = np.full(n, -np.inf, np.float32); a2 = np.full(n, -np.inf, np.float32)
            a1[c["aci"][i]] = c["acs"][i]; a2[c["aci"][i]] = c["ace"][i]; return a1[src], a2[src]
        if self.v == "ask": return c["S"][i, src], None
        if self.v == "doc":
            row = np.full(self.nkb, -np.inf, np.float32); row[c["cand_idx"][i]] = c["cand_sc"][i]; return row[thr], None
        key = src.astype(np.int64) * self.nkb + thr; K = c["ad_keys"]
        pos = np.minimum(np.searchsorted(K, key), len(K) - 1); f = K[pos] == key
        if not f.all(): self.missing.update(key[~f].tolist())
        return np.where(f, c["ad_SC"][i, pos], -np.inf), None

def retrieve(R, i, base_on, src, thr, self_on, k=5):
    bi, bs, bce = R.base(i, base_on)
    if len(src):
        s, s2 = R.selfsc(i, src, thr); s = np.where(self_on, s, -np.inf)
    else:
        s, s2 = np.zeros(0, np.float32), None
    if R.r == "ce":
        K1 = 20
        top = np.argsort(-s, kind="stable")[:K1] if len(s) else np.zeros(0, int)
        top = top[np.isfinite(s[top])] if len(top) else top
        sc1 = np.concatenate([bs, s[top]]); sc2 = np.concatenate([bce, s2[top] if len(top) else np.zeros(0, np.float32)])
        isself = np.r_[np.zeros(len(bs), bool), np.ones(len(top), bool)]; ent = np.r_[bi, top]
        o1 = np.argsort(-sc1, kind="stable")[:K1]; o = o1[np.argsort(-sc2[o1], kind="stable")][:k]; sc = sc2
    else:
        bi, bs = bi[:k], bs[:k]
        if len(s) > k: top = np.argpartition(-s, k)[:k]
        else: top = np.arange(len(s))
        ss = s[top]; keep = np.isfinite(ss); top, ss = top[keep], ss[keep]
        sc = np.concatenate([bs, ss]); isself = np.r_[np.zeros(len(bs), bool), np.ones(len(ss), bool)]
        ent = np.r_[bi, top]; o = np.argsort(-sc, kind="stable")[:k]
    threads = ent[o].copy(); so = isself[o]
    if so.any(): threads[so] = thr[ent[o][so]]
    return threads, isself[o], ent[o], sc[o], (s if R.r != "ce" else s2)

def run(sub, r, variant, user, order, cond, p_w, rounds, seed, p_c=0.9, r_reopen=0.5, q_thresh=2):
    c = get(sub, r); R = Ret(c, r, variant)
    kfam, afam, canon = c["kfam"], c["afam"], c["canon"]; nkb = len(kfam); n = len(afam)
    pw = user_pw(c, p_w, user) if cond != "static" else np.zeros(n)
    rng = np.random.default_rng(seed * 7919 + int(p_w * 1000))
    base_on = np.ones(nkb, bool); base_bad = np.zeros(nkb, int)
    src = np.zeros(n, int); thr = np.zeros(n, int); self_on = np.zeros(n, bool); self_bad = np.zeros(n, int); esc = np.zeros(n); ns = 0
    alive = np.zeros(n, bool)       # written and not removed (active or waiting for re-index)
    seen = collections.Counter(); per = int(np.ceil(len(order) / rounds)); pending = []; out = []; fresh = []; gate_t = None
    for rd in range(rounds):
        if cond == "gate" and rd == 1:
            gate_t = float(np.median(r0)); drop = [e for e in fresh if esc[e] < gate_t]
            for e in drop: alive[e] = False
            fresh = [e for e in fresh if esc[e] >= gate_t]
        for e in fresh: self_on[e] = True
        fresh = []
        for e in [e for due, e in pending if due == rd]: self_on[e] = False; alive[e] = False
        pending = [p for p in pending if p[0] != rd]
        m = collections.Counter(); r0 = []
        for i in order[rd * per:(rd + 1) * per]:
            threads, isself, ent, sc, sall = retrieve(R, i, base_on, src[:ns], thr[:ns], self_on[:ns])
            t1 = threads[0]; f1 = kfam[t1]; ok = f1 == afam[i] and f1 >= 0
            hit5 = any(kfam[t] == afam[i] for t in threads) and afam[i] >= 0
            rep = seen[afam[i]] > 0; seen[afam[i]] += 1
            m["n"] += 1; m["c"] += ok; m["h5"] += hit5; m["rep"] += rep; m["rc"] += ok and rep; m["rh5"] += hit5 and rep
            m["self"] += isself[0]; r0.append(sc[0])
            # capture diagnostics
            cf = canon[afam[i]] if afam[i] >= 0 else -1
            if cf >= 0 and ns and self_on[:ns].any():
                cpos = np.where(c["cand_idx"][i] == cf)[0]
                if len(cpos) and base_on[cf]:
                    csc = c["cand_sc"][i][cpos[0]]; act = np.where(self_on[:ns] & np.isfinite(sall))[0]
                    m["cov"] += 1
                    if len(act):
                        good = kfam[thr[act]] == afam[i]
                        m["capture"] += bool((sall[act] > csc).any())
                        m["capture_w"] += bool((sall[act][~good] > csc).any())
                        if c["static_ok"][i]:
                            m["cov_ok"] += 1; m["steal"] += bool((sall[act][~good] > csc).any())
            if isself[0]:
                sok = c["static_ok"][i]
                e_ok = kfam[thr[ent[0]]] == afam[src[ent[0]]] and kfam[thr[ent[0]]] >= 0
                m["eff_c" if e_ok else "eff_w"] += int(ok) - int(sok)
            acc = rng.random() < (p_c if ok else pw[i]); m["accepted"] += acc; m["acc_w"] += acc and not ok; m["wrong"] += not ok
            if not acc:
                if isself[0]:
                    self_bad[ent[0]] += 1
                    if cond == "reopen+quarantine" and self_bad[ent[0]] >= q_thresh: self_on[ent[0]] = False; alive[ent[0]] = False
                else:
                    base_bad[ent[0]] += 1
                    if cond == "reopen+quarantine" and base_bad[ent[0]] >= q_thresh: base_on[ent[0]] = False
            elif cond != "static":
                if cond == "gate" and gate_t is not None and sc[0] < gate_t: m["gated"] += 1; continue
                if cond == "footprint":
                    same = np.where(alive[:ns] & (thr[:ns] == t1))[0]
                    if len(same) and (variant == "doc" or c["S"][i, src[same]].max() >= c["tau"]): m["redundant"] += 1; continue
                src[ns] = i; thr[ns] = t1; esc[ns] = sc[0]; self_on[ns] = False; alive[ns] = True; fresh.append(ns); ns += 1
                m["ent_c" if ok else "ent_w"] += 1
                if not ok and cond in ("reopen", "reopen+quarantine") and rng.random() < r_reopen:
                    pending.append((rd + 1, ns - 1))
        out.append(dict(n=m["n"], acc=m["c"] / m["n"], hit5=m["h5"] / m["n"], rep=m["rep"],
                        rep_acc=m["rc"] / max(m["rep"], 1), self_share=m["self"] / m["n"], accept_rate=m["accepted"] / m["n"],
                        acc_w=m["acc_w"], wrong=m["wrong"], cov=m["cov"], capture=m["capture"], capture_w=m["capture_w"],
                        cov_ok=m["cov_ok"], steal=m["steal"], entries=int(ns), ent_c=m["ent_c"], ent_w=m["ent_w"],
                        eff_c=m["eff_c"], eff_w=m["eff_w"], gated=m["gated"], redundant=m["redundant"], active=int(self_on[:ns].sum())))
    return out, R.missing

PW_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
PW_SAFE = [0.3, 0.6, 0.8, 0.95]
VARIANTS = ["ask", "doc", "askdoc"]

def plan(r):
    P = [("ask", "const", "static", 0.0)]
    vs = ["ask"] if r == "ce" else VARIANTS
    for v in vs:
        for pw in PW_GRID: P += [(v, "const", "naive", pw), (v, "const", "reopen", pw)]
        for pw in PW_SAFE: P += [(v, "const", cnd, pw) for cnd in ("reopen+quarantine", "gate", "footprint")]
    for v in (["ask"] if r == "ce" else ["ask", "askdoc"]):
        for pw in PW_GRID: P += [(v, "diff", "naive", pw), (v, "diff", "reopen", pw)]
    ov = os.environ.get("ONLY_VARIANT")
    return [x for x in P if x[0] == ov] if ov else P

def task(args):
    sub, r, seed, rounds = args
    c = get(sub, r); n = len(c["afam"])
    order = np.random.default_rng(1000 + seed).permutation(n)
    res, miss = [], set()
    for v, u, cnd, pw in plan(r):
        cu, ms = run(sub, r, v, u, order, cnd, pw, rounds, seed); miss |= ms
        res.append(dict(sub=sub, ret=r, seed=seed, rounds=rounds, variant=v, user=u, cond=cnd, p_w=None if cnd == "static" else pw, curve=cu))
    return sub, r, res, miss

if __name__ == "__main__":
    rounds = int(sys.argv[1]); seeds = int(sys.argv[2]); subs = sys.argv[3].split(","); rets = sys.argv[4].split(","); tag = sys.argv[5]
    tasks = [(s, r, sd, rounds) for s in subs for r in rets for sd in range(seeds)]
    t = time.time(); allres = []; MISS = collections.defaultdict(set)
    with Pool(int(os.environ.get("NPROC", 12))) as p:
        for k, (s, r, res, miss) in enumerate(p.imap_unordered(task, tasks)):
            allres.extend(res); MISS[(s, r)] |= miss
            if k % 20 == 0: print(k, len(tasks), f"{time.time()-t:.0f}s", flush=True)
    nmiss = 0
    for (s, r), ms in MISS.items():
        if ms:
            mf = os.path.join(V2, f"missing_{s}_{r}.json"); old = set(json.load(open(mf))) if os.path.exists(mf) else set()
            json.dump(sorted(old | ms), open(mf, "w")); nmiss += len(ms - old) + len(ms & old)
    fn = os.path.join(RES, f"sim_v2_{tag}.json")
    json.dump(dict(runtime_s=time.time() - t, missing_pairs=nmiss, runs=allres), open(fn, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote", fn, f"{time.time()-t:.0f}s", "missing askdoc pairs:", nmiss)

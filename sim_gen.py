"""Paper-2 simulation: the stored entry carries a generated answer. Same loop, acceptance model and random streams
as sim_v2.run (cond static/naive/reopen); with every entry inactive it is sim_v2 static line for line, and with
user="root" and the v1 conditions it reproduces the first version of this file exactly.

Entry e (index in creation order): src ask src[e], evidence key evk[e] (what src's answer was generated from), gen key
"{src}|{evk}", root thread thr[e] (the base thread at the end of the evidence chain), depth dep[e] = 1 + depth of the
evidence. Entry text by variant:
    gen     : the generated answer only
    askgen  : src ask + generated answer (the realistic self-written Q&A entry)
    genthr  : generated answer + the root thread's text (the "cite the source" countermeasure)
    askhum  : src ask + the root thread's HUMAN answer, truncated to the generated answer's length   } source-bias
    hum     : the root thread's HUMAN answer, truncated to the generated answer's length             } controls
Entry score for ask i: dense = ask embedding . entry embedding; bm25 = BM_v[i, entry] (float32).

Two notions of correctness are recorded for every served answer:
    root    : the root thread of the evidence chain is in the ask's duplicate family (paper 1's label)
    content : root correct AND the independent judge (judge_gen.py) says the generated answer gives the root thread's
              solution (label YES); for roots without a human answer on record (label NOREF) content = root.
User models: user="root" accepts with p_c if root-correct else p_w (paper 1); user="content" accepts with p_c if
content-correct else p_w, so a correct-root answer that lost the solution is treated like a wrong one.
Every served (ask, evidence) key must be generated and judged; keys that are not are recorded in
cache/gen/missing{TAG}_{sub}.json and the loop (run_gen_v3.sh) iterates until none are missing.

Conditions:
    static, naive
    reopen          an entry whose answer was wrong (in the user model's sense) is reopened w.p. 0.5 before it becomes
                    retrievable (paper 1's rule; equivalent to a write-time filter with recall 0.5 and no false positives)
    reopen_dD[_fF]  reopened w.p. 0.5 only after it has been retrievable for D rounds; correct entries are falsely
                    reopened w.p. F/100 after the same delay
    prov            every entry score reduced by lam * IQR of the static top-1 scores (uniform down-weighting)
    provdepth       entry score reduced by lam * IQR * depth (provenance-aware)
    cap1            only answers generated from a base thread are stored (no entry of depth >= 2 is ever written)
    regen           when an entry is retrieved, the answer is generated from its root thread instead of the entry text
Per run: per-round aggregates (as v1, plus rep/rep_acc/cacc and wrong entries by depth) and per-ask outcome bits
(okb root, cokb content; hex of numpy.packbits in ask-index order) and the depth of the served entry per ask (sdep,
uint8 hex, 0 = base thread) for family-level bootstrap, ceiling analysis and counterfactual harm by depth.
"""
import os, sys, json, time, collections, re
import numpy as np
from multiprocessing import Pool
from common import *
import sim_v2
from sim_v2 import retrieve, user_pw, PW_GRID
from prep_gen import G, TAG, VARIANTS, DENSE, parse_ev, kb_ids
PROV_LAM = [0.5, 1.0]
CTRL = ["askhum", "hum"]          # source-bias controls (embed_control.py)
_C = {}

def getg(sub, r):
    if (sub, r) in _C: return _C[(sub, r)]
    _C.clear()
    c = sim_v2.get(sub, r)
    c = dict(c)
    if r in DENSE: c["A"] = np.load(os.path.join(G, f"{sub}_{r}_asks.npy"))
    p = os.path.join(G, f"{sub}_entries{TAG}.npz")
    if os.path.exists(p):
        z = np.load(p); keys = z["keys"].tolist(); c["gkeys"] = {k: i for i, k in enumerate(keys)}
        for v in VARIANTS:
            c[f"E_{v}"] = z[f"E_{v}_{r}"].astype(np.float32) if r in DENSE else z[f"BM_{v}"].astype(np.float32)
        c["drift"] = z["drift_ans"]
        pc = os.path.join(G, f"{sub}_entries{TAG}_ctrl.npz")
        if os.path.exists(pc):
            zc = np.load(pc)
            if zc["keys"].tolist() == keys:
                for v in CTRL:
                    nm = f"E_{v}_{r}" if r in DENSE else f"BM_{v}"
                    if nm in zc.files: c[f"E_{v}"] = zc[nm].astype(np.float32)
        jp = os.path.join(G, f"{sub}_judge{TAG}.jsonl")
        lab = np.array(["MISSING"] * len(keys), dtype=object)
        if os.path.exists(jp):
            for l in open(jp):
                d = json.loads(l); i = c["gkeys"].get(d["key"])
                if i is not None and d["label"] in ("YES", "PARTLY", "NO", "ABSTAIN", "NOREF", "FILTERED"): lab[i] = d["label"]
        c["jlab"] = lab
    else:
        c["gkeys"] = {}; c["jlab"] = np.array([], dtype=object)
    ap = os.path.join(G, f"{sub}_answers.json")
    aj = json.load(open(ap)) if os.path.exists(ap) else {}
    c["has_ans"] = set(aj)                                   # fetched (possibly with no answer on record)
    c["ans_ok"] = {k for k, v in aj.items() if v}            # an accepted or top-voted human answer exists
    c["ids"] = kb_ids(sub)
    s1 = c["cand_sc"][:, 0]; c["iqr"] = float(np.quantile(s1, 0.75) - np.quantile(s1, 0.25))
    _C[(sub, r)] = c; return c

class GRet:
    """sim_v2.Ret interface over generated entries; pen is a uniform penalty, penv a per-entry penalty."""
    def __init__(self, c, r, variant, n, pen=0.0):
        self.c, self.r, self.v, self.pen = c, r, variant, pen
        self.E = np.zeros((n, c["A"].shape[1]), np.float32) if r in DENSE else None
        self.idx = np.zeros(n, int)          # column in BM for bm25
        self.penv = np.zeros(n, np.float32)
    def base(self, i, base_on):
        c = self.c; ci, cs = c["cand_idx"][i], c["cand_sc"][i]; m = base_on[ci]; return ci[m], cs[m], None
    def selfsc(self, i, src, thr):
        ns = len(src)
        if self.r in DENSE: s = self.E[:ns] @ self.c["A"][i]
        elif f"E_{self.v}" in self.c and self.c[f"E_{self.v}"].shape[1]: s = self.c[f"E_{self.v}"][i, self.idx[:ns]]
        else: s = np.full(ns, -np.inf, np.float32)
        return (s - self.pen - self.penv[:ns]).astype(np.float32), None

def parse_cond(cond):
    m = re.fullmatch(r"reopen_d(\d+)(?:_f(\d+))?", cond)
    if cond == "reopen": return 0, 0.0
    if m: return int(m.group(1)), (int(m.group(2)) / 100 if m.group(2) else 0.0)
    return None, 0.0

def run(sub, r, variant, order, cond, p_w, rounds, seed, lam=0.0, p_c=0.9, r_reopen=0.5, user="root"):
    c = getg(sub, r); kfam, afam, canon = c["kfam"], c["afam"], c["canon"]; nkb = len(kfam); n = len(afam)
    R = GRet(c, r, variant, n, pen=lam * c["iqr"] if cond == "prov" else 0.0)
    delay, false = parse_cond(cond)
    pw = user_pw(c, p_w, "const") if cond != "static" else np.zeros(n)
    rng = np.random.default_rng(seed * 7919 + int(p_w * 1000))
    base_on = np.ones(nkb, bool)
    src = np.zeros(n, int); thr = np.zeros(n, int); dep = np.zeros(n, int); evk = [None] * n; gk = [None] * n
    self_on = np.zeros(n, bool); has = np.zeros(n, bool); ns = 0
    per = int(np.ceil(len(order) / rounds)); pending = []; out = []; fresh = []
    miss_keys, miss_posts = set(), set(); seen = collections.Counter()
    okb = np.zeros(n, bool); cokb = np.zeros(n, bool); sdep = np.zeros(n, np.uint8)   # served depth per ask (0 = base thread)
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
            rep = seen[afam[i]] > 0; seen[afam[i]] += 1         # as sim_v2: an earlier ask of the same family exists
            m["n"] += 1; m["c"] += ok; m["h5"] += hit5; m["self"] += isself[0]; m["rep"] += rep; m["rc"] += ok and rep
            # evidence key of the answer shown to ask i, and its depth
            if isself[0]:
                e0 = ent[0]; root = thr[e0]
                if cond == "regen": ev = f"t:{root}"; d = 0
                else: ev = f"e:{src[e0]}|{evk[e0]}"; d = dep[e0]
            else: e0 = None; root = t1; ev = f"t:{t1}"; d = 0
            key = f"{i}|{ev}"; gi = c["gkeys"].get(key)
            if d == 0 and c["ids"][t1] not in c["has_ans"]: miss_posts.add(int(t1))
            lab = c["jlab"][gi] if gi is not None else "MISSING"
            if lab == "MISSING": miss_keys.add(key); cok = ok; m["cunk"] += 1      # provisional until generated and judged
            elif lab in ("NOREF", "FILTERED"): cok = ok; m["cnoref"] += 1
            else: cok = ok and lab == "YES"
            m["cc"] += cok; m["rcc"] += cok and rep; m[f"lab_{lab}_d{min(d, 2)}"] += 1
            okb[i] = ok; cokb[i] = cok; sdep[i] = min(dep[e0], 3) if isself[0] else 0
            if gi is not None and "drift" in c and c["ids"][root] in c["ans_ok"]: drift[min(d, 2)] += float(c["drift"][gi]); dn[min(d, 2)] += 1
            m["self_d2"] += isself[0] and d >= 2
            if isself[0]:
                sok = c["static_ok"][i]; e_ok = kfam[thr[e0]] == afam[src[e0]] and kfam[thr[e0]] >= 0
                m["eff_c" if e_ok else "eff_w"] += int(ok) - int(sok)
                m[f"eff_{'c' if e_ok else 'w'}_d{min(dep[e0], 2)}"] += int(ok) - int(sok); m[f"served_d{min(dep[e0], 2)}"] += 1
                if not ok: m["wrong_self_d1" if dep[e0] == 1 else "wrong_self_d2"] += 1
                if not cok: m["cwrong_self_d1" if dep[e0] == 1 else "cwrong_self_d2"] += 1
            cf = canon[afam[i]] if afam[i] >= 0 else -1
            if cf >= 0 and ns and self_on[:ns].any():
                cpos = np.where(c["cand_idx"][i] == cf)[0]
                if len(cpos):
                    csc = c["cand_sc"][i][cpos[0]]; act = np.where(self_on[:ns] & np.isfinite(sall))[0]; m["cov"] += 1
                    if len(act):
                        good = kfam[thr[act]] == afam[i]
                        m["capture"] += bool((sall[act] > csc).any()); m["capture_w"] += bool((sall[act][~good] > csc).any())
                        if c["static_ok"][i]: m["cov_ok"] += 1; m["steal"] += bool((sall[act][~good] > csc).any())
            ok_user = ok if user == "root" else cok
            acc = rng.random() < (p_c if ok_user else pw[i]); m["accepted"] += acc; m["acc_w"] += acc and not ok_user
            m["wrong"] += not ok; m["cwrong"] += not cok
            if acc and cond != "static" and not (cond == "cap1" and d >= 1):
                src[ns] = i; thr[ns] = root; dep[ns] = d + 1; evk[ns] = ev; gk[ns] = key
                if gi is None: has[ns] = False; m["ent_missing"] += 1
                else:
                    has[ns] = True
                    if r in DENSE: R.E[ns] = c[f"E_{variant}"][gi]
                    else: R.idx[ns] = gi
                if cond == "provdepth": R.penv[ns] = lam * c["iqr"] * (d + 1)
                self_on[ns] = False; fresh.append(ns); ns += 1
                m["ent_c" if ok else "ent_w"] += 1; m[f"ent_d{min(d + 1, 3)}"] += 1; m[f"ent_{'c' if ok else 'w'}_d{min(d + 1, 2)}"] += 1
                m["cent_c" if cok else "cent_w"] += 1
                if delay is not None:
                    if not ok_user and rng.random() < r_reopen: pending.append((rd + 1 + delay, ns - 1))
                    elif ok_user and false > 0 and rng.random() < false: pending.append((rd + 1 + delay, ns - 1)); m["false_reopen"] += 1
        rec = dict(n=m["n"], acc=m["c"] / m["n"], cacc=m["cc"] / m["n"], hit5=m["h5"] / m["n"], rep=m["rep"], rep_acc=m["rc"] / max(m["rep"], 1),
                   rep_cacc=m["rcc"] / max(m["rep"], 1), self_share=m["self"] / m["n"], self_d2=m["self_d2"] / m["n"],
                   accept_rate=m["accepted"] / m["n"], drift=[(drift[d] / dn[d]) if dn[d] else None for d in (0, 1, 2)], drift_n=[dn[0], dn[1], dn[2]],
                   entries=int(ns), active=int(self_on[:ns].sum()))
        for k in ("acc_w", "wrong", "cwrong", "wrong_self_d1", "wrong_self_d2", "cwrong_self_d1", "cwrong_self_d2", "cov", "capture", "capture_w",
                  "cov_ok", "steal", "ent_c", "ent_w", "cent_c", "cent_w", "ent_d1", "ent_d2", "ent_d3", "ent_missing", "ent_w_d1", "ent_w_d2",
                  "ent_c_d1", "ent_c_d2", "eff_c", "eff_w", "eff_c_d1", "eff_w_d1", "eff_c_d2", "eff_w_d2", "served_d1", "served_d2",
                  "cunk", "cnoref", "false_reopen"):
            rec[k] = m[k]
        rec["labels"] = {k[4:]: v for k, v in m.items() if k.startswith("lab_")}
        out.append(rec)
    bits = dict(okb=np.packbits(okb).tobytes().hex(), cokb=np.packbits(cokb).tobytes().hex(), sdep=sdep.tobytes().hex())
    return out, miss_keys, miss_posts, bits

REOPENS = ["reopen", "reopen_d1", "reopen_d2", "reopen_d1_f10", "reopen_d1_f20"]
def plan():
    P = [("askgen", "static", 0.0, 0.0, "root")]
    for v in VARIANTS:
        for pw in PW_GRID: P += [(v, "naive", pw, 0.0, "root"), (v, "reopen", pw, 0.0, "root")]
    for lam in PROV_LAM:
        for pw in PW_GRID: P.append(("askgen", "prov", pw, lam, "root"))
    for v in CTRL:
        for pw in PW_GRID: P.append((v, "naive", pw, 0.0, "root"))
    # content-aware user
    for v in VARIANTS + CTRL:
        for pw in PW_GRID: P.append((v, "naive", pw, 0.0, "content"))
    for cond in REOPENS + ["cap1", "regen"]:
        for pw in PW_GRID: P.append(("askgen", cond, pw, 0.0, "content"))
    for cond in ("prov", "provdepth"):
        for lam in PROV_LAM:
            for pw in PW_GRID: P.append(("askgen", cond, pw, lam, "content"))
    if os.environ.get("PLAN") == "fac":        # 2x2 prompt factorial: main design, content-aware user
        P = [("askgen", "static", 0.0, 0.0, "root")] + [("askgen", "naive", pw, 0.0, "content") for pw in PW_GRID]
    if os.environ.get("PLAN") == "long":       # long stream: main design, content-aware user, naive and delayed reopen
        P = [("askgen", "static", 0.0, 0.0, "root")] + [("askgen", c, pw, 0.0, "content") for c in ("naive", "reopen_d1") for pw in PW_GRID]
    if os.environ.get("PLAN") == "min":        # external-validity runs (other prompt): main design and user models only
        P = [("askgen", "static", 0.0, 0.0, "root")] + [(v, "naive", pw, 0.0, u) for v in ("askgen", "gen") for u in ("root", "content") for pw in PW_GRID] \
            + [("askgen", "reopen_d1", pw, 0.0, "content") for pw in PW_GRID]
    ov = os.environ.get("ONLY_VARIANT")
    return [x for x in P if x[0] == ov] if ov else P

def task(args):
    sub, r, seed, rounds = args
    c = getg(sub, r); n = len(c["afam"])
    order = np.random.default_rng(1000 + seed).permutation(n)
    res, mk, mp = [], set(), set()
    for v, cnd, pw, lam, user in plan():
        if v in CTRL and f"E_{v}" not in c: continue           # control embeddings exist only for the main generator
        cu, a, b, bits = run(sub, r, v, order, cnd, pw, rounds, seed, lam, user=user); mk |= a; mp |= b
        res.append(dict(sub=sub, ret=r, seed=seed, rounds=rounds, variant=v, user=user, cond=cnd, p_w=None if cnd == "static" else pw,
                        lam=lam, curve=cu, **bits))
    return sub, r, res, mk, mp

if __name__ == "__main__":
    rounds = int(sys.argv[1]); seeds = int(sys.argv[2]); subs = sys.argv[3].split(","); rets = sys.argv[4].split(","); tag = sys.argv[5]
    tasks = [(s, r, sd, rounds) for s in subs for r in rets for sd in range(seeds)]
    t = time.time(); allres = []; MK = collections.defaultdict(set); MP = collections.defaultdict(set)
    with Pool(int(os.environ.get("NPROC", 12))) as p:
        for k, (s, r, res, mk, mp) in enumerate(p.imap_unordered(task, tasks)):
            allres.extend(res); MK[s] |= mk; MP[s] |= mp
            if k % 10 == 0: print(k, len(tasks), f"{time.time()-t:.0f}s", flush=True)
    nk = npo = 0
    for s in subs:
        for name, ms in ((f"missing{TAG}_{s}.json", MK[s]), (f"missing_posts_{s}.json", MP[s])):
            mf = os.path.join(G, name); old = set(json.load(open(mf))) if os.path.exists(mf) else set()
            json.dump(sorted(old | ms), open(mf, "w"))
        nk += len(MK[s]); npo += len(MP[s])
    fn = os.path.join(RES, f"sim_gen_{tag}.json")
    json.dump(dict(runtime_s=time.time() - t, missing_keys=nk, missing_posts=npo, runs=allres), open(fn, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote", fn, f"{time.time()-t:.0f}s", "missing gen keys:", nk, "missing posts:", npo)

"""Paper-2 analysis, version 3 (TMLR revision): results/sim_gen_{RESTAG}{sub}.json -> results/tables_gen3{OUT}.json,
tex/tab3_*{OUT}.tex, tex/macros_gen3{OUT}.tex. Every number in the paper comes from here.

What is new against analyze_gen.py (version 1 of the preprint):
  * two user models (root label as paper 1; content-aware, judged), both reported;
  * effect sizes against the ceiling: questions static retrieval gets wrong that have an earlier correctly answered
    sibling in the stream (the only questions write-back can fix), share of that ceiling captured, answers changed per
    1,000 questions;
  * uncertainty from a joint bootstrap over orders AND duplicate families, plus a leave-largest-family-out check;
  * H1 source-bias control with content and length held fixed (askgen vs askhum, gen vs hum), paired tests;
  * H2 counterfactual harm by served depth (per-ask outcome bits), judged label mix of served answers by depth;
  * H3 depth-aware crossover estimate (depth-2 wrong entries extrapolated in proportion to p_w) against the first-order
    estimate of paper 1, out of sample (calibrate on 5 orders, test on the other 5, both directions), p_ref 0.3 and 0.5;
  * H4 realistic countermeasures (delayed and noisy reopen, depth cap, regenerate from root, provenance-aware penalty),
    paired tests against naive write-back with Holm correction over configurations.
  RESTAG=v3az_ python analyze_gen3.py android,unix,webmasters,mathematica
"""
import json, os, sys, collections, math
import numpy as np
from scipy import stats
from common import *
from analyze import stream, ci, crossover

TEX = os.path.join(HERE, "tex"); PC = 0.9
OUT = os.environ.get("OUTTAG", ""); RESTAG = os.environ.get("RESTAG", "v3az_")
NAME = {"bge": "bge", "minilm": "MiniLM", "bm25": "BM25"}
VNAME = {"gen": "generated answer", "askgen": "question + generated answer", "genthr": "generated answer + cited thread",
         "askhum": "question + human answer (control)", "hum": "human answer (control)"}
PWS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
SUBS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["android", "unix", "webmasters", "mathematica"]
RNG = np.random.default_rng(0); NBOOT = int(os.environ.get("NBOOT", 2000))

G = collections.defaultdict(dict); BITS = collections.defaultdict(dict); MISSING = {}
for s in SUBS:
    d = json.load(open(os.path.join(RES, f"sim_gen_{RESTAG}{s}.json"))); MISSING[s] = d["missing_keys"]
    for r in d["runs"]:
        k = (r["sub"], r["ret"], r["variant"], r["cond"], r["p_w"], r["lam"], r.get("user", "root"))
        G[k][r["seed"]] = r["curve"]; BITS[k][r["seed"]] = (r.get("okb"), r.get("cokb"), r.get("sdep"))
RETS = [r for r in RETRIEVERS if any(k[1] == r for k in G)]
SEEDS = sorted(next(iter(G.values())).keys()); CONFIGS = [(s, r) for s in SUBS for r in RETS]
HAS = lambda *k: k in G
T2 = json.load(open(os.path.join(RES, "tables_v2.json"))) if os.path.exists(os.path.join(RES, "tables_v2.json")) else None

# ---- per-ask data --------------------------------------------------------------------------------------------------
_META = {}
def meta(s, r):
    """family of every ask, order per seed, static outcome bits."""
    if (s, r) in _META: return _META[(s, r)]
    import sim_v2
    c = sim_v2.get(s, r); afam = np.asarray(c["afam"]); n = len(afam)
    fams, finv = np.unique(afam, return_inverse=True)
    orders = {sd: np.random.default_rng(1000 + sd).permutation(n) for sd in SEEDS}
    _META[(s, r)] = dict(n=n, afam=afam, finv=finv, nfam=len(fams), orders=orders); return _META[(s, r)]

def unbits(h, n): return np.unpackbits(np.frombuffer(bytes.fromhex(h), np.uint8))[:n].astype(bool)
def outcome(k, sd, content):
    n = meta(k[0], k[1])["n"]; ob, cb, sdp = BITS[k][sd]
    return unbits(cb if content else ob, n)
def served_depth(k, sd):
    n = meta(k[0], k[1])["n"]; return np.frombuffer(bytes.fromhex(BITS[k][sd][2]), np.uint8)[:n]
def skey(s, r): return (s, r, "askgen", "static", None, 0.0, "root")
def rkey(s, r, v, cond, p, lam=0.0, user="root"): return (s, r, v, cond, p, lam, user)
def ckey(user): return "cacc" if user == "content" else "acc"

def delta(s, r, v, cond, p, lam=0.0, user="root", seeds=None, rep=False):
    seeds = seeds or SEEDS; st = G[skey(s, r)]; runs = G[rkey(s, r, v, cond, p, lam, user)]
    key = ("rep_" if rep else "") + ckey(user)
    return [100 * (stream(runs[sd], key) - stream(st[sd], key)) for sd in seeds]

def cross(s, r, v, cond="naive", lam=0.0, user="root", seeds=None, boot=True):
    seeds = seeds or SEEDS
    M = np.array([delta(s, r, v, cond, p, lam, user, seeds) for p in PWS]); x, kind = crossover(PWS, M.mean(1))
    if not boot: return dict(p=float(x), kind=kind)
    b = [crossover(PWS, M[:, RNG.integers(0, len(seeds), len(seeds))].mean(1))[0] for _ in range(1000)]
    return dict(p=float(x), kind=kind, lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)))

def famdiff(s, r, v, cond, p, lam=0.0, user="root"):
    """per-seed family sums of (run - static) outcome, shape (seeds, families), and family sizes."""
    m = meta(s, r); content = user == "content"; S = []
    for sd in SEEDS:
        d = outcome(rkey(s, r, v, cond, p, lam, user), sd, content).astype(float) - outcome(skey(s, r), sd, content).astype(float)
        S.append(np.bincount(m["finv"], weights=d, minlength=m["nfam"]))
    return np.array(S), np.bincount(m["finv"], minlength=m["nfam"]).astype(float)

def boot_delta(s, r, v, cond, p, lam=0.0, user="root", drop_largest=False):
    """Delta (points) with a 95% interval from a joint bootstrap over orders and duplicate families."""
    S, nf = famdiff(s, r, v, cond, p, lam, user)
    if drop_largest: j = int(np.argmax(nf)); S = np.delete(S, j, 1); nf = np.delete(nf, j)
    est = 100 * S.mean(0).sum() / nf.sum(); F = len(nf); B = []
    for _ in range(NBOOT):
        w = RNG.multinomial(F, np.full(F, 1 / F)); sd = RNG.integers(0, len(SEEDS), len(SEEDS))
        B.append(100 * (S[sd].mean(0) * w).sum() / (nf * w).sum())
    return dict(est=float(est), lo=float(np.percentile(B, 2.5)), hi=float(np.percentile(B, 97.5)))

def ceiling(s, r, v, cond, p, lam=0.0, user="root"):
    """eligible = static wrong with an earlier statically-correct sibling in the stream; risk = static correct.
    Returns shares averaged over orders: eligible (% of asks), captured (% of eligible fixed), lost (% of risk broken),
    other (% of non-eligible static-wrong asks fixed), net answers changed per 1,000 asks."""
    m = meta(s, r); content = user == "content"; R = []
    for sd in SEEDS:
        st = outcome(skey(s, r), sd, content); ru = outcome(rkey(s, r, v, cond, p, lam, user), sd, content)
        seen_ok = set(); el = np.zeros(m["n"], bool)
        for i in m["orders"][sd]:
            f = m["afam"][i]
            if not st[i] and f in seen_ok: el[i] = True
            if st[i]: seen_ok.add(f)
        other = ~st & ~el
        R.append((el.mean() * 100, 100 * ru[el].mean() if el.any() else np.nan, 100 * (~ru[st]).mean() if st.any() else np.nan,
                  100 * ru[other].mean() if other.any() else np.nan, 1000 * (ru.astype(int) - st.astype(int)).sum() / m["n"]))
    R = np.array(R); return dict(eligible=float(R[:, 0].mean()), captured=float(np.nanmean(R[:, 1])), lost=float(np.nanmean(R[:, 2])),
                                 other=float(np.nanmean(R[:, 3])), per1000=float(R[:, 4].mean()))

def harm_by_depth(s, r, v, cond, p, lam=0.0, user="content"):
    """counterfactual change in correct answers per 1,000 asks attributable to answers served from an entry of depth
    1 / 2 / 3+ (sum over asks served at that depth of run outcome minus static outcome)."""
    k = rkey(s, r, v, cond, p, lam, user); m = meta(s, r); content = user == "content"; H = np.zeros(4)
    for sd in SEEDS:
        dd = served_depth(k, sd); diff = outcome(k, sd, content).astype(int) - outcome(skey(s, r), sd, content).astype(int)
        for d in (1, 2, 3): H[d] += diff[dd == d].sum()
    return [float(1000 * H[d] / (m["n"] * len(SEEDS))) for d in (1, 2, 3)]

def harm_boot(s, r, v, cond, p, lam=0.0, user="content"):
    """harm by served depth with draws from a joint bootstrap over orders and families; returns (est[3], draws[NBOOT,3])."""
    k = rkey(s, r, v, cond, p, lam, user); m = meta(s, r); content = user == "content"; S = np.zeros((len(SEEDS), 3, m["nfam"]))
    for si, sd in enumerate(SEEDS):
        dd = served_depth(k, sd); diff = outcome(k, sd, content).astype(float) - outcome(skey(s, r), sd, content).astype(float)
        for j, d in enumerate((1, 2, 3)): S[si, j] = np.bincount(m["finv"], weights=diff * (dd == d), minlength=m["nfam"])
    nf = np.bincount(m["finv"], minlength=m["nfam"]).astype(float); F = m["nfam"]
    est = 1000 * S.mean(0).sum(1) / nf.sum(); D = np.zeros((NBOOT, 3))
    for b in range(NBOOT):
        w = RNG.multinomial(F, np.full(F, 1 / F)); sd = RNG.integers(0, len(SEEDS), len(SEEDS))
        D[b] = 1000 * (S[sd].mean(0) * w).sum(1) / (nf * w).sum()
    return est, D

def tot(runs, key, seeds=None, lo=0, hi=None):
    seeds = seeds or SEEDS; return sum(x.get(key, 0) for sd in seeds for x in runs[sd][lo:hi])
def ratio(runs, num, den, seeds=None):
    a = tot(runs, num, seeds, lo=5); b = tot(runs, den, seeds, lo=5); return a / b if b else float("nan")
def per_seed_ratio(runs, num, den): return [ratio({0: runs[sd]}, num, den, [0]) for sd in SEEDS]

def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(o): run = max(run, min(1.0, (m - rank) * ps[i])); adj[i] = run
    return adj
def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0): return float(d.mean()), 1.0
    return float(d.mean()), float(stats.ttest_rel(a, b).pvalue)

T = dict(meta=dict(subs=SUBS, rets=RETS, seeds=len(SEEDS), missing=MISSING, restag=RESTAG), base={}, effect={}, h1={}, h1ctrl={},
         h2={}, h3={}, h4={}, users={}, reopen_equiv={})

# ---- base and effect sizes -----------------------------------------------------------------------------------------
for s, r in CONFIGS:
    st = G[skey(s, r)]
    T["base"][f"{s}|{r}"] = {k: float(np.mean([stream(st[sd], k) for sd in SEEDS])) for k in ("acc", "cacc", "hit5")}
    for user in ("root", "content"):
        if not HAS(*rkey(s, r, "askgen", "naive", 0.95, 0.0, user)): continue
        e = {}
        for p in (0.1, 0.95):
            e[f"d{p}"] = boot_delta(s, r, "askgen", "naive", p, 0.0, user)
            e[f"d{p}_nolargest"] = boot_delta(s, r, "askgen", "naive", p, 0.0, user, drop_largest=True)
            e[f"rep{p}"] = ci(delta(s, r, "askgen", "naive", p, 0.0, user, rep=True))
            e[f"ceil{p}"] = ceiling(s, r, "askgen", "naive", p, 0.0, user)
        T["effect"][f"{s}|{r}|{user}"] = e

# ---- H1: entry designs (root and content users) and the same-content source-bias control -------------------------
DESIGNS = ["gen", "askgen", "genthr", "askhum", "hum"]
for s, r in CONFIGS:
    for user in ("root", "content"):
        for v in DESIGNS:
            if not HAS(*rkey(s, r, v, "naive", 0.95, 0.0, user)): continue
            run95 = G[rkey(s, r, v, "naive", 0.95, 0.0, user)]
            d = dict(cross=cross(s, r, v, "naive", 0.0, user), **{f"d{p}": ci(delta(s, r, v, "naive", p, 0.0, user)) for p in PWS})
            d["auc"] = float(np.mean([np.mean(delta(s, r, v, "naive", p, 0.0, user)) for p in PWS]))
            d["self95"] = 100 * float(np.mean([stream(run95[sd], "self_share", lo=5) for sd in SEEDS]))
            d["steal95"] = 100 * ratio(run95, "steal", "cov_ok"); d["capture95"] = 100 * ratio(run95, "capture", "cov")
            d["capw95"] = 100 * ratio(run95, "capture_w", "cov")
            T["h1"][f"{s}|{r}|{v}|{user}"] = d
    if T2:
        for v in ("ask", "askdoc"):
            k = f"{s}|{r}|{v}"
            if k in T2["abl"]: T["h1"][f"{s}|{r}|p1_{v}|root"] = dict(cross=T2["abl"][k]["naive"]["cross"], **{f"d{p}": T2["abl"][k]["naive"][f"d{p}"] for p in PWS},
                                                                     self95=T2["abl"][k]["self95"], capture95=T2["abl"][k]["capture95"], steal95=T2["abl"][k]["steal95"])
# paired control: generated vs human text with the same question, root and length
for user in ("root", "content"):
    for gv, hv in (("askgen", "askhum"), ("gen", "hum")):
        rows = []
        for s, r in CONFIGS:
            if not HAS(*rkey(s, r, hv, "naive", 0.95, 0.0, user)): continue
            a = G[rkey(s, r, gv, "naive", 0.95, 0.0, user)]; b = G[rkey(s, r, hv, "naive", 0.95, 0.0, user)]
            sa, sb = per_seed_ratio(a, "steal", "cov_ok"), per_seed_ratio(b, "steal", "cov_ok")
            ca, cb = per_seed_ratio(a, "capture", "cov"), per_seed_ratio(b, "capture", "cov")
            da, db = delta(s, r, gv, "naive", 0.95, 0.0, user), delta(s, r, hv, "naive", 0.95, 0.0, user)
            ms, ps = paired(np.array(sa) * 100, np.array(sb) * 100); mc, pc = paired(np.array(ca) * 100, np.array(cb) * 100); md, pd = paired(da, db)
            rows.append(dict(cfg=f"{s}|{r}", ret=r, steal=ms, p_steal=ps, capture=mc, p_capture=pc, delta=md, p_delta=pd))
        if not rows: continue
        for key in ("steal", "capture", "delta"):
            adj = holm([x[f"p_{key}"] for x in rows])
            for x, q in zip(rows, adj): x[f"q_{key}"] = float(q)
        summ = {}
        for r in RETS:
            R = [x for x in rows if x["ret"] == r]
            if R: summ[r] = {k: dict(mean=float(np.mean([x[k] for x in R])), pos=sum(x[k] > 0 for x in R), sig_pos=sum(x[k] > 0 and x[f"q_{k}"] < 0.05 for x in R),
                                     sig_neg=sum(x[k] < 0 and x[f"q_{k}"] < 0.05 for x in R), n=len(R)) for k in ("steal", "capture", "delta")}
        T["h1ctrl"][f"{gv}|{hv}|{user}"] = dict(rows=rows, by_ret=summ)

# ---- user models: how judged acceptance changes the outcome ---------------------------------------------------------
for s, r in CONFIGS:
    if HAS(*rkey(s, r, "askgen", "naive", 0.95, 0.0, "content")):
        T["users"][f"{s}|{r}"] = {u: dict(cross=cross(s, r, "askgen", "naive", 0.0, u), d95=ci(delta(s, r, "askgen", "naive", 0.95, 0.0, u)),
                                          d01=ci(delta(s, r, "askgen", "naive", 0.1, 0.0, u))) for u in ("root", "content")}

# ---- H2: depth ------------------------------------------------------------------------------------------------------
LABS = ("YES", "PARTLY", "NO", "ABSTAIN", "NOREF")
for s, r in CONFIGS:
    for p in (0.3, 0.95):
        k = rkey(s, r, "askgen", "naive", p, 0.0, "content")
        if not HAS(*k): continue
        runs = G[k]; rounds = len(runs[SEEDS[0]])
        lab = collections.Counter()
        for sd in SEEDS:
            for x in runs[sd]: lab.update(x.get("labels", {}))
        mix = {d: {l: lab.get(f"{l}_d{d}", 0) for l in LABS} for d in (0, 1, 2)}
        dr = collections.defaultdict(float); dn = collections.Counter()
        for sd in SEEDS:
            for x in runs[sd]:
                for j in range(3):
                    if x["drift"][j] is not None: dr[j] += x["drift"][j] * x["drift_n"][j]; dn[j] += x["drift_n"][j]
        T["h2"][f"{s}|{r}|{p}"] = dict(
            ent_d=[tot(runs, f"ent_d{j}") for j in (1, 2, 3)], ent_w_d=[tot(runs, "ent_w_d1"), tot(runs, "ent_w_d2")],
            self_by_round=[float(np.mean([runs[sd][j]["self_share"] for sd in SEEDS])) for j in range(rounds)],
            self_d2_by_round=[float(np.mean([runs[sd][j]["self_d2"] for sd in SEEDS])) for j in range(rounds)],
            cwrong=tot(runs, "cwrong"), cwrong_self=[tot(runs, "cwrong_self_d1"), tot(runs, "cwrong_self_d2")],
            harm=harm_by_depth(s, r, "askgen", "naive", p, 0.0, "content"), labels=mix,
            drift=[(dr[j] / dn[j]) if dn[j] else None for j in range(3)], drift_n=[dn[j] for j in range(3)])
# pooled harm by depth with intervals: per-configuration bootstrap draws averaged over configurations
HB = [harm_boot(s, r, "askgen", "naive", 0.95) for s, r in CONFIGS if HAS(*rkey(s, r, "askgen", "naive", 0.95, 0.0, "content"))]
if HB:
    D = np.mean([d for _, d in HB], 0); E = np.mean([e for e, _ in HB], 0)
    T["h2_harm_pool"] = dict(est=E.tolist(), lo=np.percentile(D, 2.5, 0).tolist(), hi=np.percentile(D, 97.5, 0).tolist(),
                             per_cfg=[dict(est=e.tolist(), lo=np.percentile(d, 2.5, 0).tolist(), hi=np.percentile(d, 97.5, 0).tolist()) for e, d in HB])
# judged YES share by served depth per configuration (each configuration one unit), Wilcoxon depth 0 vs depth >= 2
ys = []
for s, r in CONFIGS:
    k = f"{s}|{r}|0.95"
    if k not in T["h2"]: continue
    L_ = T["h2"][k]["labels"]; sh = []
    for d in (0, 2):
        n_ = sum(L_[d].get(x, 0) for x in ("YES", "PARTLY", "NO", "ABSTAIN")); sh.append(L_[d].get("YES", 0) / n_ if n_ else np.nan)
    ys.append(sh)
ys = np.array(ys); ok_ = ~np.isnan(ys).any(1)
T["h2_cfg_test"] = dict(n=int(ok_.sum()), lower=int((ys[ok_, 1] < ys[ok_, 0]).sum()), mean0=float(np.nanmean(ys[:, 0])), mean2=float(np.nanmean(ys[:, 1])),
                        p=float(stats.wilcoxon(ys[ok_, 0], ys[ok_, 1]).pvalue) if ok_.sum() > 5 else None)
# pooled label mix by served depth (content user, askgen naive, both p)
pool = {p: {d: collections.Counter() for d in (0, 1, 2)} for p in (0.3, 0.95)}
for key, v in T["h2"].items():
    p = float(key.split("|")[2])
    for d in (0, 1, 2): pool[p][d].update(v["labels"][d])
T["h2_pool"] = {str(p): {str(d): dict(pool[p][d]) for d in (0, 1, 2)} for p in pool}
# trend test: share of served answers judged YES by depth (pooled, p = 0.95), chi-square on YES vs not, depth 0 vs 2+
y0 = pool[0.95][0]["YES"]; n0 = sum(pool[0.95][0][l] for l in ("YES", "PARTLY", "NO", "ABSTAIN"))
y2 = pool[0.95][2]["YES"]; n2 = sum(pool[0.95][2][l] for l in ("YES", "PARTLY", "NO", "ABSTAIN"))
y1 = pool[0.95][1]["YES"]; n1 = sum(pool[0.95][1][l] for l in ("YES", "PARTLY", "NO", "ABSTAIN"))
T["h2_test"] = dict(yes=[y0 / max(n0, 1), y1 / max(n1, 1), y2 / max(n2, 1)], n=[n0, n1, n2],
                    p_d0_d2=float(stats.chi2_contingency([[y0, n0 - y0], [y2, n2 - y2]])[1]) if n2 else None)

# ---- H3: crossover estimates, out of sample -------------------------------------------------------------------------
def plug(s, r, v, pref, seeds, depth=False):
    runs = G[rkey(s, r, v, "naive", pref)]
    A = float(np.mean([stream(G[skey(s, r)][sd], "acc") for sd in seeds]))
    ent = lambda k: sum(x.get(k, 0) for sd in seeds for x in runs[sd][:-1])
    eff = lambda k: sum(x.get(k, 0) for sd in seeds for x in runs[sd])
    ec = eff("eff_c") / max(ent("ent_c"), 1); ew = eff("eff_w") / max(ent("ent_w"), 1)
    if not depth:
        return (-PC * A * ec / ((1 - A) * ew)) if ew < 0 else None
    n1, n2 = ent("ent_w_d1"), ent("ent_w_d2"); e1 = eff("eff_w_d1") / max(n1, 1); e2 = eff("eff_w_d2") / max(n2, 1) if n2 else 0.0
    rho = n2 / max(n1, 1)
    # wrong-entry effect W(p) = p (1-A)/(1+rho) [e1 + rho (p/pref) e2]; equals the first-order value at p = pref
    a = (1 - A) * rho * e2 / ((1 + rho) * pref); b = (1 - A) * e1 / (1 + rho); c0 = PC * A * ec
    if abs(a) < 1e-12: return (-c0 / b) if b < 0 else None
    disc = b * b - 4 * a * c0
    if disc < 0: return None
    roots = [(-b + sgn * math.sqrt(disc)) / (2 * a) for sgn in (1, -1)]
    roots = [x for x in roots if x > 0]
    return min(roots) if roots else None
def kind_of(p): return None if p is None else ("<" if p < PWS[0] else (">" if p > PWS[-1] else "="))
for v in ("askgen", "gen", "genthr"):
    for pref in (0.3, 0.5):
        rows = []
        for cal, ev in ((SEEDS[:5], SEEDS[5:]), (SEEDS[5:], SEEDS[:5])):
            for s, r in CONFIGS:
                if not HAS(*rkey(s, r, v, "naive", pref)): continue
                ob = cross(s, r, v, "naive", 0.0, "root", seeds=ev, boot=False)
                p1 = plug(s, r, v, pref, cal); p2 = plug(s, r, v, pref, cal, depth=True)
                rows.append(dict(cfg=f"{s}|{r}", obs=ob["p"], ko=ob["kind"], p1=p1, p2=p2, k1=kind_of(p1), k2=kind_of(p2)))
        if not rows: continue
        res = {}
        for tag in ("p1", "p2"):
            both = [x for x in rows if x["ko"] == "=" and x[f"k{tag[1]}"] == "="]
            if not both: res[tag] = None; continue
            o = np.array([x["obs"] for x in both]); pr = np.array([x[tag] for x in both]); err = np.abs(o - pr)
            bs = [np.mean(err[RNG.integers(0, len(err), len(err))]) for _ in range(2000)]
            res[tag] = dict(n=len(both), mae=float(err.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                            spearman=float(stats.spearmanr(o, pr)[0]) if len(both) > 2 else None,
                            same_class=sum(x["ko"] == x[f"k{tag[1]}"] for x in rows if x[f"k{tag[1]}"] is not None),
                            n_defined=sum(x[f"k{tag[1]}"] is not None for x in rows), n_rows=len(rows))
        both = [x for x in rows if x["ko"] == "=" and x["k1"] == "=" and x["k2"] == "="]
        if both:
            o = np.array([x["obs"] for x in both]); e1 = np.abs(o - np.array([x["p1"] for x in both])); e2 = np.abs(o - np.array([x["p2"] for x in both]))
            res["paired"] = dict(n=len(both), mae1=float(e1.mean()), mae2=float(e2.mean()), p=float(stats.wilcoxon(e1, e2).pvalue) if len(both) > 5 and not np.allclose(e1, e2) else None,
                                 mean_baseline=float(np.mean(np.abs(o - o.mean()))))
        T["h3"][f"{v}|{pref}"] = dict(rows=rows, **res)

# ---- H4: countermeasures, content-aware user -------------------------------------------------------------------------
RULES = [("naive", 0.0), ("reopen", 0.0), ("reopen_d1", 0.0), ("reopen_d2", 0.0), ("reopen_d1_f10", 0.0), ("reopen_d1_f20", 0.0),
         ("cap1", 0.0), ("regen", 0.0), ("prov", 0.5), ("prov", 1.0), ("provdepth", 0.5), ("provdepth", 1.0)]
for s, r in CONFIGS:
    d = {}
    for cond, lam in RULES:
        if not HAS(*rkey(s, r, "askgen", cond, 0.95, lam, "content")): continue
        nm = f"{cond}{lam if lam else ''}"
        d[nm] = {f"d{p}": ci(delta(s, r, "askgen", cond, p, lam, "content")) for p in (0.1, 0.6, 0.95)}
        d[nm]["cross"] = cross(s, r, "askgen", cond, lam, "content")
        d[nm]["auc"] = float(np.mean([np.mean(delta(s, r, "askgen", cond, p, lam, "content")) for p in PWS]))
        if cond != "naive":
            m95, p95 = paired(delta(s, r, "askgen", cond, 0.95, lam, "content"), delta(s, r, "askgen", "naive", 0.95, 0.0, "content"))
            m01, p01 = paired(delta(s, r, "askgen", cond, 0.1, lam, "content"), delta(s, r, "askgen", "naive", 0.1, 0.0, "content"))
            d[nm].update(vs95=m95, p95=p95, vs01=m01, p01=p01)
    T["h4"][f"{s}|{r}"] = d
H4S = {}
for cond, lam in RULES[1:]:
    nm = f"{cond}{lam if lam else ''}"
    cf = [(s, r) for s, r in CONFIGS if nm in T["h4"].get(f"{s}|{r}", {})]
    if not cf: continue
    q95 = holm([T["h4"][f"{s}|{r}"][nm]["p95"] for s, r in cf]); q01 = holm([T["h4"][f"{s}|{r}"][nm]["p01"] for s, r in cf])
    X = [T["h4"][f"{s}|{r}"][nm] for s, r in cf]
    H4S[nm] = dict(n=len(cf), d01=float(np.mean([x["d0.1"][0] for x in X])), d95=float(np.mean([x["d0.95"][0] for x in X])),
                   auc=float(np.mean([x["auc"] for x in X])), safe=sum(x["cross"]["kind"] == ">" for x in X),
                   safe_lb=sum(x["cross"]["kind"] == ">" and x["cross"].get("lo", 0) >= 0.95 for x in X),
                   better95=sum(x["vs95"] > 0 and q < 0.05 for x, q in zip(X, q95)), worse95=sum(x["vs95"] < 0 and q < 0.05 for x, q in zip(X, q95)),
                   worse01=sum(x["vs01"] < 0 and q < 0.05 for x, q in zip(X, q01)))
Xn = [T["h4"][f"{s}|{r}"]["naive"] for s, r in CONFIGS if "naive" in T["h4"].get(f"{s}|{r}", {})]
if Xn: H4S["naive"] = dict(n=len(Xn), d01=float(np.mean([x["d0.1"][0] for x in Xn])), d95=float(np.mean([x["d0.95"][0] for x in Xn])),
                           auc=float(np.mean([x["auc"] for x in Xn])), safe=sum(x["cross"]["kind"] == ">" for x in Xn),
                           safe_lb=sum(x["cross"]["kind"] == ">" and x["cross"].get("lo", 0) >= 0.95 for x in Xn))
T["h4_summary"] = H4S
# reopen (oracle, paper 1) equals naive write-back at half the acceptance rate (root user)
errs = []
for s, r in CONFIGS:
    if not HAS(*rkey(s, r, "askgen", "reopen", 0.95)): continue
    nv = [np.mean(delta(s, r, "askgen", "naive", p)) for p in PWS]
    for p in PWS:
        if p / 2 >= PWS[0]: errs.append(abs(np.mean(delta(s, r, "askgen", "reopen", p)) - float(np.interp(p / 2, PWS, nv))))
T["reopen_equiv"] = dict(mean_abs=float(np.mean(errs)) if errs else None, n=len(errs))

json.dump(T, open(os.path.join(RES, f"tables_gen3{OUT}.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("wrote tables_gen3", OUT, "configs", len(CONFIGS), "missing", MISSING)

"""Paper-2 analysis: results/sim_gen_{sub}.json -> results/tables_gen.json, tex/*_gen.tex, tex/macros_gen.tex.
Conventions as analyze_v2 (paired deltas vs static over seeds, t-interval, crossover by linear interpolation, bootstrap
over seeds). Paper-1 reference numbers (ask and ask+thread entries) are read from results/tables_v2.json, never retyped.

  python analyze_gen.py android,unix,webmasters,mathematica
"""
import json, os, sys, collections
import numpy as np
from common import *
from analyze import stream, ci, crossover
TEX = os.path.join(HERE, "tex"); PC = 0.9
OUT = os.environ.get("OUTTAG", "")   # suffix for output files, e.g. _local for the open-weight replication
NAME = {"bge": "bge", "minilm": "MiniLM", "bm25": "BM25"}
VNAME = {"gen": "generated answer", "askgen": "ask + generated answer", "genthr": "generated answer + cited thread"}
VARIANTS = ["gen", "askgen", "genthr"]; PWS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]; PROV_LAM = [0.5, 1.0]
SUBS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["android", "unix", "webmasters", "mathematica"]
G = collections.defaultdict(dict); MISSING = {}
for s in SUBS:
    d = json.load(open(os.path.join(RES, f"sim_gen_{os.environ.get('RESTAG', '')}{s}.json"))); MISSING[s] = d["missing_keys"]
    for r in d["runs"]: G[(r["sub"], r["ret"], r["variant"], r["cond"], r["p_w"], r["lam"])][r["seed"]] = r["curve"]
SEEDS = sorted(next(iter(G.values())).keys()); CONFIGS = [(s, r) for s in SUBS for r in RETRIEVERS]
T2 = json.load(open(os.path.join(RES, "tables_v2.json"))) if os.path.exists(os.path.join(RES, "tables_v2.json")) else None
rng = np.random.default_rng(0)

def static(s, r): return G[(s, r, "askgen", "static", None, 0.0)]
def delta(s, r, v, cond, p, lam=0.0, key="acc", seeds=SEEDS):
    st = static(s, r); runs = G[(s, r, v, cond, p, lam)]
    return [100 * (stream(runs[sd], key) - stream(st[sd], key)) for sd in seeds]
def cross(s, r, v, cond="naive", lam=0.0, seeds=SEEDS):
    M = np.array([delta(s, r, v, cond, p, lam, seeds=seeds) for p in PWS]); x, kind = crossover(PWS, M.mean(1))
    b = [crossover(PWS, M[:, rng.integers(0, len(seeds), len(seeds))].mean(1))[0] for _ in range(1000)]
    return dict(p=float(x), kind=kind, lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)))
def half(runs, key, seeds=SEEDS): return float(np.mean([stream(runs[sd], key, lo=len(runs[sd]) // 2) for sd in seeds]))
def tot(runs, key, seeds=SEEDS, lo=0): return sum(x[key] for sd in seeds for x in runs[sd][lo:])
def ratio(runs, num, den, seeds=SEEDS):
    a = tot(runs, num, seeds, lo=5); b = tot(runs, den, seeds, lo=5); return a / b if b else float("nan")
def plug(s, r, v, pref, seeds=SEEDS, order=1):
    """first-order plug-in estimate of p* (paper 1, eq. in sec. model); order=2 splits the wrong-entry effect by depth and
    weights depth-2+ entries by their observed share of served self evidence."""
    runs = G[(s, r, v, "naive", pref, 0.0)]
    A = float(np.mean([stream(static(s, r)[sd], "acc") for sd in seeds]))
    ec = tot(runs, "eff_c", seeds) / max(sum(x["ent_c"] for sd in seeds for x in runs[sd][:-1]), 1)
    if order == 1:
        ew = tot(runs, "eff_w", seeds) / max(sum(x["ent_w"] for sd in seeds for x in runs[sd][:-1]), 1)
    else:
        n1, n2 = tot(runs, "served_d1", seeds), tot(runs, "served_d2", seeds)
        e1 = tot(runs, "eff_w_d1", seeds); e2 = tot(runs, "eff_w_d2", seeds)
        ew = (e1 + e2) / max(sum(x["ent_w"] for sd in seeds for x in runs[sd][:-1]), 1)
        ew = ew * (1 + n2 / max(n1, 1))          # deeper entries multiply the exposure of a wrong write-back
    return (-PC * A * ec / ((1 - A) * ew)) if ew < 0 else None

T = dict(base={}, h1={}, h2={}, h3={}, h4={}, missing=MISSING, seeds=len(SEEDS))
for s, r in CONFIGS:
    T["base"][f"{s}|{r}"] = {k: float(np.mean([stream(static(s, r)[sd], k) for sd in SEEDS])) for k in ("acc", "hit5")}

# H1: source bias. crossover and capture per entry variant, beside paper-1 ask / ask+thread entries
for s, r in CONFIGS:
    for v in VARIANTS:
        d = {}
        for cond in ("naive", "reopen"):
            d[cond] = dict(cross=cross(s, r, v, cond), **{f"d{p}": ci(delta(s, r, v, cond, p)) for p in PWS})
        run95 = G[(s, r, v, "naive", 0.95, 0.0)]
        d["self95"] = 100 * half(run95, "self_share"); d["capture95"] = 100 * ratio(run95, "capture", "cov")
        d["capw95"] = 100 * ratio(run95, "capture_w", "cov"); d["steal95"] = 100 * ratio(run95, "steal", "cov_ok")
        T["h1"][f"{s}|{r}|{v}"] = d
    if T2:
        for v in ("ask", "askdoc"):
            k = f"{s}|{r}|{v}"
            if k in T2["abl"]: T["h1"][f"{s}|{r}|p1_{v}"] = dict(naive=dict(cross=T2["abl"][k]["naive"]["cross"], **{"d0.95": T2["abl"][k]["naive"]["d0.95"], "d0.3": T2["abl"][k]["naive"]["d0.3"]}),
                                                                 self95=T2["abl"][k]["self95"], capture95=T2["abl"][k]["capture95"], steal95=T2["abl"][k]["steal95"])

# H2: compounding. entries by depth, self evidence of depth >= 2 by round, attribution of wrong answers, drift by depth
for s, r in CONFIGS:
    for p in (0.3, 0.6, 0.95):
        runs = G[(s, r, "askgen", "naive", p, 0.0)]
        rounds = len(runs[SEEDS[0]])
        d = dict(ent_d=[tot(runs, f"ent_d{k}") for k in (1, 2, 3)],
                 self_d2_by_round=[float(np.mean([runs[sd][k]["self_d2"] for sd in SEEDS])) for k in range(rounds)],
                 self_by_round=[float(np.mean([runs[sd][k]["self_share"] for sd in SEEDS])) for k in range(rounds)],
                 wrong_self_d1=tot(runs, "wrong_self_d1"), wrong_self_d2=tot(runs, "wrong_self_d2"), wrong=tot(runs, "wrong"))
        dr = collections.defaultdict(float); dn = collections.Counter()
        for sd in SEEDS:
            for x in runs[sd]:
                for k in range(3):
                    if x["drift"][k] is not None: dr[k] += x["drift"][k] * x["drift_n"][k]; dn[k] += x["drift_n"][k]
        d["drift"] = [(dr[k] / dn[k]) if dn[k] else None for k in range(3)]; d["drift_n"] = [dn[k] for k in range(3)]
        T["h2"][f"{s}|{r}|{p}"] = d

# H3: plug-in estimate, first order vs depth-aware, against the observed crossover (askgen, naive)
for s, r in CONFIGS:
    for v in VARIANTS:
        obs = T["h1"][f"{s}|{r}|{v}"]["naive"]["cross"]
        T["h3"][f"{s}|{r}|{v}"] = dict(obs=obs, plug1=plug(s, r, v, 0.6), plug2=plug(s, r, v, 0.6, order=2))

# H4: countermeasures at p_w = 0.95 and 0.6, with the gain they keep at p_w = 0.1
for s, r in CONFIGS:
    d = {}
    for name, (v, cond, lam) in {"naive": ("askgen", "naive", 0.0), "reopen": ("askgen", "reopen", 0.0), "cite": ("genthr", "naive", 0.0),
                                 **{f"prov{lam}": ("askgen", "prov", lam) for lam in PROV_LAM}}.items():
        d[name] = {f"d{p}": ci(delta(s, r, v, cond, p, lam)) for p in (0.1, 0.6, 0.95)}
        d[name]["cross"] = cross(s, r, v, cond, lam)
    T["h4"][f"{s}|{r}"] = d

json.dump(T, open(os.path.join(RES, f"tables_gen{OUT}.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

# ---- LaTeX ---------------------------------------------------------------------------------------------------------
def cx(c): return {"<": r"$<$0.10", ">": r"$>$0.95"}.get(c["kind"], f"{c['p']:.2f}")
def pm(v): return f"{v[0]:+.2f} $\\pm$ {v[1]:.2f}"
def mac(n, v): return f"\\newcommand{{\\{n}{OUT.strip('_').capitalize()}}}{{{v}}}"   # _local -> \\nameLocal
lines = [r"\begin{tabular}{lllrrrr}", r"\toprule", r"forum & retriever & entry & $p^*$ & $\Delta$ at 0.95 & self 0.95 & steal 0.95 \\", r"\midrule"]
for s, r in CONFIGS:
    for v in (["p1_ask", "p1_askdoc"] if T2 else []) + VARIANTS:
        k = f"{s}|{r}|{v}"
        if k not in T["h1"]: continue
        d = T["h1"][k]; nm = {"p1_ask": "ask text (paper 1)", "p1_askdoc": "ask + thread (paper 1)"}.get(v) or VNAME[v]
        lines.append(f"{s} & {NAME[r]} & {nm} & {cx(d['naive']['cross'])} & {pm(d['naive']['d0.95'])} & {d['self95']:.0f}\\% & {d['steal95']:.0f}\\% \\\\")
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"; lines.append(r"\end{tabular}")
open(os.path.join(TEX, f"tab_h1_gen{OUT}.tex"), "w").write("\n".join(lines) + "\n")

lines = [r"\begin{tabular}{llrrrrrrr}", r"\toprule", r"forum & retriever & $p_w$ & entries d1/d2/d3+ & self d$\geq$2 (last round) & wrong from d1 / d$\geq$2 & drift d0 / d1 / d2+ \\", r"\midrule"]
for s, r in CONFIGS:
    for p in (0.3, 0.95):
        d = T["h2"][f"{s}|{r}|{p}"]; dr = " / ".join("--" if x is None else f"{x:.2f}" for x in d["drift"])
        lines.append(f"{s} & {NAME[r]} & {p} & {d['ent_d'][0]}/{d['ent_d'][1]}/{d['ent_d'][2]} & {100*d['self_d2_by_round'][-1]:.1f}\\% & "
                     f"{100*d['wrong_self_d1']/max(d['wrong'],1):.1f}\\% / {100*d['wrong_self_d2']/max(d['wrong'],1):.1f}\\% & {dr} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(TEX, f"tab_h2_gen{OUT}.tex"), "w").write("\n".join(lines) + "\n")

lines = [r"\begin{tabular}{lllrrr}", r"\toprule", r"forum & retriever & entry & observed $p^*$ & first-order estimate & depth-aware estimate \\", r"\midrule"]
err1, err2 = [], []
for s, r in CONFIGS:
    for v in VARIANTS:
        d = T["h3"][f"{s}|{r}|{v}"]; f = lambda x: "--" if x is None else f"{min(x, 1.0):.2f}"
        lines.append(f"{s} & {NAME[r]} & {VNAME[v]} & {cx(d['obs'])} & {f(d['plug1'])} & {f(d['plug2'])} \\\\")
        if d["obs"]["kind"] == "=":
            if d["plug1"] is not None: err1.append(abs(min(d["plug1"], 1) - d["obs"]["p"]))
            if d["plug2"] is not None: err2.append(abs(min(d["plug2"], 1) - d["obs"]["p"]))
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"; lines.append(r"\end{tabular}")
open(os.path.join(TEX, f"tab_h3_gen{OUT}.tex"), "w").write("\n".join(lines) + "\n")

lines = [r"\begin{tabular}{lllrrrr}", r"\toprule", r"forum & retriever & rule & $\Delta$ at 0.10 & $\Delta$ at 0.60 & $\Delta$ at 0.95 & $p^*$ \\", r"\midrule"]
for s, r in CONFIGS:
    for name, lab in [("naive", "none"), ("reopen", "reopen demotion"), ("cite", "store with cited thread"), ("prov0.5", r"provenance $\lambda$=0.5"), ("prov1.0", r"provenance $\lambda$=1.0")]:
        d = T["h4"][f"{s}|{r}"][name]
        lines.append(f"{s} & {NAME[r]} & {lab} & {pm(d['d0.1'])} & {pm(d['d0.6'])} & {pm(d['d0.95'])} & {cx(d['cross'])} \\\\")
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"; lines.append(r"\end{tabular}")
open(os.path.join(TEX, f"tab_h4_gen{OUT}.tex"), "w").write("\n".join(lines) + "\n")

from prep_gen import TAG as _TAG
n_gen = n_filt = 0
for s in SUBS:
    p = os.path.join(CACHE, "gen", f"{s}_gen{_TAG}.jsonl")
    if os.path.exists(p):
        for l in open(p):
            n_gen += 1; n_filt += "content filter" in json.loads(l).get("model", "")
M = [mac("nSubsGen", len(SUBS)), mac("nSeedsGen", len(SEEDS)), mac("nGenTotal", n_gen), mac("nFilteredGen", n_filt),
     mac("maePlugOne", f"{np.mean(err1):.3f}" if err1 else "--"), mac("maePlugTwo", f"{np.mean(err2):.3f}" if err2 else "--"),
     mac("nMissingGen", sum(MISSING.values()))]
# summary macros used in the text (all derived from T)
H1 = T["h1"]; above = lambda k: k in H1 and H1[k]["naive"]["cross"]["kind"] == ">"
st = lambda v: [H1[f"{s}|{r}|{v}"]["steal95"] for s, r in CONFIGS if f"{s}|{r}|{v}" in H1]
M += [mac("nConfigs", len(CONFIGS)),
      mac("nSafeAskGen", sum(above(f"{s}|{r}|askgen") for s, r in CONFIGS)),
      mac("nSafeGen", sum(above(f"{s}|{r}|gen") for s, r in CONFIGS)),
      mac("nSafeGenThr", sum(above(f"{s}|{r}|genthr") for s, r in CONFIGS)),
      mac("nSafePOneAsk", sum(above(f"{s}|{r}|p1_ask") for s, r in CONFIGS)),
      mac("nSafePOneAskDoc", sum(above(f"{s}|{r}|p1_askdoc") for s, r in CONFIGS))]
for v, n in (("askgen", "AskGen"), ("gen", "Gen"), ("genthr", "GenThr"), ("p1_ask", "POneAsk"), ("p1_askdoc", "POneAskDoc")):
    x = st(v)
    if x: M += [mac(f"stealMed{n}", f"{np.median(x):.0f}"), mac(f"stealMax{n}", f"{max(x):.0f}")]
H2 = [T["h2"][f"{s}|{r}|0.95"] for s, r in CONFIGS]
sh = lambda d, k: 100 * d[k] / max(d["wrong"], 1)
dshare = [100 * (d["ent_d"][1] + d["ent_d"][2]) / max(sum(d["ent_d"]), 1) for d in H2]
M += [mac("wrongDOneLo", f"{min(sh(d, 'wrong_self_d1') for d in H2):.1f}"), mac("wrongDOneHi", f"{max(sh(d, 'wrong_self_d1') for d in H2):.1f}"),
      mac("wrongDTwoLo", f"{min(sh(d, 'wrong_self_d2') for d in H2):.1f}"), mac("wrongDTwoHi", f"{max(sh(d, 'wrong_self_d2') for d in H2):.1f}"),
      mac("deepShareLo", f"{min(dshare):.0f}"), mac("deepShareHi", f"{max(dshare):.0f}"),
      mac("driftZero", f"{np.mean([d['drift'][0] for d in H2]):.2f}"), mac("driftOne", f"{np.mean([d['drift'][1] for d in H2]):.2f}"),
      mac("driftTwo", f"{np.mean([d['drift'][2] for d in H2 if d['drift'][2] is not None]):.2f}"),
      mac("selfDTwoHi", f"{max(100 * d['self_d2_by_round'][-1] for d in H2):.1f}"),
      mac("nPlugCfg", len(err1)), mac("nPlugCfgTwo", len(err2))]
H4 = T["h4"]
gain = lambda k, c, p: H4[k][c][f"d{p}"][0]
M += [mac("nReopenBetter", sum(gain(f"{s}|{r}", "reopen", 0.95) > gain(f"{s}|{r}", "naive", 0.95) for s, r in CONFIGS)),
      mac("nReopenSafe", sum(H4[f"{s}|{r}"]["reopen"]["cross"]["kind"] == ">" for s, r in CONFIGS)),
      mac("nNaiveSafe", sum(H4[f"{s}|{r}"]["naive"]["cross"]["kind"] == ">" for s, r in CONFIGS)),
      mac("nCiteSafe", sum(H4[f"{s}|{r}"]["cite"]["cross"]["kind"] == ">" for s, r in CONFIGS)),
      mac("nProvSafe", sum(H4[f"{s}|{r}"]["prov1.0"]["cross"]["kind"] == ">" for s, r in CONFIGS)),
      mac("gainNaiveLow", f"{np.mean([gain(f'{s}|{r}', 'naive', 0.1) for s, r in CONFIGS]):.2f}"),
      mac("gainReopenLow", f"{np.mean([gain(f'{s}|{r}', 'reopen', 0.1) for s, r in CONFIGS]):.2f}"),
      mac("gainProvLow", f"{np.mean([gain(f'{s}|{r}', 'prov0.5', 0.1) for s, r in CONFIGS]):.2f}"),
      mac("gainNaiveHigh", f"{np.mean([gain(f'{s}|{r}', 'naive', 0.95) for s, r in CONFIGS]):+.2f}"),
      mac("gainReopenHigh", f"{np.mean([gain(f'{s}|{r}', 'reopen', 0.95) for s, r in CONFIGS]):+.2f}"),
      mac("gainCiteHigh", f"{np.mean([gain(f'{s}|{r}', 'cite', 0.95) for s, r in CONFIGS]):+.2f}"),
      mac("gainProvHigh", f"{np.mean([gain(f'{s}|{r}', 'prov0.5', 0.95) for s, r in CONFIGS]):+.2f}")]
open(os.path.join(TEX, f"macros_gen{OUT}.tex"), "w").write("\n".join(M) + "\n")
print("configs", len(CONFIGS), "missing keys", MISSING, "plug MAE", np.mean(err1) if err1 else None, np.mean(err2) if err2 else None)

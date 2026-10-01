"""Version-2 analysis: results/sim_v2_*.json -> results/tables_v2.json, tex/*_v2.tex, tex/macros_v2.tex, figs/*_v2.pdf.
No hand-copied numbers: every number used in main_v2.tex comes from here."""
import json, os, sys, collections, itertools
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
from analyze import stream, ci, crossover
FIG = os.path.join(HERE, "figs"); TEX = os.path.join(HERE, "tex"); PC = 0.9
NAME = {"bge": "bge", "minilm": "MiniLM", "bm25": "BM25", "ce": "bge+CE"}
VNAME = {"ask": "ask text (v1)", "doc": "shown thread text", "askdoc": "ask + thread text"}
VARIANTS = ["ask", "doc", "askdoc"]
CE_SUBS = ["android", "mathematica", "unix", "webmasters"]

G = collections.defaultdict(dict)
for fn in sys.argv[1:] or ["sim_v2_main.json", "sim_v2_ce.json"]:
    for r in json.load(open(os.path.join(RES, fn)))["runs"]:
        G[(r["sub"], r["ret"], r["variant"], r["user"], r["cond"], r["p_w"])][r["seed"]] = r["curve"]
SEEDS = list(range(10)); PWS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]; PSAFE = [0.3, 0.6, 0.8, 0.95]
CONFIGS = [(s, r) for s in FORUMS for r in RETRIEVERS] + [(s, "ce") for s in CE_SUBS]
CONFIGS = [k for k in CONFIGS if (k[0], k[1], "ask", "const", "static", None) in G]
rng = np.random.default_rng(0)

def static(s, r): return G[(s, r, "ask", "const", "static", None)]
def delta(s, r, v, u, cond, p, key="acc", seeds=SEEDS):
    st = static(s, r); runs = G[(s, r, v, u, cond, p)]
    return [100 * (stream(runs[sd], key) - stream(st[sd], key)) for sd in seeds]
def cross(s, r, v, u, cond="naive", seeds=SEEDS, boot=True):
    M = np.array([delta(s, r, v, u, cond, p, seeds=seeds) for p in PWS]); x, kind = crossover(PWS, M.mean(1))
    if not boot: return dict(p=x, kind=kind)
    b = [crossover(PWS, M[:, rng.integers(0, len(seeds), len(seeds))].mean(1))[0] for _ in range(1000)]
    return dict(p=x, kind=kind, lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)))
def half(runs, key, seeds=SEEDS):
    return float(np.mean([stream(runs[sd], key, lo=len(runs[sd]) // 2) for sd in seeds]))
def ratio(runs, num, den, seeds=SEEDS):
    a = sum(x[num] for sd in seeds for x in runs[sd][len(runs[sd]) // 2:]); b = sum(x[den] for sd in seeds for x in runs[sd][len(runs[sd]) // 2:])
    return a / b if b else float("nan")
def plug(s, r, v, u, pref, seeds):
    runs = G[(s, r, v, u, "naive", pref)]
    A = float(np.mean([stream(static(s, r)[sd], "acc") for sd in seeds]))
    ec = sum(x["eff_c"] for sd in seeds for x in runs[sd]) / max(sum(x["ent_c"] for sd in seeds for x in runs[sd][:-1]), 1)
    ew = sum(x["eff_w"] for sd in seeds for x in runs[sd]) / max(sum(x["ent_w"] for sd in seeds for x in runs[sd][:-1]), 1)
    return (-PC * A * ec / ((1 - A) * ew)) if ew < 0 else None
def kind_of(p): return None if p is None else ("<" if p < PWS[0] else (">" if p > PWS[-1] else "="))

T = dict(base={}, abl={}, diff={}, safe={}, oos={}, ce={})
for s, r in CONFIGS:
    T["base"][f"{s}|{r}"] = {k: float(np.mean([stream(static(s, r)[sd], k) for sd in SEEDS])) for k in ("acc", "hit5")}

# 1. entry-content ablation
for s, r in CONFIGS:
    for v in (["ask"] if r == "ce" else VARIANTS):
        d = {}
        for cond in ("naive", "reopen"):
            d[cond] = dict(cross=cross(s, r, v, "const", cond),
                           **{f"d{p}": ci(delta(s, r, v, "const", cond, p)) for p in PWS},
                           **{f"h5_{p}": ci(delta(s, r, v, "const", cond, p, "hit5")) for p in (0.3, 0.95)})
        run95 = G[(s, r, v, "const", "naive", 0.95)]
        d["self95"] = 100 * half(run95, "self_share"); d["capture95"] = 100 * ratio(run95, "capture", "cov")
        d["capw95"] = 100 * ratio(run95, "capture_w", "cov"); d["steal95"] = 100 * ratio(run95, "steal", "cov_ok")
        T["abl"][f"{s}|{r}|{v}"] = d

# 2. difficulty-dependent users
for s, r in CONFIGS:
    for v in (["ask"] if r == "ce" else ["ask", "askdoc"]):
        c0 = T["abl"][f"{s}|{r}|{v}"]["naive"]["cross"]; c1 = cross(s, r, v, "diff", "naive")
        realized = {p: ratio(G[(s, r, v, "diff", "naive", p)], "acc_w", "wrong") for p in (0.3, 0.95)}
        T["diff"][f"{s}|{r}|{v}"] = dict(const=c0, diff=c1, d95=ci(delta(s, r, v, "diff", "naive", 0.95)), d30=ci(delta(s, r, v, "diff", "naive", 0.3)),
                                         rd95=ci(delta(s, r, v, "diff", "reopen", 0.95)), realized=realized)

# 3. baselines
for s, r in CONFIGS:
    for v in (["ask"] if r == "ce" else VARIANTS):
        for p in PSAFE:
            for cond in ("naive", "reopen", "reopen+quarantine", "gate", "footprint"):
                runs = G[(s, r, v, "const", cond, p)]
                ents = np.mean([runs[sd][-1]["entries"] for sd in SEEDS])
                T["safe"][f"{s}|{r}|{v}|{cond}|{p}"] = dict(d=ci(delta(s, r, v, "const", cond, p)), entries=float(ents))

# 4. out-of-sample plug-in predictor: calibrate on 5 orders, evaluate on the other 5 (both directions)
for v in ("ask", "askdoc"):
    for pref in (0.3, 0.5):
        rows = []
        for cal, ev in ((SEEDS[:5], SEEDS[5:]), (SEEDS[5:], SEEDS[:5])):
            for s, r in CONFIGS:
                if r == "ce" and v != "ask": continue
                pp = plug(s, r, v, "const", pref, cal); ob = cross(s, r, v, "const", "naive", seeds=ev, boot=False)
                rows.append(dict(cfg=f"{s}|{r}", pred=pp, kp=kind_of(pp), obs=ob["p"], ko=ob["kind"]))
        both = [x for x in rows if x["kp"] == "=" and x["ko"] == "="]
        o = np.array([x["obs"] for x in both]); pr = np.array([x["pred"] for x in both])
        same_c = sum(x["kp"] == x["ko"] for x in rows if x["kp"] is not None)
        T["oos"][f"{v}|{pref}"] = dict(n=len(both), mae=float(np.mean(np.abs(o - pr))), spearman=float(stats.spearmanr(o, pr)[0]),
                                       same_class=same_c, n_defined=sum(x["kp"] is not None for x in rows), n_rows=len(rows),
                                       median_abs_err=float(np.median(np.abs(o - pr))))
        # baseline predictor: in-sample one-parameter law with base accuracy (fit on cal, evaluate on ev)
json.dump(T, open(os.path.join(RES, "tables_v2.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

# ---------------- LaTeX ----------------
def cx(c): return {"<": r"$<$0.10", ">": r"$>$0.95"}.get(c["kind"], f"{c['p']:.2f}")
def cxi(c): return {"<": r"$<$0.10", ">": r"$>$0.95"}.get(c["kind"], f"{c['p']:.2f} [{c['lo']:.2f}, {c['hi']:.2f}]")
def pm(v): return f"{v[0]:+.2f} $\\pm$ {v[1]:.2f}"
def mac(n, v): return f"\\newcommand{{\\{n}}}{{{v}}}"
M = []
DENSE = [(s, r) for s, r in CONFIGS if r in ("bge", "minilm")]

# Table A: ablation summary per retriever x variant
L = [r"\begin{tabular}{llrrrrrrr}\toprule Retriever & Entry content & $\Delta$ @0.3 & $\Delta$ @0.95 & sig.\ $<0$ @0.95 & median $p^*$ & self-share @0.95 & capture & steal\\\midrule"]
for r in RETRIEVERS:
    for v in VARIANTS:
        ks = [f"{s}|{r}|{v}" for s in FORUMS]; A = [T["abl"][k] for k in ks]
        d3 = np.mean([a["naive"]["d0.3"][0] for a in A]); d95 = np.mean([a["naive"]["d0.95"][0] for a in A])
        neg = sum(a["naive"]["d0.95"][0] + a["naive"]["d0.95"][1] < 0 for a in A)
        ps = [min(max(a["naive"]["cross"]["p"], 0.1), 0.95) if a["naive"]["cross"]["kind"] != ">" else 1.0 for a in A]
        med = np.median(ps); meds = r"$>$0.95" if med > 0.95 else f"{med:.2f}"
        L.append(f"{NAME[r] if v=='ask' else ''} & {VNAME[v]} & {d3:+.2f} & {d95:+.2f} & {neg}/12 & {meds} & {np.mean([a['self95'] for a in A]):.0f}\\% & {np.mean([a['capture95'] for a in A]):.0f}\\% & {np.mean([a['steal95'] for a in A]):.1f}\\%\\\\")
        T.setdefault("abl_sum", {})[f"{r}|{v}"] = dict(d3=d3, d95=d95, neg=neg, med=med, self=np.mean([a['self95'] for a in A]), cap=np.mean([a['capture95'] for a in A]), capw=np.mean([a['capw95'] for a in A]), steal=np.mean([a['steal95'] for a in A]))
    if r != "bm25": L.append(r"\addlinespace[2pt]")
L.append(r"\bottomrule\end{tabular}"); open(os.path.join(TEX, "tab_ablation_v2.tex"), "w").write("\n".join(L))

# Table B: per-config crossover for each variant (naive) and askdoc+demotion
L = [r"\begin{tabular}{llllll}\toprule Forum & Retriever & ask text (v1) & shown thread & ask + thread & ask + thread, +demotion\\\midrule"]
for s in FORUMS:
    for r in RETRIEVERS:
        a = lambda v, c="naive": cxi(T["abl"][f"{s}|{r}|{v}"][c]["cross"])
        L.append(f"{s if r=='bge' else ''} & {NAME[r]} & {a('ask')} & {a('doc')} & {a('askdoc')} & {a('askdoc','reopen')}\\\\")
    if s != FORUMS[-1]: L.append(r"\addlinespace[1pt]")
L.append(r"\bottomrule\end{tabular}"); open(os.path.join(TEX, "tab_abl_cross_v2.tex"), "w").write("\n".join(L))

# Table C: difficulty-dependent users
L = [r"\begin{tabular}{llllll}\toprule Forum & Retriever & $p^*$ constant & $p^*$ difficulty-dep. & $\Delta$@0.95 constant & $\Delta$@0.95 difficulty-dep.\\\midrule"]
for s in FORUMS:
    for r in RETRIEVERS:
        d = T["diff"][f"{s}|{r}|ask"]
        L.append(f"{s if r=='bge' else ''} & {NAME[r]} & {cxi(d['const'])} & {cxi(d['diff'])} & {pm(T['abl'][f'{s}|{r}|ask']['naive']['d0.95'])} & {pm(d['d95'])}\\\\")
    if s != FORUMS[-1]: L.append(r"\addlinespace[1pt]")
L.append(r"\bottomrule\end{tabular}"); open(os.path.join(TEX, "tab_diff_v2.tex"), "w").write("\n".join(L))

# Table D: baselines (mean over 24 dense configs) per variant and p_w
CN = {"naive": "naive", "reopen": "+demotion", "reopen+quarantine": "+dem.+quar.", "gate": "confidence gate", "footprint": "footprint rule"}
L = [r"\begin{tabular}{llrrrrrr}\toprule Entry content & Rule & $\Delta$@0.3 & $\Delta$@0.6 & $\Delta$@0.8 & $\Delta$@0.95 & $\ge0$ @0.95 & entries kept\\\midrule"]
for v in VARIANTS:
    for cond in CN:
        row = [np.mean([T["safe"][f"{s}|{r}|{v}|{cond}|{p}"]["d"][0] for s, r in DENSE]) for p in PSAFE]
        nn = sum(T["safe"][f"{s}|{r}|{v}|{cond}|0.95"]["d"][0] >= 0 for s, r in DENSE)
        ek = np.mean([T["safe"][f"{s}|{r}|{v}|{cond}|0.95"]["entries"] / max(T["safe"][f"{s}|{r}|{v}|naive|0.95"]["entries"], 1) for s, r in DENSE])
        L.append(f"{VNAME[v] if cond=='naive' else ''} & {CN[cond]} & " + " & ".join(f"{x:+.2f}" for x in row) + f" & {nn}/{len(DENSE)} & {100*ek:.0f}\\%\\\\")
        T.setdefault("safe_sum", {})[f"{v}|{cond}"] = dict(d=row, nonneg95=nn, kept=ek)
    if v != VARIANTS[-1]: L.append(r"\addlinespace[2pt]")
L.append(r"\bottomrule\end{tabular}"); open(os.path.join(TEX, "tab_baselines_v2.tex"), "w").write("\n".join(L))

# Table E: reranker
CEK = [s for s in CE_SUBS if (s, "ce") in CONFIGS]
L = [r"\begin{tabular}{llrrlll}\toprule Forum & Retriever & Top-1 & Hit@5 & $p^*$ & $\Delta$@0.95 & self-share @0.95\\\midrule"]
for s in CEK:
    for r in ("bge", "ce"):
        b = T["base"][f"{s}|{r}"]; a = T["abl"][f"{s}|{r}|ask"]
        L.append(f"{s if r=='bge' else ''} & {NAME[r]} & {100*b['acc']:.1f} & {100*b['hit5']:.1f} & {cxi(a['naive']['cross'])} & {pm(a['naive']['d0.95'])} & {a['self95']:.0f}\\%\\\\")
L.append(r"\bottomrule\end{tabular}"); open(os.path.join(TEX, "tab_ce_v2.tex"), "w").write("\n".join(L))
json.dump(T, open(os.path.join(RES, "tables_v2.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

# ---------------- macros ----------------
S = T["abl_sum"]
for r in RETRIEVERS:
    for v in VARIANTS:
        k = S[f"{r}|{v}"]; tag = {"bge": "Bge", "minilm": "Mini", "bm25": "Bm"}[r] + {"ask": "Ask", "doc": "Doc", "askdoc": "Askdoc"}[v]
        M += [mac(f"dNN{tag}", f"{k['d95']:+.2f}"), mac(f"dTh{tag}", f"{k['d3']:+.2f}"), mac(f"neg{tag}", k["neg"]), mac(f"self{tag}", f"{k['self']:.0f}"),
              mac(f"cap{tag}", f"{k['cap']:.0f}"), mac(f"capw{tag}", f"{k['capw']:.0f}"), mac(f"steal{tag}", f"{k['steal']:.0f}"),
              mac(f"med{tag}", r"$>$0.95" if k["med"] > 0.95 else f"{k['med']:.2f}")]
# crossover movement under difficulty-dependent users (dense, ask variant)
mv = []; up = dn = same = 0
for s, r in DENSE:
    d = T["diff"][f"{s}|{r}|ask"]; a, b = d["const"], d["diff"]
    pa = a["p"] if a["kind"] == "=" else (0.1 if a["kind"] == "<" else 0.95); pb = b["p"] if b["kind"] == "=" else (0.1 if b["kind"] == "<" else 0.95)
    mv.append(pb - pa)
    if b["kind"] == "=" and a["kind"] == "=" and not (b["lo"] <= a["p"] <= b["hi"] or a["lo"] <= b["p"] <= a["hi"]): up += pb > pa; dn += pb < pa
    elif a["kind"] != b["kind"]: up += pb > pa; dn += pb < pa
    else: same += 1
M += [mac("diffMedShift", f"{np.median(mv):+.2f}"), mac("diffUp", up), mac("diffDown", dn), mac("diffMinShift", f"{min(mv):+.2f}"), mac("diffMaxShift", f"{max(mv):+.2f}")]
d95c = [T["abl"][f"{s}|{r}|ask"]["naive"]["d0.95"][0] for s, r in DENSE]; d95d = [T["diff"][f"{s}|{r}|ask"]["d95"][0] for s, r in DENSE]
M += [mac("diffDnnConst", f"{np.mean(d95c):+.2f}"), mac("diffDnnDiff", f"{np.mean(d95d):+.2f}"),
      mac("diffNegNN", sum(T['diff'][f'{s}|{r}|ask']['d95'][0] + T['diff'][f'{s}|{r}|ask']['d95'][1] < 0 for s, r in DENSE)),
      mac("realizedNN", f"{np.mean([T['diff'][f'{s}|{r}|ask']['realized'][0.95] for s, r in DENSE]):.2f}")]
T["diff_sum"] = dict(shift=mv, up=up, down=dn)
for v, pref in (("ask", 0.3), ("ask", 0.5), ("askdoc", 0.3)):
    o = T["oos"][f"{v}|{pref}"]; tg = {"ask": "Ask", "askdoc": "Askdoc"}[v] + {0.3: "Th", 0.5: "Fi"}[pref]
    M += [mac(f"oosN{tg}", o["n"]), mac(f"oosMAE{tg}", f"{o['mae']:.3f}"), mac(f"oosRho{tg}", f"{o['spearman']:.2f}"), mac(f"oosSame{tg}", o["same_class"]), mac(f"oosDef{tg}", o["n_defined"])]
for v in VARIANTS:
    for cond in CN:
        k = T["safe_sum"][f"{v}|{cond}"]; tg = {"ask": "Ask", "doc": "Doc", "askdoc": "Askdoc"}[v] + {"naive": "Naive", "reopen": "Dem", "reopen+quarantine": "Quar", "gate": "Gate", "footprint": "Foot"}[cond]
        M += [mac(f"bl{tg}", f"{k['d'][-1]:+.2f}"), mac(f"blTh{tg}", f"{k['d'][0]:+.2f}"), mac(f"blNN{tg}", k["nonneg95"]), mac(f"blKept{tg}", f"{100*k['kept']:.0f}")]
if CEK:
    gain = [100 * (T["base"][f"{s}|ce"]["acc"] - T["base"][f"{s}|bge"]["acc"]) for s in CEK]
    M += [mac("ceGainMin", f"{min(gain):.1f}"), mac("ceGainMax", f"{max(gain):.1f}"), mac("ceN", len(CEK)),
          mac("ceNeg", sum(T['abl'][f'{s}|ce|ask']['naive']['d0.95'][0] + T['abl'][f'{s}|ce|ask']['naive']['d0.95'][1] < 0 for s in CEK)),
          mac("ceAccMin", f"{100*min(T['base'][f'{s}|ce']['acc'] for s in CEK):.1f}"), mac("ceAccMax", f"{100*max(T['base'][f'{s}|ce']['acc'] for s in CEK):.1f}")]
M += [mac("nDenseV", len(DENSE))]
open(os.path.join(TEX, "macros_v2.tex"), "w").write("\n".join(M) + "\n")
json.dump(T, open(os.path.join(RES, "tables_v2.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

# ---------------- figures ----------------
col = {"ask": "#c05621", "doc": "#2f855a", "askdoc": "#2b6cb0"}
fig, axs = plt.subplots(1, 3, figsize=(9, 2.7), sharey=True)
for ax, r in zip(axs, RETRIEVERS):
    for v in VARIANTS:
        for cond, ls in (("naive", "-"), ("reopen", "--")):
            y = np.array([[T["abl"][f"{s}|{r}|{v}"][cond][f"d{p}"][0] for p in PWS] for s in FORUMS]).mean(0)
            ax.plot(PWS, y, ls, marker="o", ms=2.5, color=col[v], lw=1, label=VNAME[v] if cond == "naive" else None)
    ax.axhline(0, color="k", lw=0.5); ax.set_title({"bge": "bge-small", "minilm": "MiniLM-L6", "bm25": "BM25"}[r], fontsize=9); ax.tick_params(labelsize=7)
    ax.set_xlabel("$p_w$", fontsize=8)
axs[0].set_ylabel("mean change in top-1 (points)", fontsize=8); axs[0].legend(fontsize=6.5, frameon=False)
axs[1].plot([], [], "k-", lw=1, label="naive"); axs[1].plot([], [], "k--", lw=1, label="+demotion"); axs[1].legend(fontsize=6.5, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "ablation_v2.pdf")); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.2, 3))
for s, r in DENSE:
    d = T["diff"][f"{s}|{r}|ask"]; a, b = d["const"], d["diff"]
    mk = "o" if r == "bge" else "s"
    ax.plot(min(max(a["p"], 0.05), 1.0), min(max(b["p"], 0.05), 1.0), mk, ms=4, color="#2b6cb0" if r == "bge" else "#2f855a", alpha=0.8)
ax.plot([0, 1], [0, 1], "k--", lw=0.7); ax.set_xlabel("$p^*$, constant acceptance", fontsize=8); ax.set_ylabel("$p^*$, difficulty-dependent acceptance", fontsize=8)
ax.plot([], [], "o", color="#2b6cb0", label="bge-small"); ax.plot([], [], "s", color="#2f855a", label="MiniLM-L6"); ax.legend(fontsize=7, frameon=False)
ax.tick_params(labelsize=7); fig.tight_layout(); fig.savefig(os.path.join(FIG, "difficulty_v2.pdf")); plt.close(fig)
print("\n".join(M))

"""Write LaTeX tables and macros straight from results/tables_*.json (no hand-copied numbers)."""
import json, os, itertools, collections, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
T = json.load(open(os.path.join(RES, "tables_r10_all.json"))); T20 = json.load(open(os.path.join(RES, "tables_r20_unix_android_english_tex.json")))
NAME = {"bge": "bge", "minilm": "MiniLM", "bm25": "BM25"}; OUT = os.path.join(HERE, "tex")
def cx(c): return {"<": r"$<$0.10", ">": r"$>$0.95"}.get(c["kind"], f"{c['p']:.2f} [{c['lo']:.2f}, {c['hi']:.2f}]")
def pm(v): return f"{v[0]:+.2f} $\\pm$ {v[1]:.2f}"
# Table 1: data
L = [r"\begin{tabular}{lrrrr}\toprule Forum & KB posts & Asks & Families & Repeat asks\\\midrule"]
for s in FORUMS:
    m = T["meta"][s]; L.append(f"{s} & {m['kb']:,} & {m['asks']:,} & {m['families']:,} & {m['repeat_asks']:,}\\\\")
tot = {k: sum(T["meta"][s][k] for s in FORUMS) for k in ["kb", "asks", "families", "repeat_asks"]}
L.append(r"\midrule " + f"total & {tot['kb']:,} & {tot['asks']:,} & {tot['families']:,} & {tot['repeat_asks']:,}\\\\" + r"\bottomrule\end{tabular}")
open(os.path.join(OUT, "tab_data.tex"), "w").write("\n".join(L))
# Table 2: base accuracy + crossover + plug-in prediction
L = [r"\begin{tabular}{llrrlll}\toprule Forum & Retriever & Top-1 & Hit@5 & Observed $p^*$ (top-1) & Observed $p^*$ (hit@5) & Plug-in $\hat p^*$\\\midrule"]
for s in FORUMS:
    for r in RETRIEVERS:
        k = f"{s}|{r}"; b = T["base"][k]; pl = T["plug"][k]["0.3"]["pstar"]
        pls = "--" if pl is None else (r"$>$0.95" if pl > 0.95 else (r"$<$0.10" if pl < 0.1 else f"{pl:.2f}"))
        L.append(f"{s if r=='bge' else ''} & {NAME[r]} & {100*b['acc']:.1f} & {100*b['hit5']:.1f} & {cx(T['cross'][k+'|acc'])} & {cx(T['cross'][k+'|hit5'])} & {pls}\\\\")
    if s != FORUMS[-1]: L.append(r"\addlinespace[1pt]")
L.append(r"\bottomrule\end{tabular}"); open(os.path.join(OUT, "tab_cross.tex"), "w").write("\n".join(L))
# Table 3: safeguards at p_w in {0.3, 0.95}, dense retrievers
L = [r"\begin{tabular}{llrlll}\toprule Forum & Retriever & $p_w$ & Naive & + demotion & + demotion + quarantine\\\midrule"]
for s in FORUMS:
    for r in ["bge", "minilm"]:
        for p in ["0.3", "0.95"]:
            k = f"{s}|{r}"
            L.append(f"{s if (r=='bge' and p=='0.3') else ''} & {NAME[r] if p=='0.3' else ''} & {p} & {pm(T['naive'][f'{k}|naive|{p}|acc'])} & {pm(T['safe'][f'{k}|reopen|{p}|acc'])} & {pm(T['safe'][f'{k}|reopen+quarantine|{p}|acc'])}\\\\")
    if s != FORUMS[-1]: L.append(r"\addlinespace[1pt]")
L.append(r"\bottomrule\end{tabular}"); open(os.path.join(OUT, "tab_safe.tex"), "w").write("\n".join(L))
# Table 4: monitors
lab = {"accept_rate": "Acceptance rate (baseline)", "self_share": "Self-written share of top-1", "selfdis": "Pilot signal: self vs.\\ human top entry",
       "ind_dis": "Disagreement with independent retriever", "ind_rise": "\\quad rise since round 1", "bank_drift": "Probe-bank drift"}
L = [r"\begin{tabular}{lrrrrr}\toprule Signal & $\rho$ (all runs) & $\rho$ (fixed $p_w$) & AUROC r2 & AUROC r5 & AUROC r10\\\midrule"]
for m in ["accept_rate", "self_share", "selfdis", "ind_dis", "ind_rise", "bank_drift"]:
    v = T["monitor"][m]; a = v["auroc_by_round"]
    L.append(f"{lab[m]} & {v['spearman_runs']:+.2f} & {v['spearman_fixed_p']:+.2f} & {a[1]:.2f} & {a[4]:.2f} & {a[9]:.2f}\\\\")
L.append(r"\bottomrule\end{tabular}"); open(os.path.join(OUT, "tab_monitor.tex"), "w").write("\n".join(L))
# Table 5: 10 vs 20 rounds
L = [r"\begin{tabular}{llllll}\toprule Forum & Retriever & $p^*$ (10 rounds) & $p^*$ (20 rounds) & Naive, $p_w{=}0.95$ (10) & (20)\\\midrule"]
for s in ["android", "english", "tex", "unix"]:
    for r in ["bge", "minilm"]:
        k = f"{s}|{r}"
        L.append(f"{s if r=='bge' else ''} & {NAME[r]} & {cx(T['cross'][k+'|acc'])} & {cx(T20['cross'][k+'|acc'])} & {pm(T['naive'][k+'|naive|0.95|acc'])} & {pm(T20['naive'][k+'|naive|0.95|acc'])}\\\\")
L.append(r"\bottomrule\end{tabular}"); open(os.path.join(OUT, "tab_r20.tex"), "w").write("\n".join(L))
# macros used in text
def mac(n, v): return f"\\newcommand{{\\{n}}}{{{v}}}"
f = T["fit"]["acc"]; pe = T["plug_eval"]["0.3"]; pe5 = T["plug_eval"]["0.5"]; pe95 = T["plug_eval"]["0.95"]
dense = [f"{s}|{r}" for s in FORUMS for r in ["bge", "minilm"]]
n95 = [T["naive"][k + "|naive|0.95|acc"][0] for k in dense]; rq95 = [T["safe"][k + "|reopen+quarantine|0.95|acc"][0] for k in dense]
rec = [(b - a) / -a for a, b in zip(n95, rq95) if a < 0]
neg95 = sum(T["naive"][k + "|naive|0.95|acc"][0] + T["naive"][k + "|naive|0.95|acc"][1] < 0 for k in dense)
ss = [T["selfshare"][k + "|naive|0.95"][0] for k in dense]; ssb = [T["selfshare"][f"{s}|bm25|naive|0.95"][0] for s in FORUMS]
rqpos3 = sum(T["safe"][k + "|reopen+quarantine|0.3|acc"][0] > 0 for k in dense)
qdiff = [abs(T["safe"][k + "|reopen+quarantine|0.95|acc"][0] - T["safe"][k + "|reopen|0.95|acc"][0]) for k in dense]
crossd = [T["cross"][k + "|acc"]["p"] for k in dense if T["cross"][k + "|acc"]["kind"] == "="]
M = [mac("kappaOne", f"{f['kappa']:.2f}"), mac("rsqOne", f"{f['r2']:.2f}"), mac("nOne", f["n"]), mac("maeOne", f"{f['mae']:.2f}"), mac("rhoOne", f"{f['spearman']:.2f}"),
     mac("plugN", pe["n_both_interior"]), mac("plugMAE", f"{pe['mae']:.3f}"), mac("plugRho", f"{pe['spearman']:.2f}"), mac("plugAgree", pe["within_ci_or_same_censoring"]), mac("plugTot", pe["n"]),
     mac("plugMAEfive", f"{pe5['mae']:.3f}"), mac("plugMAEnn", f"{pe95['mae']:.3f}"),
     mac("negNF", neg95), mac("nDense", len(dense)), mac("recMed", f"{100*np.median(rec):.0f}"), mac("recMin", f"{100*min(rec):.0f}"), mac("recMax", f"{100*max(rec):.0f}"),
     mac("naiveMin", f"{min(n95):.2f}"), mac("naiveMax", f"{max(n95):.2f}"), mac("ssMin", f"{min(ss):.0f}"), mac("ssMax", f"{max(ss):.0f}"), mac("ssBmMax", f"{max(ssb):.0f}"),
     mac("rqPosThree", rqpos3), mac("qDiffMax", f"{max(qdiff):.2f}"), mac("crossMin", f"{min(crossd):.2f}"), mac("crossMax", f"{max(crossd):.2f}"),
     mac("totKB", f"{sum(T['meta'][s]['kb'] for s in FORUMS):,}"), mac("totAsks", f"{sum(T['meta'][s]['asks'] for s in FORUMS):,}"), mac("totRep", f"{sum(T['meta'][s]['repeat_asks'] for s in FORUMS):,}")]
open(os.path.join(OUT, "macros.tex"), "w").write("\n".join(M) + "\n")
print("\n".join(M))
# figure: self-written share over 20 rounds (bge), from raw results
J = json.load(open(os.path.join(RES, "sim_r20_unix_android_english_tex.json")))
G = collections.defaultdict(list)
for r in J["runs"]:
    if r["cond"] in ("naive", "reopen+quarantine") and r["p_w"] in (0.3, 0.95): G[(r["sub"], r["ret"], r["cond"], r["p_w"])].append([x["self_share"] for x in r["curve"]])
fig, axs = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
for ax, ret in zip(axs, ["bge", "bm25"]):
    for s, c in zip(["android", "english", "tex", "unix"], ["#2b6cb0", "#2f855a", "#c05621", "#6b46c1"]):
        for (cond, p), ls in {("naive", 0.95): "-", ("naive", 0.3): ":", ("reopen+quarantine", 0.95): "--"}.items():
            y = 100 * np.mean(G[(s, ret, cond, p)], 0); ax.plot(range(1, 21), y, ls, color=c, lw=1, label=s if (cond, p) == ("naive", 0.95) else None)
    ax.set_title({"bge": "bge-small", "bm25": "BM25"}[ret], fontsize=9); ax.set_xlabel("round (of 20)", fontsize=8); ax.tick_params(labelsize=7)
axs[0].set_ylabel("self-written share of top-1 (%)", fontsize=8); axs[0].legend(fontsize=7, frameon=False)
axs[1].plot([], [], "k-", lw=1, label="naive, $p_w$=0.95"); axs[1].plot([], [], "k:", lw=1, label="naive, $p_w$=0.3"); axs[1].plot([], [], "k--", lw=1, label="+dem.+quar., $p_w$=0.95"); axs[1].legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "selfshare_r20.pdf"))
# extra macros: safeguards at p_w=0.95 for dense retrievers
turned = sum(b >= 0 for a, b in zip(n95, rq95)); still = [(b - a) / -a for a, b in zip(n95, rq95) if a < 0 and b < 0]
open(os.path.join(OUT, "macros.tex"), "a").write("\n".join([mac("rqTurned", turned), mac("recMedStill", f"{100*np.median(still):.0f}"), mac("nStill", len(still)),
    mac("recMinStill", f"{100*min(still):.0f}"), mac("recMaxStill", f"{100*max(still):.0f}")]) + "\n")
# crossover figure: (a) one-parameter law, (b) plug-in model
col = {"bge": "#2b6cb0", "minilm": "#2f855a", "bm25": "#c05621"}
fig, axs = plt.subplots(1, 2, figsize=(7, 3))
for s, r in itertools.product(FORUMS, RETRIEVERS):
    k = f"{s}|{r}"; c = T["cross"][k + "|acc"]; A = T["base"][k]["acc"]; pl = T["plug"][k]["0.3"]["pstar"]
    mk = {"=": "o", "<": "v", ">": "^"}[c["kind"]]
    yerr = [[c["p"] - c["lo"]], [c["hi"] - c["p"]]] if c["kind"] == "=" else None
    axs[0].errorbar(0.9 * A / (1 - A), c["p"], yerr=yerr, fmt=mk, color=col[r], ms=4, lw=0.6)
    if pl is not None:
        axs[1].errorbar(min(max(pl, 0.05), 1.0), c["p"], yerr=yerr, fmt=mk, color=col[r], ms=4, lw=0.6)
xx = np.linspace(0, 0.5, 20); axs[0].plot(xx, f["kappa"] * xx, "k--", lw=0.8)
axs[0].set_xlabel("$p_c A/(1-A)$", fontsize=8); axs[0].set_title(f"(a) one-parameter law, $R^2$={f['r2']:.2f}", fontsize=9)
axs[1].plot([0, 1], [0, 1], "k--", lw=0.8); axs[1].set_xlabel("plug-in prediction $\\hat p^*$ (calibrated at $p_w$=0.3; clipped)", fontsize=8)
axs[1].set_title(f"(b) plug-in model, MAE={pe['mae']:.3f}", fontsize=9)
for ax in axs: ax.set_ylabel("observed crossover $p^*$", fontsize=8); ax.set_ylim(0, 1); ax.tick_params(labelsize=7)
for r in RETRIEVERS: axs[0].plot([], [], "o", color=col[r], label={"bge": "bge-small", "minilm": "MiniLM-L6", "bm25": "BM25"}[r])
axs[0].plot([], [], "kv", label="$p^*<0.10$"); axs[0].plot([], [], "k^", label="$p^*>0.95$"); axs[0].legend(fontsize=6, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "crossover_two.pdf"))

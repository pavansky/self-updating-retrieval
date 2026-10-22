"""Paper-2 TMLR version: macros and tables from results/tables_gen3*.json, the answer/judge caches and the audit.
Writes tmlr/macros.tex, tmlr/tab3_*.tex, tmlr/tab_full.tex, tmlr/tab_audit3.tex. No number in the paper is typed by hand.
  python write_tex_gen3.py
"""
import json, os, collections
import numpy as np
from scipy import stats
from common import HERE, RES, CACHE

OUTD = os.path.join(HERE, "tmlr"); G = os.path.join(CACHE, "gen")
T = json.load(open(os.path.join(RES, "tables_gen3.json")))
TL = json.load(open(os.path.join(RES, "tables_gen3_local.json"))) if os.path.exists(os.path.join(RES, "tables_gen3_local.json")) else None
TO = json.load(open(os.path.join(RES, "tables_gen3_open.json"))) if os.path.exists(os.path.join(RES, "tables_gen3_open.json")) else None
SUBS = T["meta"]["subs"]; RETS = T["meta"]["rets"]; CONFIGS = [(s, r) for s in SUBS for r in RETS]
NAME = {"bge": "bge", "minilm": "MiniLM", "bm25": "BM25"}
M = {}
def mac(name, v): M[name] = v
def f1(x): return f"{x:.1f}"
def f2(x): return f"{x:.2f}"
def pm(v): return f"{v[0]:+.2f} $\\pm$ {v[1]:.2f}"
def cx(c): return {"<": r"$<$0.10", ">": r"$>$0.95"}.get(c["kind"], f"{c['p']:.2f}")
def write(name, lines): open(os.path.join(OUTD, name), "w").write("\n".join(lines) + "\n")

# ---- corpus and generation --------------------------------------------------------------------------------------------
import sim_v2
sizes, multi, largest = [], [], []
for s in SUBS:
    af = np.asarray(sim_v2.get(s, "bge")["afam"]); cnt = collections.Counter(af.tolist())
    sizes.append(len(af)); multi.append(100 * sum(v for v in cnt.values() if v > 1) / len(af)); largest.append(100 * max(cnt.values()) / len(af))
mac("nAsksMin", f"{min(sizes):,}"); mac("nAsksMax", f"{max(sizes):,}"); mac("multiShareMin", f"{min(multi):.0f}"); mac("multiShareMax", f"{max(multi):.0f}")
mac("largestFamMax", f"{max(largest):.0f}")
ngen = nfilt = 0; njud = 0; labc = collections.Counter()
for s in SUBS:
    for l in open(os.path.join(G, f"{s}_gen_az.jsonl")): d = json.loads(l); ngen += 1; nfilt += "content filter" in d.get("model", "")
    for l in open(os.path.join(G, f"{s}_judge_az.jsonl")): d = json.loads(l); labc[d["label"]] += 1
njud = sum(v for k, v in labc.items() if k not in ("ERR", "NOREF"))
mac("nGenTotal", f"{ngen:,}"); mac("nFilteredGen", str(nfilt)); mac("nJudged", f"{njud:,}")
pool_all = collections.Counter()
for k, v in T["h2"].items():
    for d in ("0", "1", "2"):
        for l, n in v["labels"][d].items(): pool_all[l] += n
mac("noRefShare", f"{100 * pool_all['NOREF'] / max(sum(pool_all.values()), 1):.0f}")

# ---- audit: three judges on the 900-answer sample ------------------------------------------------------------------
A = json.load(open(os.path.join(RES, "audit_gen_az_n900.json")))
J = {}
for s in SUBS:
    for l in open(os.path.join(G, f"{s}_judge_az.jsonl")): d = json.loads(l); J[d["key"]] = d["label"]
for x in A: x["judge_ds"] = {"ABSTAIN": "NO"}.get(J.get(x["key"], "MISSING"), J.get(x["key"], "MISSING"))
A3 = [x for x in A if x["judge_ds"] in ("YES", "PARTLY", "NO")]
def agree(a, b): return 100 * np.mean([x[a] == x[b] for x in A3])
def kappa(a, b):
    la = [x[a] for x in A3]; lb = [x[b] for x in A3]; po = np.mean([p == q for p, q in zip(la, lb)])
    pe = sum((la.count(l) / len(la)) * (lb.count(l) / len(lb)) for l in ("YES", "PARTLY", "NO")); return (po - pe) / (1 - pe)
mac("nAuditBig", str(len(A))); mac("judgeAgreeAz", f"{agree('judge_ds', 'judge_az'):.0f}"); mac("judgeAgreeLoc", f"{agree('judge_ds', 'judge_local'):.0f}")
mac("judgeAgreePair", f"{agree('judge_az', 'judge_local'):.0f}")
mac("kappaDsAz", f2(kappa("judge_ds", "judge_az"))); mac("kappaDsLoc", f2(kappa("judge_ds", "judge_local"))); mac("kappaAzLoc", f2(kappa("judge_az", "judge_local")))
L = [r"\begin{table}[h]\centering\small", r"\caption{Faithfulness audit of \nAuditBig{} cached answers, 25 per forum at each evidence depth and order of sampling, judged by three models against the root thread's human answer (ABSTAIN counted as NO). Cohen's $\kappa$: DeepSeek and gpt-5.4-mini \kappaDsAz, DeepSeek and gemma \kappaDsLoc, gpt-5.4-mini and gemma \kappaAzLoc. Trend $p$: Cochran--Armitage test for the share of NO across depths.}\label{tab:audit}",
     r"\begin{tabular}{lrrrrrrrrr}", r"\toprule", r" & \multicolumn{3}{c}{DeepSeek-V4.1-Flash} & \multicolumn{3}{c}{gpt-5.4-mini} & \multicolumn{3}{c}{gemma-4-26b} \\",
     r"evidence depth & yes & partly & no & yes & partly & no & yes & partly & no \\", r"\midrule"]
trend = {}
for j in ("judge_ds", "judge_az", "judge_local"):
    ks = np.array([sum(x[j] == "NO" for x in A3 if min(x["depth"], 2) == d) for d in (0, 1, 2)]); ns = np.array([sum(1 for x in A3 if min(x["depth"], 2) == d) for d in (0, 1, 2)])
    sc = np.arange(3); p = ks.sum() / ns.sum(); Tz = (sc * (ks - ns * p)).sum(); V = p * (1 - p) * ((ns * sc ** 2).sum() - (ns * sc).sum() ** 2 / ns.sum())
    trend[j] = float(2 * (1 - stats.norm.cdf(abs(Tz / np.sqrt(V)))))
for d, dn in ((0, "0 (thread)"), (1, "1"), (2, r"2 or more")):
    S = [x for x in A3 if min(x["depth"], 2) == d]; row = [dn]
    for j in ("judge_ds", "judge_az", "judge_local"):
        for l in ("YES", "PARTLY", "NO"): row.append(f"{100 * np.mean([x[j] == l for x in S]):.0f}\\%")
    L.append(" & ".join(row) + r" \\")
L += [r"\midrule", "trend $p$ (no) & & & " + f"{trend['judge_ds']:.3f}" + " & & & " + f"{trend['judge_az']:.3f}" + " & & & " + f"{trend['judge_local']:.3f}" + r" \\",
      r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write("tab_audit3.tex", L)
mac("auditTrendDs", f"{trend['judge_ds']:.3f}"); mac("auditTrendAz", f"{trend['judge_az']:.3f}"); mac("auditTrendLoc", f"{trend['judge_local']:.3f}")

# ---- effect sizes against the ceiling (content-aware user) --------------------------------------------------------------
E = {k: v for k, v in T["effect"].items() if k.endswith("|content")}
el = [v["ceil0.95"]["eligible"] for v in E.values()]; cap1 = [v["ceil0.1"]["captured"] for v in E.values()]; cap9 = [v["ceil0.95"]["captured"] for v in E.values()]
lost9 = [v["ceil0.95"]["lost"] for v in E.values()]; lost1 = [v["ceil0.1"]["lost"] for v in E.values()]; per9 = [v["ceil0.95"]["per1000"] for v in E.values()]; per1 = [v["ceil0.1"]["per1000"] for v in E.values()]
mac("fixMin", f"{min(el):.0f}"); mac("fixMax", f"{max(el):.0f}"); mac("capLowMean", f1(np.mean(cap1))); mac("capHighMean", f1(np.mean(cap9)))
mac("capLowMax", f1(max(cap1))); mac("lostHighMean", f1(np.mean(lost9))); mac("lostLowMean", f1(np.mean(lost1)))
mac("netHighMin", f"{min(per9):+.1f}"); mac("netHighMax", f"{max(per9):+.1f}"); mac("netLowMin", f"{min(per1):+.1f}"); mac("netLowMax", f"{max(per1):+.1f}")
ex0 = sum(1 for v in E.values() if v["d0.95"]["lo"] <= 0 <= v["d0.95"]["hi"]); mac("nCiZeroHigh", str(ex0))
ex1 = sum(1 for v in E.values() if v["d0.1"]["lo"] > 0); mac("nPosLow", str(ex1)); mac("nConfigs", str(len(E)))
L = [r"\begin{table}[t]\centering\small", r"\caption{How much is at stake, content-aware user, question and generated answer entries. Fixable: share of questions that static retrieval answers wrongly and that have an earlier, correctly answered member of their family. Fixed: share of fixable questions answered correctly with write-back. Broken: share of statically correct questions answered wrongly. Net: change in correct answers per 1{,}000 questions. $\Delta$: points, 95\% interval from a joint bootstrap over orders and families; the last column drops the largest family.}\label{tab:effect}",
     r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{llrrrrrrrr}", r"\toprule", r"forum & retriever & static & fixable & fixed 0.1 / 0.95 & other fixed 0.95 & broken 0.95 & net 0.95 & $\Delta$ at 0.95 & without largest \\", r"\midrule"]
for s, r in CONFIGS:
    v = E.get(f"{s}|{r}|content")
    if not v: continue
    b = T["base"][f"{s}|{r}"]["cacc"] * 100; d = v["d0.95"]; dl = v["d0.95_nolargest"]
    L.append(f"{s} & {NAME[r]} & {b:.1f}\\% & {v['ceil0.95']['eligible']:.1f}\\% & {v['ceil0.1']['captured']:.1f} / {v['ceil0.95']['captured']:.1f}\\% & {v['ceil0.95']['other']:.1f}\\% & {v['ceil0.95']['lost']:.1f}\\% & {v['ceil0.95']['per1000']:+.1f} & "
             f"{d['est']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}] & {dl['est']:+.2f} [{dl['lo']:+.2f}, {dl['hi']:+.2f}] \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
write("tab3_effect.tex", L)

# ---- H1 --------------------------------------------------------------------------------------------------------------
def h1(s, r, v, u): return T["h1"].get(f"{s}|{r}|{v}|{u}")
for u, U in (("content", "C"), ("root", "R")):
    for v, V in (("askgen", "AskGen"), ("gen", "Gen"), ("genthr", "GenThr"), ("askhum", "AskHum"), ("hum", "Hum"), ("p1_ask", "POneAsk"), ("p1_askdoc", "POneAskDoc")):
        X = [h1(s, r, v, u if not v.startswith("p1_") else "root") for s, r in CONFIGS]; X = [x for x in X if x]
        if not X: continue
        mac(f"nSafe{V}{U}", str(sum(x["cross"]["kind"] == ">" for x in X)))
        mac(f"nRobust{V}{U}", str(sum(x["cross"]["kind"] == ">" and x["cross"].get("lo", 0) >= 0.95 for x in X)))
        mac(f"stealMed{V}{U}", f"{np.median([x['steal95'] for x in X]):.1f}")
for u, U in (("root", "R"), ("content", "C")):
    for gv, hv, V in (("askgen", "askhum", "Ask"), ("gen", "hum", "Ans")):
        c = T["h1ctrl"].get(f"{gv}|{hv}|{u}")
        if not c: continue
        for r in RETS:
            b = c["by_ret"].get(r)
            if not b: continue
            R = NAME[r].replace("BM25", "Bm").replace("MiniLM", "Mini").replace("bge", "Bge")
            mac(f"ctrlSteal{V}{R}{U}", f"{b['steal']['mean']:+.1f}"); mac(f"ctrlCap{V}{R}{U}", f"{b['capture']['mean']:+.1f}"); mac(f"ctrlDelta{V}{R}{U}", f"{b['delta']['mean']:+.2f}")
            mac(f"ctrlStealSig{V}{R}{U}", f"{b['steal']['sig_pos']}/{b['steal']['sig_neg']}/{b['steal']['n']}")
            mac(f"ctrlDeltaSig{V}{R}{U}", f"{b['delta']['sig_pos']}/{b['delta']['sig_neg']}/{b['delta']['n']}")
        rows = c["rows"]
        mac(f"ctrlStealPos{V}{U}", str(sum(x["steal"] > 0 for x in rows))); mac(f"ctrlStealSigPos{V}{U}", str(sum(x["steal"] > 0 and x["q_steal"] < 0.05 for x in rows)))
        mac(f"ctrlStealSigNeg{V}{U}", str(sum(x["steal"] < 0 and x["q_steal"] < 0.05 for x in rows)))
        mac(f"ctrlDeltaSigNeg{V}{U}", str(sum(x["delta"] < 0 and x["q_delta"] < 0.05 for x in rows))); mac(f"ctrlDeltaSigPos{V}{U}", str(sum(x["delta"] > 0 and x["q_delta"] < 0.05 for x in rows)))
        mac(f"ctrlN{V}{U}", str(len(rows)))
L = [r"\begin{table}[t]\centering\small", r"\caption{Source-bias control, root-label user, $p_w=0.95$. Each generated design is compared with the same entries carrying the root thread's human answer cut to the same length. Steal: correct static top-1 displaced by an entry; capture: an entry outscores the correct thread; both in \%. Differences are generated minus human, averaged over the four forums. Last column: configurations where the paired difference in steal over orders is significant after Holm correction, positive / negative / total.}\label{tab:h1ctrl}",
     r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{llrrrrrr}", r"\toprule", r"design pair & retriever & steal (gen.) & steal (human) & $\Delta$steal & $\Delta$capture & $\Delta$ (points) & sig.\ $\Delta$steal \\", r"\midrule"]
for gv, hv, lab in (("askgen", "askhum", "question + answer"), ("gen", "hum", "answer only")):
    c = T["h1ctrl"].get(f"{gv}|{hv}|root")
    if not c: continue
    for r in RETS:
        b = c["by_ret"].get(r)
        if not b: continue
        sg = np.mean([h1(s, r, gv, "root")["steal95"] for s in SUBS if h1(s, r, gv, "root")]); sh = np.mean([h1(s, r, hv, "root")["steal95"] for s in SUBS if h1(s, r, hv, "root")])
        L.append(f"{lab} & {NAME[r]} & {sg:.1f} & {sh:.1f} & {b['steal']['mean']:+.1f} & {b['capture']['mean']:+.1f} & {b['delta']['mean']:+.2f} & {b['steal']['sig_pos']}/{b['steal']['sig_neg']}/{b['steal']['n']} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"; L += [r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
write("tab3_h1ctrl.tex", L)

# user models
U = T["users"]
mac("nUserEarlier", str(sum(1 for v in U.values() if (v["content"]["cross"]["p"] if v["content"]["cross"]["kind"] != ">" else 1.0) < (v["root"]["cross"]["p"] if v["root"]["cross"]["kind"] != ">" else 1.0) - 1e-9)))
mac("dHighRoot", f"{np.mean([v['root']['d95'][0] for v in U.values()]):+.2f}"); mac("dHighContent", f"{np.mean([v['content']['d95'][0] for v in U.values()]):+.2f}")
mac("dLowRoot", f"{np.mean([v['root']['d01'][0] for v in U.values()]):+.2f}"); mac("dLowContent", f"{np.mean([v['content']['d01'][0] for v in U.values()]):+.2f}")

# ---- H2 --------------------------------------------------------------------------------------------------------------
H = {k: v for k, v in T["h2"].items() if k.endswith("|0.95")}
deep = [100 * (v["ent_d"][1] + v["ent_d"][2]) / max(sum(v["ent_d"]), 1) for v in H.values()]
mac("deepShareMin", f"{min(deep):.0f}"); mac("deepShareMax", f"{max(deep):.0f}")
mac("selfDeepLastMax", f1(max(100 * v["self_d2_by_round"][-1] for v in H.values())))
y = T["h2_test"]["yes"]; mac("yesDZero", f"{100 * y[0]:.0f}"); mac("yesDOne", f"{100 * y[1]:.0f}"); mac("yesDTwo", f"{100 * y[2]:.0f}")
pt = T["h2_test"]["p_d0_d2"]; mac("yesTestP", "--" if pt is None else ("$<$0.001" if pt < 0.001 else f"{pt:.3f}"))
mac("nServedDeep", f"{T['h2_test']['n'][2]:,}")
harm = np.array([v["harm"] for v in H.values()])
pass
mac("harmDTwoMin", f"{harm[:, 1].min():+.2f}"); mac("harmDTwoMax", f"{harm[:, 1].max():+.2f}")
cw = [100 * v["cwrong_self"][1] / max(v["cwrong"], 1) for v in H.values()]; cw1 = [100 * v["cwrong_self"][0] / max(v["cwrong"], 1) for v in H.values()]
mac("wrongDeepMin", f1(min(cw))); mac("wrongDeepMax", f1(max(cw))); mac("wrongOneMin", f1(min(cw1))); mac("wrongOneMax", f1(max(cw1)))
dr = np.array([[x if x is not None else np.nan for x in v["drift"]] for v in H.values()]); mac("driftZero", f2(np.nanmean(dr[:, 0]))); mac("driftOne", f2(np.nanmean(dr[:, 1]))); mac("driftTwo", f2(np.nanmean(dr[:, 2])))

# ---- H3 --------------------------------------------------------------------------------------------------------------
L = [r"\begin{table}[t]\centering\small", r"\caption{Predicting the crossover on held-out orders (calibrate on five orders, evaluate on the other five, both directions; root-label user). MAE over configuration--split pairs where both the observed and the predicted crossover lie inside the grid, with a bootstrap 95\% interval. Mean baseline: predicting every observed crossover by their mean. Paired: the two estimates on the pairs where both are defined, with a Wilcoxon signed-rank test.}\label{tab:h3}",
     r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{llrrrrrr}", r"\toprule", r"design & $p_{\rm ref}$ & $n$ & first-order MAE & depth-aware MAE & mean baseline & Spearman (first-order) & Wilcoxon $p$ \\", r"\midrule"]
for v in ("askgen", "gen", "genthr"):
    for pref in (0.3, 0.5):
        d = T["h3"].get(f"{v}|{pref}")
        if not d or not d.get("p1"): continue
        p1, p2, pa = d["p1"], d.get("p2"), d.get("paired")
        L.append(f"{v.replace('askgen', 'question + answer').replace('genthr', 'answer + thread').replace('gen', 'answer only')} & {pref} & {p1['n']} & {p1['mae']:.3f} [{p1['lo']:.3f}, {p1['hi']:.3f}] & "
                 + (f"{p2['mae']:.3f} [{p2['lo']:.3f}, {p2['hi']:.3f}]" if p2 else "--") + " & " + (f"{pa['mean_baseline']:.3f}" if pa else "--") + " & "
                 + (f"{p1['spearman']:.2f}" if p1.get("spearman") is not None else "--") + " & " + (f"{pa['p']:.2f}" if pa and pa.get("p") is not None else "--") + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
write("tab3_h3.tex", L)
d = T["h3"].get("askgen|0.3", {})
if d.get("p1"): mac("maeOneThree", f"{d['p1']['mae']:.3f}"); mac("nThree", str(d["p1"]["n"]))
if d.get("p2"): mac("maeTwoThree", f"{d['p2']['mae']:.3f}")
if d.get("paired"): mac("maeBaseThree", f"{d['paired']['mean_baseline']:.3f}"); mac("wilcoxThree", f"{d['paired']['p']:.2f}" if d["paired"].get("p") is not None else "--")
d = T["h3"].get("askgen|0.5", {})
if d.get("p1"): mac("maeOneFive", f"{d['p1']['mae']:.3f}")
if d.get("p2"): mac("maeTwoFive", f"{d['p2']['mae']:.3f}")

# ---- H4 --------------------------------------------------------------------------------------------------------------
S4 = T["h4_summary"]
RN = [("naive", "naive write-back"), ("reopen", "reopen, oracle (Guthikonda, 2026)"), ("reopen_d1", "reopen after 1 round"), ("reopen_d2", "reopen after 2 rounds"),
      ("reopen_d1_f10", "reopen after 1 round, 10\\% false"), ("reopen_d1_f20", "reopen after 1 round, 20\\% false"), ("cap1", "depth cap (depth 1 only)"),
      ("regen", "regenerate from root"), ("prov0.5", "uniform penalty, $\\lambda=0.5$"), ("prov1.0", "uniform penalty, $\\lambda=1$"),
      ("provdepth0.5", "depth penalty, $\\lambda=0.5$"), ("provdepth1.0", "depth penalty, $\\lambda=1$")]
L = [r"\begin{table}[t]\centering\small", r"\caption{Countermeasures, content-aware user, question and generated answer entries, mean over the twelve configurations. AUC: mean $\Delta$ over the $p_w$ grid. Safe / robust: configurations with no crossover below 0.95, and those where in addition no bootstrap resample of the orders crosses earlier. Better / worse: configurations where the rule is significantly better or worse than naive write-back at $p_w=0.95$, and significantly worse at $p_w=0.1$ (paired over orders, Holm-corrected).}\label{tab:h4}",
     r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{lrrrrrr}", r"\toprule", r"rule & $\Delta$ at 0.1 & $\Delta$ at 0.95 & AUC & safe / robust & better / worse at 0.95 & worse at 0.1 \\", r"\midrule"]
for k, lab in RN:
    x = S4.get(k)
    if not x: continue
    L.append(f"{lab} & {x['d01']:+.2f} & {x['d95']:+.2f} & {x['auc']:+.2f} & {x['safe']} / {x['safe_lb']} & " + (f"{x['better95']} / {x['worse95']} & {x['worse01']}" if k != "naive" else "-- & --") + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
write("tab3_h4.tex", L)
for k, V in (("naive", "Naive"), ("reopen", "Oracle"), ("reopen_d1", "DOne"), ("reopen_d2", "DTwo"), ("reopen_d1_f10", "FTen"), ("reopen_d1_f20", "FTwenty"),
             ("cap1", "Cap"), ("regen", "Regen"), ("prov0.5", "ProvHalf"), ("prov1.0", "ProvOne"), ("provdepth0.5", "DepHalf"), ("provdepth1.0", "DepOne")):
    x = S4.get(k)
    if not x: continue
    mac(f"hfour{V}Low", f"{x['d01']:+.2f}"); mac(f"hfour{V}High", f"{x['d95']:+.2f}"); mac(f"hfour{V}Auc", f"{x['auc']:+.2f}"); mac(f"hfour{V}Safe", str(x["safe"]))
    if k != "naive": mac(f"hfour{V}Better", str(x["better95"])); mac(f"hfour{V}Worse", str(x["worse95"])); mac(f"hfour{V}WorseLow", str(x["worse01"]))
if T["reopen_equiv"]["mean_abs"] is not None: mac("reopenEquiv", f"{T['reopen_equiv']['mean_abs']:.2f}")

# ---- robustness: other generator and prompt ----------------------------------------------------------------------------
for tab, P in ((TL, "Loc"), (TO, "Open")):
    if not tab: continue
    X = [v for k, v in tab["h1"].items() if k.endswith("|askgen|content")]
    mac(f"nSafeAskGen{P}", str(sum(x["cross"]["kind"] == ">" for x in X))); mac(f"nCfg{P}", str(len(X)))
    Hh = {k: v for k, v in tab["h2"].items() if k.endswith("|0.95")}
    if Hh:
        yy = tab["h2_test"]["yes"]; mac(f"yesDZero{P}", f"{100 * yy[0]:.0f}"); mac(f"yesDTwo{P}", f"{100 * yy[2]:.0f}")
        pass
    Uo = tab.get("users", {})
    if Uo: mac(f"dHighContent{P}", f"{np.mean([v['content']['d95'][0] for v in Uo.values()]):+.2f}"); mac(f"dLowContent{P}", f"{np.mean([v['content']['d01'][0] for v in Uo.values()]):+.2f}")

# ---- appendix: full per-configuration tables ------------------------------------------------------------------------------
L = [r"\begin{table}[h]\centering\scriptsize", r"\caption{Entry designs by user model and configuration: crossover $p^*$ with bootstrap interval, $\Delta$ (points, mean $\pm$ 95\% interval over orders) at $p_w=0.1$ and $0.95$, share of answers served from entries and steal rate at $p_w=0.95$. Pointer rows are from \citet{guthikonda2026accepted} (root label).}\label{tab:fullh1}",
     r"\begin{adjustbox}{max width=\linewidth, max totalheight=0.9\textheight}", r"\begin{tabular}{lllrrrrr}", r"\toprule",
     r"config & user & design & $p^*$ [95\% interval] & $\Delta$ at 0.1 & $\Delta$ at 0.95 & self & steal \\", r"\midrule"]
DES = [("p1_ask", "root", "question only (pointer)"), ("p1_askdoc", "root", "question + thread (pointer)")] + [(v, u, n) for u in ("root", "content") for v, n in
       (("gen", "answer"), ("askgen", "question + answer"), ("genthr", "answer + thread"), ("hum", "human answer (ctrl)"), ("askhum", "question + human answer (ctrl)"))]
for s, r in CONFIGS:
    for v, u, n in DES:
        d = h1(s, r, v, u)
        if not d: continue
        c = d["cross"]; ci_ = f" [{c['lo']:.2f}, {c['hi']:.2f}]" if "lo" in c else ""
        L.append(f"{s} {NAME[r]} & {u} & {n} & {cx(c)}{ci_} & {pm(d['d0.1']) if 'd0.1' in d else '--'} & {pm(d['d0.95'])} & {d['self95']:.0f}\\% & {d['steal95']:.1f}\\% \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"; L += [r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
L += [r"\begin{table}[h]\centering\scriptsize", r"\caption{Countermeasures by configuration, content-aware user: $\Delta$ (points) at $p_w=0.1$ and $0.95$ and the crossover.}\label{tab:fullh4}",
      r"\begin{adjustbox}{max width=\linewidth, max totalheight=0.9\textheight}", r"\begin{tabular}{ll" + "r" * len(RN) + "}", r"\toprule",
      "config & & " + " & ".join(f"\\rotatebox{{70}}{{{lab}}}" for _, lab in RN) + r" \\", r"\midrule"]
for s, r in CONFIGS:
    d = T["h4"].get(f"{s}|{r}", {})
    for key, lab in (("d0.1", "$\\Delta$ 0.1"), ("d0.95", "$\\Delta$ 0.95"), ("cross", "$p^*$")):
        row = [f"{s} {NAME[r]}" if key == "d0.1" else "", lab]
        for k, _ in RN:
            x = d.get(k)
            row.append("--" if not x else (cx(x["cross"]) if key == "cross" else f"{x[key][0]:+.2f}"))
        L.append(" & ".join(row) + r" \\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"; L += [r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
write("tab_full.tex", L)

for tab, P in ((T, ""), (TO, "Open"), (TL, "Loc")):
    if not tab or "h2_harm_pool" not in tab: continue
    hp = tab["h2_harm_pool"]
    for j, D in enumerate(("One", "Two", "Three")):
        mac(f"harmD{D}{P}", f"{hp['est'][j]:+.2f}"); mac(f"harmD{D}Lo{P}", f"{hp['lo'][j]:+.2f}"); mac(f"harmD{D}Hi{P}", f"{hp['hi'][j]:+.2f}")
    ct = tab["h2_cfg_test"]; mac(f"yesCfgLower{P}", str(ct["lower"])); mac(f"yesCfgN{P}", str(ct["n"]))
    mac(f"yesCfgP{P}", "--" if ct["p"] is None else (f"{ct['p']:.4f}" if ct["p"] < 0.01 else f"{ct['p']:.2f}"))
# main prompt on the two open-prompt forums, for a like-for-like comparison
cc = {d: collections.Counter() for d in ("0", "1", "2")}
for k, v in T["h2"].items():
    if k.split("|")[0] in ("android", "webmasters") and k.endswith("|0.95"):
        for d in cc: cc[d].update(v["labels"][d])
yy = [100 * cc[d]["YES"] / max(sum(cc[d][x] for x in ("YES", "PARTLY", "NO", "ABSTAIN")), 1) for d in ("0", "1", "2")]
mac("yesDZeroAW", f"{yy[0]:.0f}"); mac("yesDTwoAW", f"{yy[2]:.0f}")
oth = [v["ceil0.95"]["other"] for v in E.values()]; mac("otherHighMean", f1(np.mean(oth)))
DF = json.load(open(os.path.join(RES, "drift_floor.json"))); mac("driftFloor", f2(np.mean([v["floor"] for v in DF.values()])))
open(os.path.join(OUTD, "macros.tex"), "w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())) + "\n")
print(len(M), "macros written")

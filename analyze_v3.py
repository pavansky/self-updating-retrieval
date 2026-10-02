"""Version-3 analysis: results/sim_v2_main.json (+ sim_v2_het.json) and results/sim_v3_*.json
-> results/tables_v3.json, tex/tab_*_v3.tex, tex/macros_v3.tex, figs/*_v3.pdf. Every v3 number in main_v3.tex comes from here."""
import json, os, sys, collections
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
from analyze import stream, ci, crossover
FIG = os.path.join(HERE, "figs"); TEX = os.path.join(HERE, "tex")
NAME = {"bge": "bge", "minilm": "MiniLM", "bm25": "BM25"}; TAG = {"bge": "Bge", "minilm": "Mini", "bm25": "Bm"}
VNAME = {"ask": "ask text", "doc": "shown thread text", "askdoc": "ask + thread text"}
PWS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]; SEEDS = list(range(10))
FILTER_Q = [0.25, 0.5, 0.75]; CACHE_Q = [0.5, 0.75, 0.9, 0.95]; HET = "msmarcoh"
rng = np.random.default_rng(0)

G = collections.defaultdict(dict); SANITY = []
def ingest(fn, v3):
    p = os.path.join(RES, fn)
    if not os.path.exists(p): print("missing", fn); return None
    J = json.load(open(p))
    for r in J["runs"]:
        k = (r["sub"], r["ret"], r["variant"], r["user"], r["cond"], r["p_w"], r.get("param"))
        if v3 and r["cond"] == "naive": SANITY.append((k, r["seed"], r["curve"])); continue
        G[k][r["seed"]] = r["curve"]
    return {x: J[x] for x in J if x != "runs"}
META = dict(v2main=ingest("sim_v2_main.json", False), v3main=ingest("sim_v3_main.json", True),
            v2het=ingest("sim_v2_het.json", False), v3het=ingest("sim_v3_het.json", True))
# sanity: sim_v3 naive must equal sim_v2 naive
mism = sum(1 for k, sd, cu in SANITY if [x["acc"] for x in cu] != [x["acc"] for x in G[k[:6] + (None,)][sd]])
print("sanity naive runs:", len(SANITY), "mismatches:", mism)

def static(s, r): return G[(s, r, "ask", "const", "static", None, None)]
def delta(s, r, v, cond, p, q=None, key="acc"):
    st = static(s, r); runs = G[(s, r, v, "const", cond, p, q)]
    return [100 * (stream(runs[sd], key) - stream(st[sd], key)) for sd in SEEDS]
def cross(s, r, v, cond, q=None):
    M = np.array([delta(s, r, v, cond, p, q) for p in PWS]); x, kind = crossover(PWS, M.mean(1))
    b = [crossover(PWS, M[:, rng.integers(0, 10, 10)].mean(1))[0] for _ in range(1000)]
    return dict(p=float(x), kind=kind, lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)))
def tot(runs, key): return float(np.mean([sum(x[key] for x in runs[sd]) for sd in SEEDS]))
def pclip(c): return 1.0 if c["kind"] == ">" else min(max(c["p"], 0.1), 0.95)
def cx(c): return {"<": r"$<$0.10", ">": r"$>$0.95"}.get(c["kind"], f"{c['p']:.2f}")
def cxi(c): return {"<": r"$<$0.10", ">": r"$>$0.95"}.get(c["kind"], f"{c['p']:.2f} [{c['lo']:.2f}, {c['hi']:.2f}]")
def pm(v): return f"{v[0]:+.2f} $\\pm$ {v[1]:.2f}"
def med(ps): m = float(np.median(ps)); return r"$>$0.95" if m > 0.95 else (r"$<$0.10" if m < 0.1 else f"{m:.2f}")
def mac(n, v): return f"\\newcommand{{\\{n}}}{{{v}}}"
QN = {0.25: "Q", 0.5: "H", 0.75: "T", 0.9: "N", 0.95: "F"}   # macro tags for quantiles
T = dict(cfg={}, sum={}); M = []

CONFIGS = [(s, r) for s in FORUMS for r in RETRIEVERS if (s, r, "ask", "const", "filter", 0.95, 0.5) in G]
DENSE = [(s, r) for s, r in CONFIGS if r != "bm25"]; BM = [(s, r) for s, r in CONFIGS if r == "bm25"]

# ---------- per-config numbers ----------
def cfg_rows(s, r):
    out = {}
    for v in ("ask", "askdoc"):
        rules = [("naive", None, PWS), ("reopen", None, PWS), ("gate", None, [0.3, 0.6, 0.8, 0.95]), ("footprint", None, [0.3, 0.6, 0.8, 0.95])]
        rules += [("filter", q, PWS) for q in FILTER_Q]
        if v == "ask": rules += [("cache", q, PWS) for q in CACHE_Q]
        for cond, q, grid in rules:
            if (s, r, v, "const", cond, 0.95, q) not in G: continue
            k = f"{s}|{r}|{v}|{cond}|{q}"; d = {}
            d["d95"] = ci(delta(s, r, v, cond, 0.95, q)); d["d30"] = ci(delta(s, r, v, cond, 0.3, q))
            d["cross"] = cross(s, r, v, cond, q) if grid is PWS else None
            run = G[(s, r, v, "const", cond, 0.95, q)]
            d["entries"] = float(np.mean([run[sd][-1]["entries"] for sd in SEEDS])); d["ent_c"] = tot(run, "ent_c"); d["ent_w"] = tot(run, "ent_w")
            if cond == "cache":
                d["hit"] = tot(run, "hits") / tot(run, "n"); d["hit_prec"] = tot(run, "hit_ok") / max(tot(run, "hits"), 1e-9)
                r3 = G[(s, r, v, "const", cond, 0.3, q)]; d["hit30"] = tot(r3, "hits") / tot(r3, "n"); d["hit_prec30"] = tot(r3, "hit_ok") / max(tot(r3, "hits"), 1e-9)
                Sf = np.load(os.path.join(CACHE, f"{s}_{r}.npz"))["S"]; Sf = np.where(np.isfinite(Sf), Sf, -np.inf); d["tau"] = float(np.quantile(Sf.max(1), q))
            if cond == "filter": d["filtered"] = tot(run, "filtered")
            out[k] = d
    for k in list(out):
        s_, r_, v_, c_, q_ = k.split("|")
        nv = out[f"{s_}|{r_}|{v_}|naive|None"]
        out[k]["kept"] = out[k]["entries"] / max(nv["entries"], 1); out[k]["kept_c"] = out[k]["ent_c"] / max(nv["ent_c"], 1); out[k]["kept_w"] = out[k]["ent_w"] / max(nv["ent_w"], 1)
    return out
for s, r in CONFIGS + ([(HET, r) for r in RETRIEVERS if (HET, r, "ask", "const", "naive", 0.95, None) in G]):
    T["cfg"].update(cfg_rows(s, r))

# ---------- Table: filter vs other rules (dense mean; BM25 mean) ----------
RN = [("naive", None, "naive"), ("reopen", None, "+demotion"), ("gate", None, "confidence gate"), ("footprint", None, "footprint rule")] + \
     [("filter", q, f"CE filter, $q={q}$") for q in FILTER_Q]
L = [r"\begin{tabular}{llrrrrlrrr}\toprule Entry content & Rule & $\Delta$@0.3 & $\Delta$@0.95 & $\ge0$@0.95 & sig.\ $<0$@0.95 & median $p^*$ & kept & correct kept & wrong kept\\\midrule"]
for v in ("ask", "askdoc"):
    for cond, q, nm in RN:
        A = [T["cfg"][f"{s}|{r}|{v}|{cond}|{q}"] for s, r in DENSE]
        d3 = np.mean([a["d30"][0] for a in A]); d95 = np.mean([a["d95"][0] for a in A])
        nn = sum(a["d95"][0] >= 0 for a in A); neg = sum(a["d95"][0] + a["d95"][1] < 0 for a in A)
        mp = med([pclip(a["cross"]) for a in A]) if A[0]["cross"] else "--"; kept = np.mean([a["kept"] for a in A])
        kc = np.mean([a["kept_c"] for a in A]); kw = np.mean([a["kept_w"] for a in A])
        B = [T["cfg"][f"{s}|{r}|{v}|{cond}|{q}"] for s, r in BM]; b95 = np.mean([a["d95"][0] for a in B])
        T["sum"][f"filt|{v}|{cond}|{q}"] = dict(d3=d3, d95=d95, nn=nn, neg=neg, med=mp, kept=kept, bm95=b95, bmneg=sum(a["d95"][0] + a["d95"][1] < 0 for a in B),
                                                bmmax=max(abs(a["d95"][0]) for a in B))
        L.append(f"{VNAME[v] if cond=='naive' else ''} & {nm} & {d3:+.2f} & {d95:+.2f} & {nn}/{len(DENSE)} & {neg}/{len(DENSE)} & {mp} & {100*kept:.0f}\\% & {100*kc:.0f}\\% & {100*kw:.0f}\\%\\\\")
        tg = {"ask": "Ask", "askdoc": "Ad"}[v] + {"naive": "Naive", "reopen": "Dem", "gate": "Gate", "footprint": "Foot", "filter": "Flt"}[cond] + (QN[q] if q else "")
        M += [mac(f"tF{tg}", f"{d95:+.2f}"), mac(f"tT{tg}", f"{d3:+.2f}"), mac(f"tNN{tg}", nn), mac(f"tNeg{tg}", neg), mac(f"tMed{tg}", mp), mac(f"tKept{tg}", f"{100*kept:.0f}"), mac(f"tKeptC{tg}", f"{100*kc:.0f}"), mac(f"tKeptW{tg}", f"{100*kw:.0f}"), mac(f"tBm{tg}", f"{b95:+.2f}")]
    if v == "ask": L.append(r"\addlinespace[2pt]")
L.append(r"\bottomrule\end{tabular}"); open(os.path.join(TEX, "tab_filter_v3.tex"), "w").write("\n".join(L))
bmx = max(T["sum"][f"filt|{v}|filter|{q}"]["bmmax"] for v in ("ask", "askdoc") for q in FILTER_Q)
M += [mac("tBmFltMaxAbs", f"{bmx:.2f}"), mac("nDenseT", len(DENSE))]
# per-retriever filter detail (ask entries, q=0.5)
for r in ("bge", "minilm"):
    A = [T["cfg"][f"{s}|{r}|ask|filter|0.5"] for s in FORUMS]; N0 = [T["cfg"][f"{s}|{r}|ask|naive|None"] for s in FORUMS]
    M += [mac(f"tFlt{TAG[r]}", f"{np.mean([a['d95'][0] for a in A]):+.2f}"), mac(f"tNaive{TAG[r]}", f"{np.mean([a['d95'][0] for a in N0]):+.2f}")]

# ---------- Table: semantic cache ----------
L = [r"\begin{tabular}{llrrrrrrl}\toprule Retriever & $\tau$ (quantile) & $\tau$ & hit rate & hit prec. & $\Delta$@0.3 & $\Delta$@0.95 & sig.\ $<0$@0.95 & median $p^*$\\\midrule"]
for r in RETRIEVERS:
    N0 = [T["cfg"][f"{s}|{r}|ask|naive|None"] for s in FORUMS]
    L.append(f"{NAME[r]} & naive write-back & -- & -- & -- & {np.mean([a['d30'][0] for a in N0]):+.2f} & {np.mean([a['d95'][0] for a in N0]):+.2f} & {sum(a['d95'][0]+a['d95'][1]<0 for a in N0)}/12 & {med([pclip(a['cross']) for a in N0])}\\\\")
    for q in CACHE_Q:
        A = [T["cfg"][f"{s}|{r}|ask|cache|{q}"] for s in FORUMS]
        k = dict(tau=np.mean([a["tau"] for a in A]), hit=np.mean([a["hit"] for a in A]), prec=np.mean([a["hit_prec"] for a in A]),
                 d3=np.mean([a["d30"][0] for a in A]), d95=np.mean([a["d95"][0] for a in A]), neg=sum(a["d95"][0] + a["d95"][1] < 0 for a in A),
                 pos3=sum(a["d30"][0] - a["d30"][1] > 0 for a in A), med=med([pclip(a["cross"]) for a in A]),
                 hit30=np.mean([a["hit30"] for a in A]), prec30=np.mean([a["hit_prec30"] for a in A]))
        T["sum"][f"cache|{r}|{q}"] = k
        L.append(f" & {q} & {k['tau']:.2f} & {100*k['hit']:.1f}\\% & {100*k['prec']:.0f}\\% & {k['d3']:+.2f} & {k['d95']:+.2f} & {k['neg']}/12 & {k['med']}\\\\")
        tg = TAG[r] + QN[q]
        M += [mac(f"cTau{tg}", f"{k['tau']:.2f}"), mac(f"cHit{tg}", f"{100*k['hit']:.1f}"), mac(f"cPrec{tg}", f"{100*k['prec']:.0f}"), mac(f"cT{tg}", f"{k['d3']:+.2f}"),
              mac(f"cF{tg}", f"{k['d95']:+.2f}"), mac(f"cNeg{tg}", k["neg"]), mac(f"cMed{tg}", k["med"]), mac(f"cPos{tg}", k["pos3"]),
              mac(f"cHitTh{tg}", f"{100*k['hit30']:.1f}"), mac(f"cPrecTh{tg}", f"{100*k['prec30']:.0f}")]
    if r != "bm25": L.append(r"\addlinespace[2pt]")
L.append(r"\bottomrule\end{tabular}"); open(os.path.join(TEX, "tab_cache_v3.tex"), "w").write("\n".join(L))
# base top-1 precision for comparison with hit precision
for r in RETRIEVERS:
    M += [mac(f"cBase{TAG[r]}", f"{100*np.mean([np.mean([stream(static(s, r)[sd], 'acc') for sd in SEEDS]) for s in FORUMS]):.1f}")]

# ---------- heterogeneous corpus ----------
HR = [r for r in RETRIEVERS if (HET, r, "ask", "const", "naive", 0.95, None) in G]
if HR:
    st = json.load(open(os.path.join(HERE, "data_het", HET, "stats.json")))
    M += [mac("hFam", f"{st['families']:,}".replace(",", "{,}")), mac("hAsks", f"{st['asks']:,}".replace(",", "{,}")), mac("hKB", f"{st['kb']:,}".replace(",", "{,}")),
          mac("hFamAvail", f"{st['families_available']:,}".replace(",", "{,}")), mac("hAskWords", f"{st['mean_ask_words']:.1f}"), mac("hKBWords", f"{st['mean_kb_words']:.0f}"),
          mac("hNbCos", f"{st.get('bge_cos_canon_to_id_neighbour', float('nan')):.2f}"), mac("hBgCos", f"{st.get('bge_cos_canon_to_random_bg', float('nan')):.2f}")]
    L = [r"\begin{tabular}{llrlllll}\toprule Retriever & Entry content & Top-1 & $\Delta$@0.1 & $\Delta$@0.3 & $\Delta$@0.95 & $p^*$ & self-share @0.95\\\midrule"]
    for r in HR:
        A = float(np.mean([stream(static(HET, r)[sd], "acc") for sd in SEEDS])); M.append(mac(f"hAcc{TAG[r]}", f"{100*A:.1f}"))
        for v in ("ask", "askdoc", "doc"):
            if (HET, r, v, "const", "naive", 0.95, None) not in G: continue
            c = cross(HET, r, v, "naive"); d1 = ci(delta(HET, r, v, "naive", 0.1)); d3 = ci(delta(HET, r, v, "naive", 0.3)); d95 = ci(delta(HET, r, v, "naive", 0.95))
            run = G[(HET, r, v, "const", "naive", 0.95, None)]
            ss = 100 * float(np.mean([stream(run[sd], "self_share", lo=5) for sd in SEEDS]))
            steal = 100 * sum(x["steal"] for sd in SEEDS for x in run[sd][5:]) / max(sum(x["cov_ok"] for sd in SEEDS for x in run[sd][5:]), 1)
            T["sum"][f"het|{r}|{v}"] = dict(acc=A, d1=d1, d3=d3, d95=d95, cross=c, self=ss, steal=steal)
            L.append(f"{NAME[r] if v=='ask' else ''} & {VNAME[v]} & {100*A if v=='ask' else '':{'.1f' if v=='ask' else ''}} & {pm(d1)} & {pm(d3)} & {pm(d95)} & {cxi(c)} & {ss:.0f}\\%\\\\")
            tg = TAG[r] + {"ask": "Ask", "askdoc": "Ad", "doc": "Doc"}[v]
            M += [mac(f"hF{tg}", f"{d95[0]:+.2f}"), mac(f"hFci{tg}", f"{d95[1]:.2f}"), mac(f"hT{tg}", f"{d3[0]:+.2f}"), mac(f"hO{tg}", f"{d1[0]:+.2f}"),
                  mac(f"hP{tg}", cx(c)), mac(f"hPi{tg}", cxi(c)), mac(f"hSelf{tg}", f"{ss:.0f}"), mac(f"hSteal{tg}", f"{steal:.1f}")]
        # rules on het: demotion, filter, cache
        for v, cond, q in (("ask", "reopen", None), ("askdoc", "reopen", None), ("ask", "filter", 0.5), ("askdoc", "filter", 0.5), ("ask", "cache", 0.9), ("ask", "cache", 0.5)):
            k = T["cfg"].get(f"{HET}|{r}|{v}|{cond}|{q}")
            if k:
                tg = TAG[r] + {"ask": "Ask", "askdoc": "Ad"}[v] + {"reopen": "Dem", "filter": "Flt", "cache": "Cache"}[cond] + (QN[q] if q else "")
                M += [mac(f"hF{tg}", f"{k['d95'][0]:+.2f}"), mac(f"hT{tg}", f"{k['d30'][0]:+.2f}"), mac(f"hP{tg}", cx(k["cross"]))]
                if cond == "cache": M += [mac(f"hHit{tg}", f"{100*k['hit']:.1f}"), mac(f"hPrec{tg}", f"{100*k['hit_prec']:.0f}")]
        L.append(r"\addlinespace[2pt]")
    L[-1] = r"\bottomrule\end{tabular}"; open(os.path.join(TEX, "tab_het_v3.tex"), "w").write("\n".join(L))
    # pre-simulation diagnostic as in v1: share of asks for which the best other ask of the same family / of a different family outscores the best KB passage
    for r in HR:
        z = np.load(os.path.join(CACHE, f"{HET}_{r}.npz")); S = z["S"]; af = z["afam"]; best = z["cand_sc"][:, 0]
        same = af[:, None] == af[None, :]; Sm = np.where(same, S, -np.inf).max(1); So = np.where(~same, S, -np.inf).max(1)
        M += [mac(f"hSame{TAG[r]}", f"{100*np.mean(Sm > best):.0f}"), mac(f"hOther{TAG[r]}", f"{100*np.mean(So > best):.0f}")]
    # same diagnostic on CQADupStack (bge), for comparison
for r in RETRIEVERS:
    v = []
    for s in FORUMS:
        z = np.load(os.path.join(CACHE, f"{s}_{r}.npz")); S = z["S"]; af = z["afam"]; best = z["cand_sc"][:, 0]
        same = af[:, None] == af[None, :]; v.append(100 * np.mean(np.where(~same, S, -np.inf).max(1) > best))
    M += [mac(f"cqOther{TAG[r]}", f"{np.mean(v):.0f}")]

M += [mac("vthreeSanityRuns", len(SANITY)), mac("vthreeSanityMis", mism)]
for k, m in META.items():
    if m: M += [mac(f"miss{k.replace('2','two').replace('3','three')}", m.get("missing_ce_pairs", m.get("missing_pairs", 0)))]
open(os.path.join(TEX, "macros_v3.tex"), "w").write("\n".join(M) + "\n")
json.dump(T, open(os.path.join(RES, "tables_v3.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

# ---------- figure: change vs p_w for rules (dense mean), and het ----------
fig, axs = plt.subplots(1, 3 if HR else 2, figsize=(9.5 if HR else 6.5, 2.8))
for ax, v in zip(axs, ("ask", "askdoc")):
    for cond, q, lab, colr, ls in (("naive", None, "naive", "#c05621", "-"), ("reopen", None, "+demotion", "#2b6cb0", "-"),
                                    ("filter", 0.5, "CE filter q=0.5", "#2f855a", "-"), ("filter", 0.75, "CE filter q=0.75", "#2f855a", "--")) + \
                                   ((("cache", 0.5, "cache tau q=0.5", "#6b46c1", "-"), ("cache", 0.9, "cache tau q=0.9", "#6b46c1", "--")) if v == "ask" else ()):
        y = [np.mean([np.mean(delta(s, r, v, cond, p, q)) for s, r in DENSE]) for p in PWS]
        ax.plot(PWS, y, ls, marker="o", ms=2.3, lw=1, color=colr, label=lab)
    ax.axhline(0, color="k", lw=0.5); ax.set_title(f"CQADupStack, {VNAME[v]} (dense)", fontsize=8.5); ax.tick_params(labelsize=7); ax.set_xlabel("$p_w$", fontsize=8)
axs[0].set_ylabel("mean change in top-1 (points)", fontsize=8); axs[0].legend(fontsize=6, frameon=False)
if HR:
    ax = axs[2]
    for r, colr in (("bge", "#2b6cb0"), ("minilm", "#2f855a"), ("bm25", "#718096")):
        if r not in HR: continue
        for v, ls in (("ask", "-"), ("askdoc", "--")):
            ax.plot(PWS, [np.mean(delta(HET, r, v, "naive", p)) for p in PWS], ls, marker="o", ms=2.3, lw=1, color=colr, label=f"{NAME[r]}, {VNAME[v]}")
    ax.axhline(0, color="k", lw=0.5); ax.set_title("MS MARCO families, naive", fontsize=8.5); ax.tick_params(labelsize=7); ax.set_xlabel("$p_w$", fontsize=8); ax.legend(fontsize=5.5, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "rules_v3.pdf")); plt.close(fig)
print("\n".join(M))

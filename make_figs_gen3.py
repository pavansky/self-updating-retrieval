"""Paper-2 (TMLR version) figures from results/tables_gen3*.json, results/tables_v2.json and the audit files.
Writes figs/g3_*.pdf.   python make_figs_gen3.py
"""
import json, os, math, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import HERE, RES

FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
T = json.load(open(os.path.join(RES, "tables_gen3.json")))
TL = json.load(open(os.path.join(RES, "tables_gen3_local.json"))) if os.path.exists(os.path.join(RES, "tables_gen3_local.json")) else None
TO = json.load(open(os.path.join(RES, "tables_gen3_open.json"))) if os.path.exists(os.path.join(RES, "tables_gen3_open.json")) else None
SUBS = T["meta"]["subs"]; RETS = T["meta"]["rets"]
NAME = {"bge": "bge", "minilm": "MiniLM", "bm25": "BM25"}
PWS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
OI = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#999999", "#000000"]
C = {"askgen": OI[0], "gen": OI[1], "genthr": OI[2], "askhum": OI[5], "hum": OI[4], "p1_ask": OI[3], "p1_askdoc": OI[7]}
LAB = {"askgen": "question + generated answer", "gen": "generated answer", "genthr": "generated answer + cited thread",
       "askhum": "question + human answer (control)", "hum": "human answer (control)",
       "p1_ask": "question only (pointer, Guthikonda 2026)", "p1_askdoc": "question + thread (pointer, Guthikonda 2026)"}
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.3,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
def h1(s, r, v, u): return T["h1"].get(f"{s}|{r}|{v}|{u}")
def save(fig, name): fig.savefig(os.path.join(FIG, name), bbox_inches="tight"); plt.close(fig)

# Fig 2. Delta vs p_w, content-aware user, all configurations: designs and the root-label user for question+answer
fig, axs = plt.subplots(len(SUBS), len(RETS), figsize=(7.0, 1.85 * len(SUBS)), sharex=True, squeeze=False)
for i, s in enumerate(SUBS):
    for j, r in enumerate(RETS):
        ax = axs[i, j]
        for v in ("p1_ask", "p1_askdoc"):
            d = h1(s, r, v, "root")
            if d: ax.plot(PWS, [d[f"d{p}"][0] for p in PWS], "--", color=C[v], lw=0.8, label=LAB[v])
        for v in ("gen", "genthr", "askgen"):
            d = h1(s, r, v, "content")
            if not d: continue
            m = np.array([d[f"d{p}"][0] for p in PWS]); hw = np.array([d[f"d{p}"][1] for p in PWS])
            ax.plot(PWS, m, "-o", ms=1.8, lw=1.1 if v == "askgen" else 0.8, color=C[v], label=LAB[v])
            if v == "askgen": ax.fill_between(PWS, m - hw, m + hw, color=C[v], alpha=0.15, lw=0)
        d = h1(s, r, "askgen", "root")
        if d: ax.plot(PWS, [d[f"d{p}"][0] for p in PWS], ":", color=C["askgen"], lw=1.0, label="question + generated answer, root-label user")
        ax.axhline(0, color="k", lw=0.5); ax.set_title(f"{s}, {NAME[r]}")
        if j == 0: ax.set_ylabel(r"$\Delta$ accuracy (points)")
        if i == len(SUBS) - 1: ax.set_xlabel(r"$p_w$")
h_, l_ = axs[0, 0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.03, 1, 1)); save(fig, "g3_delta_curves.pdf")

# Fig 3. Source-bias control: generated vs length-matched human text, paired by configuration
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3))
for k, (gv, hv) in enumerate((("askgen", "askhum"), ("gen", "hum"))):
    for j, r in enumerate(RETS):
        for s_i, s in enumerate(SUBS):
            a, b = h1(s, r, gv, "root"), h1(s, r, hv, "root")
            if not a or not b: continue
            x = j + (k - 0.5) * 0.3
            axs[0].plot([x - 0.06, x + 0.06], [b["steal95"], a["steal95"]], "-", color="#bbbbbb", lw=0.7)
            axs[0].scatter([x - 0.06], [b["steal95"]], s=10, color=C[hv], zorder=3, label=LAB[hv] if (j == 0 and s_i == 0) else None)
            axs[0].scatter([x + 0.06], [a["steal95"]], s=10, color=C[gv], zorder=3, label=LAB[gv] if (j == 0 and s_i == 0) else None)
            axs[1].plot([x - 0.06, x + 0.06], [b["capture95"], a["capture95"]], "-", color="#bbbbbb", lw=0.7)
            axs[1].scatter([x - 0.06], [b["capture95"]], s=10, color=C[hv], zorder=3); axs[1].scatter([x + 0.06], [a["capture95"]], s=10, color=C[gv], zorder=3)
for ax, t in ((axs[0], r"steal rate at $p_w=0.95$ (%)"), (axs[1], r"capture rate at $p_w=0.95$ (%)")):
    ax.set_xticks(range(len(RETS))); ax.set_xticklabels([NAME[r] for r in RETS]); ax.set_ylabel(t)
axs[0].set_title("Correct top-1 displaced by an entry"); axs[1].set_title("An entry outscores the correct thread")
axs[0].legend(frameon=False, fontsize=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
# paired difference in Delta at 0.95 (generated minus human), per configuration
for k, (gv, hv) in enumerate((("askgen", "askhum"), ("gen", "hum"))):
    rows = T["h1ctrl"].get(f"{gv}|{hv}|root", {}).get("rows", [])
    for x in rows:
        j = RETS.index(x["ret"]); xx = j + (k - 0.5) * 0.3
        sig = x["q_delta"] < 0.05
        axs[2].scatter([xx], [x["delta"]], s=14 if sig else 8, color=C[gv], marker="o" if sig else "x", lw=0.8)
axs[2].axhline(0, color="k", lw=0.5); axs[2].set_xticks(range(len(RETS))); axs[2].set_xticklabels([NAME[r] for r in RETS])
axs[2].set_ylabel(r"$\Delta_{\rm generated}-\Delta_{\rm human}$ at $p_w=0.95$"); axs[2].set_title("Paired difference (dots: Holm $q<0.05$)")
fig.tight_layout(); save(fig, "g3_source_bias.pdf")

# Fig 4. Depth: (a) served share by round, (b) judged label mix by served depth, (c) counterfactual change by depth
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7))
for p, ls in ((0.3, ":"), (0.95, "-")):
    for r_i, r in enumerate(RETS):
        rows = [T["h2"][f"{s}|{r}|{p}"] for s in SUBS if f"{s}|{r}|{p}" in T["h2"]]
        if not rows: continue
        a = np.mean([x["self_by_round"] for x in rows], 0) * 100; b = np.mean([x["self_d2_by_round"] for x in rows], 0) * 100
        rr = np.arange(1, len(a) + 1)
        axs[0].plot(rr, a, ls, color=OI[r_i], lw=0.9, label=f"{NAME[r]}, any entry, $p_w$={p}")
        axs[0].plot(rr, b, ls, color=OI[r_i], lw=1.8, alpha=0.5)
axs[0].set_xlabel("round (tenth of the stream)"); axs[0].set_ylabel("answers served from entries (%)")
axs[0].set_title("Answers served from entries"); axs[0].legend(frameon=False, fontsize=4.6, loc="upper left", ncol=1)
pool = T["h2_pool"]["0.95"]; L = ("YES", "PARTLY", "NO", "ABSTAIN"); cols = [OI[2], OI[1], OI[3], OI[7]]
left = np.zeros(3)
for l, col in zip(L, cols):
    tot = np.array([sum(pool[str(d)].get(x, 0) for x in L) for d in (0, 1, 2)], float)
    v = np.array([pool[str(d)].get(l, 0) for d in (0, 1, 2)]) / np.maximum(tot, 1) * 100
    axs[1].barh([0, 1, 2], v, left=left, color=col, label=l.lower()); left += v
axs[1].set_yticks([0, 1, 2]); axs[1].set_yticklabels(["thread", "depth 1", r"depth $\geq2$"]); axs[1].invert_yaxis()
axs[1].set_xlabel("served answers (%)"); axs[1].set_title("Judged content by depth"); axs[1].legend(frameon=False, fontsize=5, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.45))
for r_i, r in enumerate(RETS):
    H = np.array([T["h2"][f"{s}|{r}|0.95"]["harm"] for s in SUBS if f"{s}|{r}|0.95" in T["h2"]])
    if len(H): axs[2].bar(np.arange(3) + (r_i - 1) * 0.25, H.mean(0), 0.24, color=OI[r_i], label=NAME[r])
axs[2].axhline(0, color="k", lw=0.5); axs[2].set_xticks(range(3)); axs[2].set_xticklabels(["depth 1", "depth 2", "depth 3+"])
axs[2].set_ylabel("change per 1,000 questions"); axs[2].set_title("Change in correct answers by depth"); axs[2].legend(frameon=False, fontsize=5.5)
fig.tight_layout(); save(fig, "g3_depth.pdf")

# Fig 5. Crossover prediction out of sample
fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.6), sharey=True)
for ax, pref in zip(axs, (0.3, 0.5)):
    for v, mk in (("askgen", "o"), ("gen", "s"), ("genthr", "^")):
        d = T["h3"].get(f"{v}|{pref}")
        if not d: continue
        for tag, col in (("p1", OI[0]), ("p2", OI[3])):
            xs = [x["obs"] for x in d["rows"] if x["ko"] == "=" and x[f"k{tag[1]}"] == "="]; ys = [min(x[tag], 1) for x in d["rows"] if x["ko"] == "=" and x[f"k{tag[1]}"] == "="]
            ax.scatter(xs, ys, s=10, marker=mk, color=col, alpha=0.75, lw=0, label=f"{'first-order' if tag == 'p1' else 'depth-aware'}, {v}")
    ax.plot([0, 1], [0, 1], "k-", lw=0.5); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"observed $p^*$ (held-out orders)"); ax.set_title(f"calibrated at $p_w$={pref}")
axs[0].set_ylabel(r"predicted $p^*$"); axs[1].legend(frameon=False, fontsize=4.8)
fig.tight_layout(); save(fig, "g3_h3.pdf")

# Fig 6. Countermeasures: gain kept at p_w=0.1 against outcome at p_w=0.95 (content-aware user)
RN = {"naive": "naive", "reopen": "reopen (oracle, Guthikonda 2026)", "reopen_d1": "reopen after 1 round", "reopen_d2": "reopen after 2 rounds",
      "reopen_d1_f10": "reopen, 10% false", "reopen_d1_f20": "reopen, 20% false", "cap1": "depth cap", "regen": "regenerate from root",
      "prov0.5": r"uniform penalty $\lambda$=.5", "prov1.0": r"uniform penalty $\lambda$=1", "provdepth0.5": r"depth penalty $\lambda$=.5", "provdepth1.0": r"depth penalty $\lambda$=1"}
fig, ax = plt.subplots(figsize=(4.4, 3.2))
for k_i, (rule, lab) in enumerate(RN.items()):
    xs, ys = [], []
    for s in SUBS:
        for r in RETS:
            d = T["h4"].get(f"{s}|{r}", {}).get(rule)
            if d: xs.append(d["d0.1"][0]); ys.append(d["d0.95"][0])
    if not xs: continue
    col = OI[k_i % len(OI)]; mk = ["D", "o", "v", "^", "<", ">", "s", "P", "X", "*", "h", "p"][k_i]
    ax.scatter(xs, ys, s=6, color=col, alpha=0.35, lw=0)
    ax.scatter([np.mean(xs)], [np.mean(ys)], s=45, color=col, edgecolor="k", lw=0.5, marker=mk, label=lab)
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel(r"$\Delta$ at $p_w=0.1$ (gain kept, points)"); ax.set_ylabel(r"$\Delta$ at $p_w=0.95$ (points)")
ax.set_title("Countermeasures, content-aware user (small: configurations, large: mean)")
ax.legend(frameon=False, fontsize=4.8, loc="center left", bbox_to_anchor=(1.0, 0.5))
save(fig, "g3_countermeasures.pdf")

# Fig 7. Effect size against the ceiling
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
lab, el, cap, lost = [], [], [], []
for s in SUBS:
    for r in RETS:
        e = T["effect"].get(f"{s}|{r}|content")
        if not e: continue
        lab.append(f"{s[:4]} {NAME[r]}"); el.append(e["ceil0.95"]["eligible"]); cap.append([e["ceil0.1"]["captured"], e["ceil0.95"]["captured"]])
        lost.append([e["ceil0.1"]["lost"], e["ceil0.95"]["lost"]])
x = np.arange(len(lab)); cap = np.array(cap); lost = np.array(lost)
axs[0].bar(x, el, color=OI[7]); axs[0].set_xticks(x); axs[0].set_xticklabels(lab, rotation=70, fontsize=5.5)
axs[0].set_ylabel("questions write-back could fix (%)"); axs[0].set_title("Ceiling: static wrong, earlier sibling right")
axs[1].bar(x - 0.2, cap[:, 0], 0.2, color=OI[2], label=r"fixed, $p_w=0.1$ (% of ceiling)"); axs[1].bar(x, cap[:, 1], 0.2, color=OI[0], label=r"fixed, $p_w=0.95$ (% of ceiling)")
axs[1].bar(x + 0.2, lost[:, 1], 0.2, color=OI[3], label=r"broken, $p_w=0.95$ (% of static-correct)")
axs[1].set_xticks(x); axs[1].set_xticklabels(lab, rotation=70, fontsize=5.5); axs[1].set_ylabel("%"); axs[1].set_title("What write-back fixes and breaks (content-aware user)")
axs[1].legend(frameon=False, fontsize=5, loc="upper left")
fig.tight_layout(); save(fig, "g3_ceiling.pdf")

# Fig 8. Generators and prompts (android, plus the open-prompt forums)
others = [("gpt-5.4-mini, evidence-only prompt", T, "-"), ("gemma-4-26b, evidence-only prompt", TL, "--"), ("gpt-5.4-mini, open prompt", TO, ":")]
subs8 = sorted(set(["android"] + (TO["meta"]["subs"] if TO else [])))
fig, axs = plt.subplots(len(subs8), len(RETS), figsize=(7.0, 1.9 * len(subs8)), sharex=True, squeeze=False)
for i, s in enumerate(subs8):
    for j, r in enumerate(RETS):
        ax = axs[i, j]
        for lab_, tab, ls in others:
            if tab is None: continue
            d = tab["h1"].get(f"{s}|{r}|askgen|content")
            if d: ax.plot(PWS, [d[f"d{p}"][0] for p in PWS], ls, color=OI[0], lw=1.0, label=lab_)
        ax.axhline(0, color="k", lw=0.5); ax.set_title(f"{s}, {NAME[r]}")
        if j == 0: ax.set_ylabel(r"$\Delta$ (points)")
        if i == len(subs8) - 1: ax.set_xlabel(r"$p_w$")
fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, "g3_generators.pdf")
print("figures written")

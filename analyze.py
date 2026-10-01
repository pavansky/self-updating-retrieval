"""Turn results/sim_*.json into tables (results/tables.json, tex/*.tex) and figures (figs/)."""
import json, os, sys, collections, itertools
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
FIG = os.path.join(HERE, "figs"); TEX = os.path.join(HERE, "tex")
NAME = {"bge": "bge-small", "minilm": "MiniLM-L6", "bm25": "BM25"}
PC = 0.9

def stream(curve, key="acc", lo=0):
    c = curve[lo:]; n = np.array([x["n"] for x in c])
    if key.startswith("rep_"):
        w = np.array([x["rep"] for x in c]); return float(np.sum(np.array([x[key] for x in c]) * w) / max(w.sum(), 1))
    return float(np.sum(np.array([x[key] for x in c]) * n) / n.sum())

def ci(d):
    d = np.asarray(d, float); m = d.mean()
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    return m, h

def crossover(pws, deltas):
    """first p at which mean delta goes from >=0 to <0 (linear interpolation)."""
    pws = np.asarray(pws); d = np.asarray(deltas)
    if d[0] < 0: return pws[0], "<"
    for k in range(1, len(d)):
        if d[k] < 0:
            return pws[k-1] + (pws[k] - pws[k-1]) * d[k-1] / (d[k-1] - d[k]), "="
    return pws[-1], ">"

def load(fn):
    J = json.load(open(os.path.join(RES, fn)))
    G = collections.defaultdict(dict)
    for r in J["runs"]:
        G[(r["sub"], r["ret"], r["cond"], r["p_w"])][r["seed"]] = r["curve"]
    return J["meta"], G

def analyze(fn, tag):
    meta, G = load(fn)
    subs = sorted({k[0] for k in G}); rets = [r for r in RETRIEVERS if any(k[1] == r for k in G)]
    seeds = sorted(G[(subs[0], rets[0], "static", None)])
    pws = sorted({k[3] for k in G if k[2] == "naive"}); psafe = sorted({k[3] for k in G if k[2] == "reopen"})
    T = dict(meta=meta, seeds=len(seeds), base={}, naive={}, safe={}, cross={}, selfshare={}, linear={})
    rng = np.random.default_rng(0)
    for s, r in itertools.product(subs, rets):
        st = G[(s, r, "static", None)]
        T["base"][f"{s}|{r}"] = {k: ci([stream(st[sd], k) for sd in seeds])[0] for k in ["acc", "hit5", "rep_acc", "rep_hit5"]}
        D = {}
        for cond in ("naive", "reopen", "reopen+quarantine"):
            for p in (pws if cond == "naive" else psafe):
                runs = G[(s, r, cond, p)]
                for key in ("acc", "hit5", "rep_acc", "rep_hit5"):
                    d = [100 * (stream(runs[sd], key) - stream(st[sd], key)) for sd in seeds]
                    D[(cond, p, key)] = d
                    (T["naive"] if cond == "naive" else T["safe"])[f"{s}|{r}|{cond}|{p}|{key}"] = ci(d)
                ss = [stream(runs[sd], "self_share", lo=len(runs[sd]) // 2) for sd in seeds]
                T["selfshare"][f"{s}|{r}|{cond}|{p}"] = ci([100 * x for x in ss])
        for key in ("acc", "hit5"):
            M = np.array([D[("naive", p, key)] for p in pws])          # pws x seeds
            x, kind = crossover(pws, M.mean(1))
            boot = [crossover(pws, M[:, rng.integers(0, len(seeds), len(seeds))].mean(1))[0] for _ in range(1000)]
            T["cross"][f"{s}|{r}|{key}"] = dict(p=x, kind=kind, lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)))
            sl = stats.linregress(pws, M.mean(1)); T["linear"][f"{s}|{r}|{key}"] = dict(r2=sl.rvalue ** 2, slope=sl.slope, icpt=sl.intercept)
    # analytic model  p* = kappa * p_c * A / (1 - A)
    fit = {}
    for key, akey in (("acc", "acc"), ("hit5", "hit5")):
        pts = [(T["base"][f"{s}|{r}"][akey], T["cross"][f"{s}|{r}|{key}"]) for s, r in itertools.product(subs, rets)]
        fin = [(A, c["p"]) for A, c in pts if c["kind"] == "="]
        if len(fin) >= 3:
            X = np.array([PC * A / (1 - A) for A, _ in fin]); Y = np.array([p for _, p in fin])
            kappa = float(np.sum(X * Y) / np.sum(X * X)); pred = kappa * X
            loo = [np.sum(np.delete(X, i) * np.delete(Y, i)) / np.sum(np.delete(X, i) ** 2) * X[i] for i in range(len(X))]
            fit[key] = dict(kappa=kappa, n=len(fin), r2=float(1 - np.sum((Y - pred) ** 2) / np.sum((Y - Y.mean()) ** 2)),
                            mae=float(np.mean(np.abs(Y - pred))), loo_mae=float(np.mean(np.abs(Y - np.array(loo)))),
                            spearman=float(stats.spearmanr(X, Y)[0]),
                            censored=[(f"{s}|{r}", T["cross"][f"{s}|{r}|{key}"]["kind"], PC * T["base"][f"{s}|{r}"][akey] / (1 - T["base"][f"{s}|{r}"][akey]) * kappa)
                                      for s, r in itertools.product(subs, rets) if T["cross"][f"{s}|{r}|{key}"]["kind"] != "="])
    # plug-in first-order model: p* = - p_c A e_c / ((1-A) e_w), per-entry effects e_c, e_w measured at one reference p_w
    plug = {}
    for s_, r in itertools.product(subs, rets):
        A = T["base"][f"{s_}|{r}"]["acc"]; row = {}
        for pref in pws:
            runs = G[(s_, r, "naive", pref)]
            ec = sum(x["eff_c"] for sd in seeds for x in runs[sd]) / max(sum(x["ent_c"] for sd in seeds for x in runs[sd][:-1]), 1)
            ew = sum(x["eff_w"] for sd in seeds for x in runs[sd]) / max(sum(x["ent_w"] for sd in seeds for x in runs[sd][:-1]), 1)
            row[pref] = dict(e_c=ec, e_w=ew, kappa=(-ec / ew) if ew < 0 else None,
                             pstar=(-PC * A * ec / ((1 - A) * ew)) if ew < 0 else None)
        plug[f"{s_}|{r}"] = row
    T["plug"] = plug
    ev = {}
    for pref in (0.3, 0.5, 0.95):
        obs, pred, agree = [], [], 0; tot = 0
        for k, row in plug.items():
            c = T["cross"][k + "|acc"]; pp = row[pref]["pstar"]
            if pp is None: continue
            tot += 1
            ppc = min(max(pp, pws[0]), pws[-1])
            kind = "<" if pp < pws[0] else (">" if pp > pws[-1] else "=")
            if c["kind"] == "=" and kind == "=": obs.append(c["p"]); pred.append(pp)
            agree += (c["kind"] == kind) and (kind != "=" or (c["lo"] - 0.05 <= pp <= c["hi"] + 0.05))
        obs, pred = np.array(obs), np.array(pred)
        ev[pref] = dict(n_both_interior=len(obs), mae=float(np.mean(np.abs(obs - pred))) if len(obs) else None,
                        spearman=float(stats.spearmanr(obs, pred)[0]) if len(obs) > 2 else None,
                        within_ci_or_same_censoring=int(agree), n=tot)
    T["plug_eval"] = ev
    T["fit"] = fit
    # monitors
    MON = ["ind_dis", "bank_drift", "selfdis", "self_share", "accept_rate"]
    rows = []
    for s, r in itertools.product(subs, rets):
        st = G[(s, r, "static", None)]
        for cond in ("naive", "reopen", "reopen+quarantine"):
            for p in (pws if cond == "naive" else psafe):
                for sd in seeds:
                    cu = G[(s, r, cond, p)][sd]; R = len(cu)
                    for rd in range(R):
                        rows.append(dict(sub=s, ret=r, cond=cond, p=p, seed=sd, rd=rd,
                                         harm=100 * (cu[rd]["acc"] - st[sd][rd]["acc"]),
                                         bharm=100 * (cu[rd]["bank_acc"] - st[sd][rd]["bank_acc"]),
                                         **{m: cu[rd][m] for m in MON},
                                         ind_rise=cu[rd]["ind_dis"] - st[sd][0]["ind_dis"]))
    mon = {}
    R = max(x["rd"] for x in rows) + 1
    for m in MON + ["ind_rise"]:
        rho_all, rho_fixed, auc_r = [], [], collections.defaultdict(list)
        for s, r in itertools.product(subs, rets):
            sel = [x for x in rows if x["sub"] == s and x["ret"] == r]
            last = [x for x in sel if x["rd"] >= R // 2]
            # run-level averages over the second half
            agg = collections.defaultdict(list)
            for x in last: agg[(x["cond"], x["p"], x["seed"])].append((x[m], x["bharm"], x["harm"]))
            mv = np.array([np.mean([a for a, _, _ in v]) for v in agg.values()]); hv = np.array([np.mean([c for _, _, c in v]) for v in agg.values()])
            if mv.std() > 0: rho_all.append(stats.spearmanr(mv, hv)[0])
            for p in psafe:   # within a fixed acceptance rate: does the monitor rank conditions/seeds by harm?
                ks = [k for k in agg if k[1] == p]
                a = np.array([np.mean([q[0] for q in agg[k]]) for k in ks]); h = np.array([np.mean([q[2] for q in agg[k]]) for k in ks])
                if a.std() > 0 and h.std() > 0: rho_fixed.append(stats.spearmanr(a, h)[0])
            # detection: is this run harmful by end of stream (mean harm over second half < 0)?
            lab = {k: np.mean([c for _, _, c in v]) < 0 for k, v in agg.items()}
            for rd in range(R):
                xs = [x for x in sel if x["rd"] == rd]
                y = np.array([lab[(x["cond"], x["p"], x["seed"])] for x in xs]); sc = np.array([x[m] for x in xs])
                if 0 < y.sum() < len(y):
                    from sklearn.metrics import roc_auc_score
                    auc_r[rd].append(roc_auc_score(y, sc))
        mon[m] = dict(spearman_runs=float(np.nanmean(rho_all)) if rho_all else None,
                      spearman_runs_sd=float(np.nanstd(rho_all)) if rho_all else None,
                      spearman_fixed_p=float(np.nanmean(rho_fixed)) if rho_fixed else None,
                      auroc_by_round=[float(np.mean(auc_r[rd])) if auc_r[rd] else None for rd in range(R)],
                      n_configs=len(rho_all))
    T["monitor"] = mon
    json.dump(T, open(os.path.join(RES, f"tables_{tag}.json"), "w"), indent=1)
    return T, G, subs, rets, pws, psafe, seeds

def figs(T, G, subs, rets, pws, tag):
    col = {"bge": "#2b6cb0", "minilm": "#2f855a", "bm25": "#c05621"}
    n = len(subs); nc = 4; nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(10, 2.2 * nr), sharex=True)
    for ax, s in zip(axs.flat, subs):
        for r in rets:
            m = np.array([T["naive"][f"{s}|{r}|naive|{p}|acc"][0] for p in pws]); h = np.array([T["naive"][f"{s}|{r}|naive|{p}|acc"][1] for p in pws])
            ax.plot(pws, m, "-o", ms=2.5, color=col[r], label=NAME[r]); ax.fill_between(pws, m - h, m + h, color=col[r], alpha=0.2)
        ax.axhline(0, color="k", lw=0.6); ax.set_title(s, fontsize=9); ax.tick_params(labelsize=7)
    for ax in axs.flat[n:]: ax.axis("off")
    axs.flat[0].legend(fontsize=7, frameon=False)
    fig.supxlabel("probability a wrong answer is accepted ($p_w$)", fontsize=9); fig.supylabel("change in top-1 accuracy (points)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"delta_{tag}.pdf")); plt.close(fig)
    # crossover vs model
    f = T["fit"].get("acc")
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for s, r in itertools.product(subs, rets):
        A = T["base"][f"{s}|{r}"]["acc"]; c = T["cross"][f"{s}|{r}|acc"]; x = PC * A / (1 - A)
        mk = {"=": "o", "<": "v", ">": "^"}[c["kind"]]
        ax.errorbar(x, c["p"], yerr=[[c["p"] - c["lo"]], [c["hi"] - c["p"]]] if c["kind"] == "=" else None, fmt=mk, color=col[r], ms=4, lw=0.7)
    if f:
        xx = np.linspace(0, ax.get_xlim()[1], 50); ax.plot(xx, f["kappa"] * xx, "k--", lw=0.8, label=f"$p^*={f['kappa']:.2f}\\,p_cA/(1-A)$")
        ax.legend(fontsize=7, frameon=False)
    for r in rets: ax.plot([], [], "o", color=col[r], label=NAME[r])
    ax.legend(fontsize=7, frameon=False); ax.set_xlabel("$p_c A/(1-A)$  (A = base top-1 accuracy)", fontsize=8); ax.set_ylabel("observed crossover $p^*$", fontsize=8)
    ax.set_ylim(0, 1); ax.tick_params(labelsize=7); fig.tight_layout(); fig.savefig(os.path.join(FIG, f"crossover_{tag}.pdf")); plt.close(fig)
    # monitor AUROC by round
    fig, ax = plt.subplots(figsize=(4.2, 3))
    lab = {"ind_dis": "independent-retriever disagreement", "bank_drift": "probe-bank drift", "selfdis": "pilot signal (self vs human)", "self_share": "self-written share", "ind_rise": "disagreement rise vs round 1", "accept_rate": "acceptance rate (baseline)"}
    for m, v in T["monitor"].items():
        y = v["auroc_by_round"]; ax.plot(range(1, len(y) + 1), y, "-o", ms=2.5, label=lab[m])
    ax.axhline(0.5, color="k", lw=0.5); ax.set_xlabel("round", fontsize=8); ax.set_ylabel("AUROC for harmful run", fontsize=8)
    ax.legend(fontsize=6, frameon=False); ax.tick_params(labelsize=7); fig.tight_layout(); fig.savefig(os.path.join(FIG, f"monitor_{tag}.pdf")); plt.close(fig)

if __name__ == "__main__":
    for fn in sys.argv[1:]:
        tag = fn.replace("sim_", "").replace(".json", "")
        T, G, subs, rets, pws, psafe, seeds = analyze(fn, tag); figs(T, G, subs, rets, pws, tag)
        print(tag, json.dumps(T["fit"], indent=0)[:600]); print({m: (v["spearman_runs"], v["spearman_fixed_p"], [round(a, 2) if a else a for a in v["auroc_by_round"]]) for m, v in T["monitor"].items()})

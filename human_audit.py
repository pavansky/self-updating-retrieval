"""Human audit of the content judge (TMLR revision). make: sample 200 answers (50 per forum; depth 0, 1, 2+ balanced) and
write tmlr/human_audit/sheet.csv for a person to label; score: read the filled sheet and report agreement with the three
judges and Cohen's kappa.   GEN_TAG=_az python human_audit.py make | score"""
import csv, json, os, sys, random, collections
import numpy as np
from common import load
from prep_gen import G, TAG, load_gen, kb_ids, parse_ev
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmlr", "human_audit")
SUBS = ["android", "unix", "webmasters", "mathematica"]
def make(n_per=50, seed=7):
    rng = random.Random(seed); rows = []
    for sub in SUBS:
        kb, kfam, ask, afam = load(sub); ids = kb_ids(sub); ans = json.load(open(os.path.join(G, f"{sub}_answers.json"))); g = load_gen(sub)
        J = {json.loads(l)["key"]: json.loads(l)["label"] for l in open(os.path.join(G, f"{sub}_judge{TAG}.jsonl"))}
        by = collections.defaultdict(list)
        for k in g:
            root, depth, _ = parse_ev(k.split("|", 1)[1])
            if ans.get(ids[root]) and J.get(k) in ("YES", "PARTLY", "NO", "ABSTAIN"): by[min(depth, 2)].append((k, root))
        per = [n_per // 3 + (1 if i < n_per % 3 else 0) for i in range(3)]
        for d, m in zip((0, 1, 2), per):
            for k, root in rng.sample(by[d], m):
                rows.append(dict(id=len(rows) + 1, forum=sub, key=k, question=ask[int(k.split("|")[0])], reference=" ".join(ans[ids[root]]["body"].split()[:400]),
                                 system_answer=g[k], your_label=""))
    rng.shuffle(rows)
    with open(os.path.join(OUT, "sheet.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "forum", "question", "reference", "system_answer", "your_label", "key"]); w.writeheader(); w.writerows(rows)
    print("wrote", len(rows), "rows")
def score(target=None):
    if target is None and len(sys.argv) > 2:
        target = sys.argv[2]
    J = {}
    for sub in SUBS:
        for l in open(os.path.join(G, f"{sub}_judge{TAG}.jsonl")): d = json.loads(l); J[d["key"]] = d["label"]
    labs = ("YES", "PARTLY", "NO", "ABSTAIN")
    def calc_metrics(h, m):
        po = np.mean([a == b for a, b in zip(h, m)])
        pe = sum((h.count(l) / len(h)) * (m.count(l) / len(m)) for l in labs)
        kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
        yes_acc = np.mean([(a == "YES") == (b == "YES") for a, b in zip(h, m)])
        return po, kappa, yes_acc

    users = ["liya", "sow", "sky"]
    user_rows = {}
    for u in users:
        p = os.path.join(OUT, f"human_labels_{u}.csv")
        if os.path.exists(p):
            user_rows[u] = [r for r in csv.DictReader(open(p)) if r["your_label"].strip() and r.get("source") == "label.html"]

    src = next((p for p in (os.path.join(OUT, f"human_labels_{target}.csv" if target and target in users else "human_labels.csv"),
                            os.path.expanduser("~/Downloads/human_labels.csv")) if os.path.exists(p)), None)
    assert src, "export human_labels.csv from label.html first"
    rows = [r for r in csv.DictReader(open(src)) if r["your_label"].strip() and r.get("source") == "label.html"]
    h = [r["your_label"].strip().upper() for r in rows]; m = [J[r["key"]] for r in rows]
    po, kappa, yes_acc = calc_metrics(h, m)

    print(f"=== Human Audit Score (default: {os.path.basename(src)}) ===")
    print(f"n={len(rows)} agreement={po:.3f} kappa={kappa:.3f} YES_vs_not={yes_acc:.3f}")

    summary = dict(n=len(rows), agreement=po, kappa=kappa, yes_vs_not=yes_acc)
    if user_rows:
        print("\n=== Annotator Breakdown vs AI Judge ===")
        print(f"{'Annotator':<10} {'Agreement':<12} {'Cohen Kappa':<14} {'YES vs Not':<12}")
        print("-" * 50)
        summary["annotators"] = {}
        for u, urows in user_rows.items():
            uh = [r["your_label"].strip().upper() for r in urows]
            um = [J[r["key"]] for r in urows]
            upo, ukappa, uyes = calc_metrics(uh, um)
            summary["annotators"][u] = dict(n=len(urows), agreement=upo, kappa=ukappa, yes_vs_not=uyes)
            print(f"{u:<10} {upo:<12.3f} {ukappa:<14.3f} {uyes:<12.3f}")

        if len(user_rows) >= 2:
            print("\n=== Pairwise Inter-Annotator Agreement ===")
            print(f"{'Pair':<15} {'Agreement':<12} {'Cohen Kappa':<14}")
            print("-" * 45)
            summary["inter_annotator"] = {}
            pairs = [("liya", "sow"), ("sow", "sky"), ("liya", "sky")]
            for u1, u2 in pairs:
                if u1 in user_rows and u2 in user_rows:
                    h1 = [r["your_label"].strip().upper() for r in user_rows[u1]]
                    h2 = [r["your_label"].strip().upper() for r in user_rows[u2]]
                    ppo, pkappa, _ = calc_metrics(h1, h2)
                    pair_name = f"{u1} vs {u2}"
                    summary["inter_annotator"][pair_name] = dict(agreement=ppo, kappa=pkappa)
                    print(f"{pair_name:<15} {ppo:<12.3f} {pkappa:<14.3f}")

    json.dump(summary, open(os.path.join(OUT, "score.json"), "w"), indent=2)
    print(f"\nSaved scores to {os.path.join(OUT, 'score.json')}")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    {"make": make, "score": score}[cmd]()

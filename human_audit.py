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
def score():
    rows = [r for r in csv.DictReader(open(os.path.join(OUT, "sheet.csv"))) if r["your_label"].strip()]
    J = {}
    for sub in SUBS:
        for l in open(os.path.join(G, f"{sub}_judge{TAG}.jsonl")): d = json.loads(l); J[d["key"]] = d["label"]
    h = [r["your_label"].strip().upper() for r in rows]; m = [J[r["key"]] for r in rows]
    labs = ("YES", "PARTLY", "NO", "ABSTAIN"); po = np.mean([a == b for a, b in zip(h, m)])
    pe = sum((h.count(l) / len(h)) * (m.count(l) / len(m)) for l in labs)
    print(f"n={len(rows)} agreement={po:.3f} kappa={(po - pe) / (1 - pe):.3f}")
    print("YES vs not agreement", np.mean([(a == "YES") == (b == "YES") for a, b in zip(h, m)]))
    json.dump(dict(n=len(rows), agreement=po, kappa=(po - pe) / (1 - pe)), open(os.path.join(OUT, "score.json"), "w"))
if __name__ == "__main__": {"make": make, "score": score}[sys.argv[1]]()

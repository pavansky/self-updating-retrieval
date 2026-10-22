"""Paper-2 faithfulness audit (Appendix app:audit). Samples generated answers from the released cache, stratified by evidence
depth (0 = base thread, 1 = entry generated from a thread, 2+ = entry generated from an entry), and asks two judges whether
the served answer states the solution of the root thread's accepted answer. Judges: gpt-5.4-mini (Azure) and the local
open-weight model; the generator is gpt-5.4-mini, so the local judge is the independent one.
Usage: GEN_TAG=_az python audit_gen.py android,unix,webmasters,mathematica [n_per_depth=70] [seed=0]
       GEN_TAG=_az python audit_gen.py tables      (rewrite the table and macros from the saved judgements)
Writes results/audit_gen{TAG}.json and tex/tab_audit_gen.tex."""
import json, os, sys, random, collections, urllib.request, time
import concurrent.futures as cf
from prep_gen import G, TAG, AZ_EP, AZ_KEY_FILE, GEN_URL, parse_ev, load_gen, kb_ids
from common import load, HERE
RES = os.path.join(HERE, "results"); TEX = os.path.join(HERE, "tex")

JUDGE = ("You are grading a technical support system. You are given a question, a reference answer written by the community "
         "and accepted as the solution, and the system's answer. Decide whether the system's answer gives the same solution as "
         "the reference. Reply with exactly one word: YES (same solution, possibly less detailed), PARTLY (overlaps but misses "
         "or changes the key step), or NO (different, wrong or no solution).")

def judge_az(q, ref, ans):
    url = f"{AZ_EP}/openai/deployments/gpt-5.4-mini/chat/completions?api-version=2024-12-01-preview"
    body = {"messages": [{"role": "system", "content": JUDGE}, {"role": "user", "content": f"Question: {q}\n\nReference answer:\n{ref}\n\nSystem answer:\n{ans}"}],
            "max_completion_tokens": 20, "reasoning_effort": "minimal", "seed": 0}
    hdr = {"Content-Type": "application/json", "api-key": open(AZ_KEY_FILE).read().strip()}
    for _ in range(20):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), hdr), timeout=120))
            return d["choices"][0]["message"]["content"].strip().upper().split()[0].strip(".,")
        except urllib.error.HTTPError as e:
            if e.code == 429: time.sleep(min(float(e.headers.get("Retry-After", 0) or 5), 30)); continue
            if e.code == 400: return "FILTERED"
            time.sleep(3)
        except Exception: time.sleep(3)
    return "ERR"

def judge_local(q, ref, ans):
    body = {"model": "google/gemma-4-26b-a4b-qat", "temperature": 0, "max_tokens": 10, "reasoning_effort": "none", "seed": 0,
            "messages": [{"role": "system", "content": JUDGE}, {"role": "user", "content": f"Question: {q}\n\nReference answer:\n{ref}\n\nSystem answer:\n{ans}"}]}
    for _ in range(8):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(GEN_URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
            return d["choices"][0]["message"]["content"].strip().upper().split()[0].strip(".,")
        except Exception: time.sleep(5)
    return "ERR"

def main(subs, n_per=70, seed=0):
    rng = random.Random(seed); rows = []
    for sub in subs:
        kb, kfam, ask, afam = load(sub); ids = kb_ids(sub)
        ans = json.load(open(os.path.join(G, f"{sub}_answers.json"))); g = load_gen(sub)
        by_d = collections.defaultdict(list)
        for k in g:
            j, ev = k.split("|", 1); root, depth, _ = parse_ev(ev)
            if ans.get(ids[root]): by_d[min(depth, 2)].append((k, root, depth))
        for d in (0, 1, 2):
            for k, root, depth in rng.sample(by_d[d], min(n_per, len(by_d[d]))):
                j = int(k.split("|", 1)[0])
                rows.append(dict(sub=sub, key=k, depth=depth, correct=bool(kfam[root] == afam[j] and kfam[root] >= 0),
                                 q=ask[j], ref=ans[ids[root]]["body"], ans=g[k]))
    print(len(rows), "answers to judge", flush=True)
    with cf.ThreadPoolExecutor(6) as ex:
        for r, v in zip(rows, ex.map(lambda r: judge_az(r["q"], r["ref"], r["ans"]), rows)): r["judge_az"] = v
    print("azure judge done", flush=True)
    with cf.ThreadPoolExecutor(4) as ex:
        for r, v in zip(rows, ex.map(lambda r: judge_local(r["q"], r["ref"], r["ans"]), rows)): r["judge_local"] = v
    print("local judge done", flush=True)
    json.dump(rows, open(os.path.join(RES, f"audit_gen{TAG}.json"), "w"), indent=1)
    tables(rows)

def tables(rows):
    # table: by depth x evidence-label, share YES / PARTLY / NO per judge
    lines = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"evidence depth & label & $n$ & \multicolumn{3}{c}{gpt-5.4-mini judge} & \multicolumn{3}{c}{open-weight judge} \\",
             r" & & & yes & partly & no & yes & partly & no \\", r"\midrule"]
    agree = sum(r["judge_az"] == r["judge_local"] for r in rows)
    for d, dn in ((0, "0 (thread)"), (1, "1"), (2, "2+")):
        for lab, ln in ((True, "correct"), (False, "wrong")):
            S = [r for r in rows if min(r["depth"], 2) == d and r["correct"] == lab]
            if not S: continue
            f = lambda j, v: f"{100*sum(r[j] == v for r in S)/len(S):.0f}\\%"
            lines.append(f"{dn} & {ln} & {len(S)} & {f('judge_az','YES')} & {f('judge_az','PARTLY')} & {f('judge_az','NO')} & "
                         f"{f('judge_local','YES')} & {f('judge_local','PARTLY')} & {f('judge_local','NO')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TEX, "tab_audit_gen.tex"), "w").write("\n".join(lines) + "\n")
    mac = [f"\\newcommand{{\\nAudit}}{{{len(rows)}}}", f"\\newcommand{{\\auditAgree}}{{{100*agree/len(rows):.0f}}}"]
    for d, dn in ((0, "Zero"), (1, "One"), (2, "Two")):
        S = [r for r in rows if min(r["depth"], 2) == d]
        for j, jn in (("judge_az", "Az"), ("judge_local", "Loc")):
            mac.append(f"\\newcommand{{\\auditNo{jn}{dn}}}{{{100*sum(r[j] == 'NO' for r in S)/len(S):.0f}}}")
    open(os.path.join(TEX, "macros_audit.tex"), "w").write("\n".join(mac) + "\n")
    print("agreement", agree / len(rows), flush=True)

if __name__ == "__main__":
    if sys.argv[1] == "tables": tables(json.load(open(os.path.join(RES, f"audit_gen{TAG}.json")))); sys.exit()
    main(sys.argv[1].split(","), int(sys.argv[2]) if len(sys.argv) > 2 else 70, int(sys.argv[3]) if len(sys.argv) > 3 else 0)

"""Content labels for every generated answer (paper 2, content-aware correctness).
An independent judge (DeepSeek-V4.1-Flash on Azure AI Foundry; the generator is gpt-5.4-mini) reads the question, the
reference answer of the ROOT thread of the answer's evidence chain (accepted, else top-voted human answer) and the
generated answer, and returns one label:
    YES     the answer gives the same solution as the reference (possibly less detailed)
    PARTLY  it overlaps but misses or changes the key step
    NO      a different, wrong or no solution
    ABSTAIN it says the information it has does not answer the question and offers no solution
    NOREF   (no call) the root thread has no human answer on record, so content cannot be checked
Labels are cached in cache/gen/{sub}_judge{TAG}.jsonl, one per key, and only missing keys are judged.
  GEN_TAG=_az JUDGE_WORKERS=4 python judge_gen.py android,unix,webmasters,mathematica
"""
import os, sys, json, time, random, urllib.request, urllib.error
import concurrent.futures as cf
from common import load
from prep_gen import G, TAG, AZ_EP, AZ_KEY_FILE, load_gen, kb_ids, parse_ev

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "DeepSeek-V4.1-Flash")
LABELS = ("YES", "PARTLY", "NO", "ABSTAIN")
SYSTEM = ("You grade answers from a technical support system. You are given the user's question, a reference answer written "
          "by the community and accepted as the solution, and the system's answer. Reply with exactly one word.\n"
          "YES: the system's answer gives the same solution as the reference, possibly with less detail.\n"
          "PARTLY: it overlaps with the reference but misses or changes the key step.\n"
          "NO: it gives a different solution, a wrong solution, or no usable solution.\n"
          "ABSTAIN: it says the information available does not answer the question and does not offer a solution.")

def ref_text(root, ids, ans, maxw=600):
    a = ans.get(ids[root]); return " ".join(a["body"].split()[:maxw]) if a else None

def call(q, ref, a):
    url = f"{AZ_EP}/models/chat/completions?api-version=2024-05-01-preview"
    body = {"model": JUDGE_MODEL, "temperature": 0, "max_tokens": 6, "seed": 0,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"Question: {q}\n\nReference answer:\n{ref}\n\nSystem answer:\n{a}\n\nLabel:"}]}
    hdr = {"Content-Type": "application/json", "api-key": open(AZ_KEY_FILE).read().strip()}
    tries = 0
    while True:
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), hdr), timeout=120))
            t = (d["choices"][0]["message"]["content"] or "").strip().upper()
            for l in LABELS:
                if t.startswith(l): return l
            return "UNPARSED:" + t[:20]
        except urllib.error.HTTPError as e:
            if e.code == 429: time.sleep(min(float(e.headers.get("Retry-After", 0) or 5), 30) + random.random()); continue
            if e.code == 400: return "FILTERED"
            tries += 1; time.sleep(3 * tries)
        except Exception:
            tries += 1; time.sleep(3 * tries)
        if tries >= 8: return "ERR"

def load_judge(sub):
    p = os.path.join(G, f"{sub}_judge{TAG}.jsonl"); j = {}
    if os.path.exists(p):
        for l in open(p):
            d = json.loads(l)
            if d["label"] not in ("ERR",): j[d["key"]] = d["label"]
    return j

def main(subs, workers=int(os.environ.get("JUDGE_WORKERS", 4))):
    for sub in subs:
        kb, kfam, ask, afam = load(sub); ids = kb_ids(sub)
        ans = json.load(open(os.path.join(G, f"{sub}_answers.json"))); g = load_gen(sub); have = load_judge(sub)
        need = [k for k in g if k not in have]
        print(f"{sub}: {len(need)} to judge, {len(have)} cached", flush=True)
        out = open(os.path.join(G, f"{sub}_judge{TAG}.jsonl"), "a"); t0 = time.time(); n = 0
        def work(k):
            j, ev = k.split("|", 1); root = parse_ev(ev)[0]; ref = ref_text(root, ids, ans)
            return k, ("NOREF" if ref is None else call(ask[int(j)], ref, g[k]))
        with cf.ThreadPoolExecutor(workers) as ex:
            for f in cf.as_completed([ex.submit(work, k) for k in need]):
                k, lab = f.result(); out.write(json.dumps(dict(key=k, label=lab, judge=JUDGE_MODEL)) + "\n"); out.flush(); n += 1
                if n % 500 == 0: print(f"  {sub} {n}/{len(need)} {time.time()-t0:.0f}s", flush=True)
        print(f"{sub}: judged {n} in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main(sys.argv[1].split(","))

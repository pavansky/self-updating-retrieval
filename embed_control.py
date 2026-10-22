"""Source-bias control for paper 2 (H1). For every generated-answer key, build the entry text
    askhum : src question + the root thread's HUMAN answer (accepted, else top-voted; the thread's own text when the
             thread has no answer on record), truncated to the word count of the generated answer for the same key,
so that askhum and askgen differ only in who wrote the answer part (same question, same root, same length);
    hum    : the same human text without the question (the control for the answer-only design, gen).
Writes cache/gen/{sub}_entries{TAG}_ctrl.npz with keys (same order as {sub}_entries{TAG}.npz), E_askhum_{bge,minilm}
(float16) and BM_askhum (asks x keys, float16). sim_gen.py loads it when present.
  GEN_TAG=_az python embed_control.py android,unix,webmasters,mathematica
"""
import os, sys, json, time
import numpy as np
from common import load
from prep_gen import G, TAG, DENSE, load_gen, kb_ids, parse_ev, st, bm25_matrix

CTRL = ["askhum", "hum"]

def human_text(k, kb, ask, ids, ans, g, with_ask=True):
    j, ev = k.split("|", 1); root = parse_ev(ev)[0]
    a = ans.get(ids[root]); body = a["body"] if a else kb[root]
    n = len(g[k].split()); h = " ".join(body.split()[:n])
    return ask[int(j)] + "\n" + h if with_ask else h

def main(subs):
    for sub in subs:
        t0 = time.time()
        kb, kfam, ask, afam = load(sub); ids = kb_ids(sub); g = load_gen(sub)
        ans = json.load(open(os.path.join(G, f"{sub}_answers.json")))
        z = np.load(os.path.join(G, f"{sub}_entries{TAG}.npz")); keys = z["keys"].tolist()
        assert all(k in g for k in keys)
        save = dict(keys=np.array(keys))
        for v, wa in (("askhum", True), ("hum", False)):
            txt = [human_text(k, kb, ask, ids, ans, g, wa) for k in keys]
            for r in DENSE:
                save[f"E_{v}_{r}"] = st(r).encode(txt, batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype(np.float16)
            save[f"BM_{v}"] = bm25_matrix(kb, ask, txt)
        np.savez(os.path.join(G, f"{sub}_entries{TAG}_ctrl.npz"), **save)
        print(f"control {sub}: {len(keys)} keys {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main(sys.argv[1].split(","))

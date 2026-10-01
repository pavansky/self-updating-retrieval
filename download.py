"""Download the 12 CQADupStack subforums (BEIR format) from the public MTEB mirror on Hugging Face."""
import os, shutil
from huggingface_hub import hf_hub_download
FORUMS = ["android","english","gaming","gis","mathematica","physics","programmers","stats","tex","unix","webmasters","wordpress"]
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
for f in FORUMS:
    for fn in ["corpus.jsonl", "queries.jsonl", "qrels/test.tsv"]:
        dst = os.path.join(D, f, fn)
        if os.path.exists(dst): continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(hf_hub_download(f"mteb/cqadupstack-{f}", fn, repo_type="dataset"), dst)
    print(f, "ok", flush=True)

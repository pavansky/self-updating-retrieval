#!/bin/sh
# Iterate cross-encoder / ask+thread caches until sim_v3 needs no missing pair (as for v2).
F=android,english,gaming,gis,mathematica,physics,programmers,stats,tex,unix,webmasters,wordpress
for it in 1 2 3 4 5 6; do
  .venv/bin/python prep_v3.py cepairs $F
  .venv/bin/python prep_v2.py askdoc bge,minilm,bm25
  .venv/bin/python sim_v3.py 10 10 $F bge,minilm,bm25 main | tail -1 | tee -a results/sim_v3_main_iter.log
  tail -1 results/sim_v3_main_iter.log | grep -q "missing CE pairs: 0 missing askdoc pairs: 0" && break
done

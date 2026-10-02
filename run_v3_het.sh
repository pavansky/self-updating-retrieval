#!/bin/sh
# Heterogeneous corpus (MS MARCO families): index, then iterate ask+thread / cross-encoder caches until no pair is missing.
.venv/bin/python prep_het_v3.py build
.venv/bin/python prep_het_v3.py index bge,minilm,bm25
for it in 1 2 3 4 5 6; do
  .venv/bin/python prep_het_v3.py askdoc bge,minilm,bm25
  .venv/bin/python sim_v2.py 10 10 msmarcoh bge,minilm,bm25 het | tail -1 | tee -a results/sim_het_iter.log
  tail -1 results/sim_het_iter.log | grep -q "missing askdoc pairs: 0$" && break
done
for it in 1 2 3 4 5 6; do
  .venv/bin/python prep_het_v3.py cepairs bge,minilm,bm25
  .venv/bin/python prep_het_v3.py askdoc bge,minilm,bm25
  .venv/bin/python sim_v3.py 10 10 msmarcoh bge,minilm,bm25 het | tail -1 | tee -a results/sim_het_iter.log
  tail -1 results/sim_het_iter.log | grep -q "missing CE pairs: 0 missing askdoc pairs: 0" && break
done

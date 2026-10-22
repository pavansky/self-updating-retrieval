#!/bin/sh
# Paper-2 driver for one or more forums: generate -> embed -> simulate, iterated until the simulation needs no further
# generated answers (deeper chains appear only after shallower ones exist). Usage: ./run_gen.sh android [seeds] [maxiter]
# Waits for any running prep_gen.py gen first. Logs to results/run_gen_<sub>.log.
set -e
cd "$(dirname "$0")"
SUB=$1; SEEDS=${2:-10}; MAXIT=${3:-6}
PY=.venv/bin/python
# (no wait: the Azure loop may run alongside)
for it in $(seq 1 $MAXIT); do
  echo "== $SUB iteration $it $(date)"
  GEN_WORKERS=4 $PY prep_gen.py gen $SUB local
  $PY prep_gen.py embed $SUB
  NPROC=10 $PY sim_gen.py 10 $SEEDS $SUB bge,minilm,bm25 $SUB | tail -2
  miss=$($PY -c "import json;print(json.load(open('results/sim_gen_$SUB.json'))['missing_keys'])")
  echo "missing keys after iteration $it: $miss"
  [ "$miss" = "0" ] && break
done
echo "== $SUB done $(date)"

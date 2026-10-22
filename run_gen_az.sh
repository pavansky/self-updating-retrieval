#!/bin/sh
# Azure generator loop over several forums at once: generate (all forums concurrently) -> embed -> simulate -> repeat
# until no forum needs further generated answers. Usage: GEN_BACKEND=azure GEN_TAG=_az ./run_gen_az.sh "android unix webmasters" [seeds] [maxiter]
set -e
cd "$(dirname "$0")"
SUBS=$1; SEEDS=${2:-10}; MAXIT=${3:-6}; PY=.venv/bin/python
export GEN_BACKEND=${GEN_BACKEND:-azure} GEN_TAG=${GEN_TAG:-_az} GEN_WORKERS=${GEN_WORKERS:-16}
for it in $(seq 1 $MAXIT); do
  echo "== iteration $it $(date)"
  for s in $SUBS; do $PY prep_gen.py gen $s > results/gen${GEN_TAG}_$s.log 2>&1 & done; wait
  for s in $SUBS; do tail -1 results/gen${GEN_TAG}_$s.log; done
  for s in $SUBS; do $PY prep_gen.py embed $s; done
  total=0
  for s in $SUBS; do
    NPROC=10 $PY sim_gen.py 10 $SEEDS $s bge,minilm,bm25 az_$s | tail -1
    miss=$($PY -c "import json;print(json.load(open('results/sim_gen_az_$s.json'))['missing_keys'])"); total=$((total + miss))
  done
  echo "missing keys after iteration $it: $total"
  [ "$total" = "0" ] && break
done
echo "== done $(date)"

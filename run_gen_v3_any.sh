#!/bin/sh
# Paper-2 v3 loop: generate -> judge -> embed (+ source-bias controls) -> simulate, until no served key is missing.
# Usage: GEN_TAG=_az [GEN_PROMPT=open] [PLAN=min] ./run_gen_v3.sh "android unix webmasters mathematica" RESTAG [maxiter]
# Writes results/sim_gen_<RESTAG><sub>.json and logs to results/v3_<RESTAG>loop.log.
cd "$(dirname "$0")"
SUBS=$1; RT=$2; MAXIT=${3:-8}; PY=.venv/bin/python
for it in $(seq 1 $MAXIT); do
  echo "== $RT iteration $it $(date)"
  for s in $SUBS; do GEN_BACKEND=${GEN_BACKEND:-azure} GEN_WORKERS=${GEN_WORKERS:-6} $PY prep_gen.py gen $s > results/v3_${RT}gen_$s.log 2>&1 & done; wait
  for s in $SUBS; do tail -n1 results/v3_${RT}gen_$s.log; done
  for s in $SUBS; do JUDGE_WORKERS=${JUDGE_WORKERS:-12} $PY judge_gen.py $s > results/v3_${RT}judge_$s.log 2>&1 & done; wait
  for s in $SUBS; do tail -n1 results/v3_${RT}judge_$s.log; done
  for s in $SUBS; do $PY prep_gen.py embed $s 2>&1 | grep -E "^embed|Error|Traceback"; done
  if [ "${CTRL:-1}" = "1" ]; then for s in $SUBS; do $PY embed_control.py $s 2>&1 | grep -E "^control|Error|Traceback"; done; fi
  miss=0
  for s in $SUBS; do
    NPROC=${NPROC:-12} $PY sim_gen.py ${ROUNDS:-10} 10 $s bge,minilm,bm25 $RT$s | tail -1
    m=$($PY -c "import json;print(json.load(open('results/sim_gen_$RT$s.json'))['missing_keys'])" 2>/dev/null || echo 999999); miss=$((miss + m))
  done
  echo "missing keys after iteration $it: $miss"
  [ "$miss" = "0" ] && break
done
echo "== $RT done $(date)"

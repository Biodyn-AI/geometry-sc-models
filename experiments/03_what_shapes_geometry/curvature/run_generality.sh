#!/bin/bash
# Route Q generality: 4 models x 3 human lineages + mouse pancreas (pre-declared NEGATIVE control).
# Pancreas runs in FULL mode (abundance guard + rank sweep) because it is a control, not a filler cell.
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
for ds in setty lung gut; do
  for m in scgpt geneformer state maxtoki; do
    [ -f "results/q_${m}_${ds}.json" ] && { echo "[have] $m/$ds"; continue; }
    echo "=== $m / $ds (quick) ==="
    OMP_NUM_THREADS=4 nice -n 5 $PY run_substrate.py $m $ds --quick
  done
done
for m in scgpt geneformer state maxtoki; do
  echo "=== $m / pancreas (FULL, negative control) ==="
  OMP_NUM_THREADS=4 nice -n 5 $PY run_substrate.py $m pancreas
done
echo "ALL GENERALITY DONE"

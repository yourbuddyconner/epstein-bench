#!/bin/bash
source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate epstein-bench
set -e
cd "$(dirname "$0")/.."
for sys in bm25 dense hybrid closed_book parametric; do
  echo "=== BASELINE $sys $(date) ==="
  python baselines/run_baseline.py --system $sys --split full --out build/preds_v11_$sys.jsonl
  slug=$(echo "$sys-reference" | tr '_' '-')
  python -m epstein_bench submit build/preds_v11_$sys.jsonl --name "$sys (reference)" --split full --out submissions
  echo "=== SCORE $sys $(date) ==="
  python -m epstein_bench validate "submissions/$slug"
done
echo "=== BASELINES DONE $(date) ==="

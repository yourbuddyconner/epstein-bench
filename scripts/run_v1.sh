#!/bin/bash
# Full v1.0 generation run. Resumable: every stage's LLM calls are disk-cached.
set -e
source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate epstein-bench
cd "$(dirname "$0")/.."
echo "=== CORPUS $(date) ==="
python -m epstein_bench corpus --limit 20000
echo "=== GENERATE $(date) ==="
python -m epstein_bench generate --target 1000
echo "=== VERIFY $(date) ==="
python -m epstein_bench verify
echo "=== POOL $(date) ==="
python -m epstein_bench pool
echo "=== FINALIZE $(date) ==="
python -m epstein_bench finalize --target 1000
echo "=== DONE $(date) ==="

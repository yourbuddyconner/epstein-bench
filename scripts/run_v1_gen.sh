#!/bin/bash
# Resume v1.1 from generate (scan + select already cached in build/).
source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate epstein-bench
set -e
cd "$(dirname "$0")/.."
echo "=== GENERATE $(date) ==="
python -m epstein_bench generate --target 1000
echo "=== VERIFY $(date) ==="
python -m epstein_bench verify
echo "=== POOL $(date) ==="
python -m epstein_bench pool
echo "=== FINALIZE $(date) ==="
python -m epstein_bench finalize --target 1000
echo "=== DONE $(date) ==="

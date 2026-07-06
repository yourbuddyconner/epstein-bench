#!/bin/bash
# Full generation run (v1.1 flow). Resumable: scan parts are per-shard,
# LLM calls are disk-cached.
# NB: set -e only after conda setup — conda.sh trips set -e internally.
source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate epstein-bench
set -e
cd "$(dirname "$0")/.."
echo "=== SCAN $(date) ==="
python -m epstein_bench scan ${SCAN_SHARDS:+--shards $SCAN_SHARDS}
echo "=== SELECT $(date) ==="
python -m epstein_bench select
echo "=== GENERATE $(date) ==="
python -m epstein_bench generate --target 1000
echo "=== VERIFY $(date) ==="
python -m epstein_bench verify
echo "=== POOL $(date) ==="
python -m epstein_bench pool
echo "=== FINALIZE $(date) ==="
python -m epstein_bench finalize --target 1000
echo "=== DONE $(date) ==="

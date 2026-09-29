#!/usr/bin/env bash
# Run from the existing BatLiNet-context-grid-v1 checkout. No test evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
DATA_ROOT=${1:-/root/autodl-tmp/BatLiNet-main2/data/processed}
CONTEXT=artifacts/context_grid_v1/mix20.pt
DATA=artifacts/mix20_matched_baselines_v1/mix20.pt
PROTOCOLS=protocols/mix20_matched_v1
test -f "$CONTEXT" || { echo "Missing existing context cache: $CONTEXT" >&2; exit 1; }
python -B scripts/test_matched_baselines.py
python -B scripts/matched_baselines.py prepare --context-data "$CONTEXT" \
  --data-root "$DATA_ROOT" --output "$DATA"
python -B scripts/matched_baselines.py protocols --context-data "$CONTEXT" \
  --output-dir "$PROTOCOLS"
for seed in 0 1 2 3 4 5 6 7; do
  for model in batlinet latent_cross_attention; do
    python -B scripts/matched_baselines.py train \
      --context-data "$CONTEXT" --data "$DATA" --protocol-dir "$PROTOCOLS" \
      --workspace "workspaces/mix20_matched_baselines_v1/${model}_seed${seed}" \
      --model "$model" --seed "$seed" --epochs 1000 \
      --batch-size 8 --accumulation 16 --evaluate-every 25 \
      --device cuda:0 --amp --skip-complete
  done
done
echo "Both baselines completed seeds 0-7. Test set was NOT evaluated."

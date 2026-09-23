#!/usr/bin/env bash
# First GPU timing/validation run; never touches the test set.
set -euo pipefail
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
DATA_ROOT=${1:-/root/autodl-tmp/BatLiNet-main2/data/processed}
RUN=${2:-workspaces/context_grid_v1/mix20_pilot_seed0}
DATA=artifacts/context_grid_v1/mix20.pt
if [[ -e "$RUN" ]]; then
  echo "运行目录已存在，请使用新的目录名称：$RUN" >&2
  exit 1
fi
python -B scripts/test_context_grid_lifetime.py
if [[ ! -f "$DATA" ]]; then
  python scripts/context_grid_lifetime.py prepare --dataset mix20 \
    --cycles 20 --data-root "$DATA_ROOT" --output "$DATA"
fi
python scripts/context_grid_lifetime.py train --data "$DATA" \
  --workspace "$RUN" --device cuda:0 --seed 0 --epochs 20 \
  --batch-size 8 --accumulation 16 --evaluate-every 5 --amp

#!/usr/bin/env bash
# Reuse the completed original-encoder runs and the exact existing MIX-20 data.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
EXISTING_ROOT=${1:-/root/autodl-tmp/BatLiNet-context-grid-v1}
MODE=${2:-pilot}
case "$MODE" in
  pilot) SEEDS=(0 1 2) ;;
  full) SEEDS=(0 1 2 3 4 5 6 7) ;;
  *) echo "用法：bash scripts/run_mix20_cycle_mixer.sh 已有实验目录 pilot或full" >&2; exit 1 ;;
esac
CONTEXT="$EXISTING_ROOT/artifacts/context_grid_v1/mix20.pt"
DATA="$EXISTING_ROOT/artifacts/mix20_matched_baselines_v1/mix20.pt"
PROTOCOLS="$EXISTING_ROOT/protocols/mix20_matched_v1"
BASELINES="$EXISTING_ROOT/workspaces/mix20_matched_baselines_v1"
for path in "$CONTEXT" "$DATA"; do
  test -f "$path" || { echo "缺少已有缓存：$path" >&2; exit 1; }
done
python -B scripts/test_cycle_mixer_latent_cross_attention.py
python -B scripts/evaluate_cycle_mixer_suite.py --context-data "$CONTEXT" --data "$DATA" \
  --baseline-runs "$BASELINES" --protocol-dir "$PROTOCOLS" \
  --seeds "${SEEDS[@]}" --baseline-only
for seed in "${SEEDS[@]}"; do
  for model in latent_cycle_mixer latent_cycle_conv; do
    python -B scripts/matched_baselines.py train \
      --context-data "$CONTEXT" --data "$DATA" --protocol-dir "$PROTOCOLS" \
      --workspace "workspaces/mix20_cycle_mixer_v1/${model}_seed${seed}" \
      --model "$model" --seed "$seed" --epochs 1000 \
      --batch-size 8 --accumulation 16 --evaluate-every 25 \
      --device cuda:0 --amp --skip-complete
  done
done
python -B scripts/evaluate_cycle_mixer_suite.py --context-data "$CONTEXT" --data "$DATA" \
  --baseline-runs "$BASELINES" --protocol-dir "$PROTOCOLS" --seeds "${SEEDS[@]}"
echo "训练完成并已汇总验证集。测试集没有参与训练或检查点选择。"

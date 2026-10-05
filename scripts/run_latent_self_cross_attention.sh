#!/usr/bin/env bash
# Sequential variant-A training and fixed-protocol evaluation on one GPU.
set -Eeuo pipefail

if [[ "${1:-}" == "--help" ]]; then
    cat <<'USAGE'
Usage: bash scripts/run_latent_self_cross_attention.sh [mix_20|mix_100] [all|train|evaluate]
Defaults: mix_20 all; seeds 0-7; device cuda:0.
Environment overrides: START_SEED, END_SEED, DEVICE, PROTOCOL_DIR, TRAIN_WS, EVAL_WS,
                       PROGRESS_INTERVAL_SECONDS (default 30).
Existing checkpoints/predictions are never overwritten or automatically retrained.
USAGE
    exit 0
fi

TASK="${1:-mix_20}"
STAGE="${2:-all}"
[[ $# -le 2 ]] || { echo "Too many arguments." >&2; exit 2; }
[[ "$TASK" == "mix_20" || "$TASK" == "mix_100" ]] || {
    echo "Task must be mix_20 or mix_100." >&2; exit 2;
}
[[ "$STAGE" == "all" || "$STAGE" == "train" || "$STAGE" == "evaluate" ]] || {
    echo "Stage must be all, train, or evaluate." >&2; exit 2;
}
START_SEED="${START_SEED:-0}"
END_SEED="${END_SEED:-7}"
[[ "$START_SEED" =~ ^[0-7]$ && "$END_SEED" =~ ^[0-7]$ ]] || {
    echo "START_SEED and END_SEED must be integers from 0 to 7." >&2; exit 2;
}
(( START_SEED <= END_SEED )) || { echo "Invalid seed range." >&2; exit 2; }

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export PROGRESS_INTERVAL_SECONDS="${PROGRESS_INTERVAL_SECONDS:-30}"
[[ "$OMP_NUM_THREADS" =~ ^[1-9][0-9]*$ && "$MKL_NUM_THREADS" =~ ^[1-9][0-9]*$ ]] || {
    echo "OMP_NUM_THREADS and MKL_NUM_THREADS must be positive integers." >&2; exit 2;
}
[[ "$PROGRESS_INTERVAL_SECONDS" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ &&
   "$PROGRESS_INTERVAL_SECONDS" =~ [1-9] ]] || {
    echo "PROGRESS_INTERVAL_SECONDS must be positive and finite." >&2; exit 2;
}
DEVICE="${DEVICE:-cuda:0}"
CONFIG="configs/ablation/diff_branch/batlinet_latent_self_cross_attention/$TASK.yaml"
TRAIN_WS="${TRAIN_WS:-workspaces/fair_train/batlinet_latent_self_cross_attention/$TASK}"
EVAL_WS="${EVAL_WS:-workspaces/fair_eval/batlinet_latent_self_cross_attention/$TASK/protocol_v1_formal}"
if [[ "$TASK" == "mix_20" ]]; then
    DEFAULT_PROTOCOL="/root/autodl-tmp/BatLiNet-main2/artifacts/fixed_test_support_indices/mix_20/protocol_v1"
else
    DEFAULT_PROTOCOL="/root/autodl-tmp/BatLiNet-main3/artifacts/fixed_test_support_indices/mix_100/protocol_v1"
fi
PROTOCOL_DIR="${PROTOCOL_DIR:-$DEFAULT_PROTOCOL}"
MODEL_NAME="潜在自注意力与交叉注意力"
TOTAL_RUNS=$((END_SEED - START_SEED + 1))
CURRENT_SEED="未开始"
CURRENT_PHASE="准备"
trap 'exit_code=$?; echo "[任务失败] 模型=$MODEL_NAME | 任务=$TASK | 种子=$CURRENT_SEED | 阶段=$CURRENT_PHASE | 退出码=$exit_code" >&2' ERR

[[ -d data/processed ]] || {
    echo "Missing data/processed. Link the existing processed data before running." >&2; exit 1;
}
shopt -s nullglob
# Check the whole requested range before starting an expensive training run.
for ((seed=START_SEED; seed<=END_SEED; seed++)); do
    checkpoints=("$TRAIN_WS"/*_seed_"${seed}"_epoch_1000.ckpt)
    if [[ "$STAGE" != "evaluate" ]]; then
        existing=("$TRAIN_WS"/*_seed_"${seed}"_epoch_*.ckpt)
        (( ${#existing[@]} == 0 )) || {
            echo "Checkpoint already exists for seed $seed. Use evaluate or a new TRAIN_WS." >&2; exit 1;
        }
    else
        (( ${#checkpoints[@]} == 1 )) || {
            echo "Expected exactly one epoch-1000 checkpoint for seed $seed." >&2; exit 1;
        }
    fi
    if [[ "$STAGE" != "train" ]]; then
        [[ -f "$PROTOCOL_DIR/seed_$seed.pt" ]] || {
            echo "Missing existing fixed protocol: $PROTOCOL_DIR/seed_$seed.pt" >&2; exit 1;
        }
        predictions=("$EVAL_WS"/predictions_seed_"${seed}"_*.pkl)
        (( ${#predictions[@]} == 0 )) || {
            echo "Prediction already exists for seed $seed. Use a new EVAL_WS." >&2; exit 1;
        }
    fi
done

mkdir -p "$TRAIN_WS"
if [[ "$STAGE" != "train" ]]; then
    mkdir -p "$EVAL_WS"
fi
if [[ "$STAGE" == "evaluate" ]]; then
    echo "[任务开始] 模型=$MODEL_NAME | 任务=$TASK | 阶段=$STAGE | 待训练=0次 | 待评估=${TOTAL_RUNS}次"
else
    echo "[任务开始] 模型=$MODEL_NAME | 任务=$TASK | 阶段=$STAGE | 待训练=${TOTAL_RUNS}次 | 种子=$START_SEED-$END_SEED"
fi
for ((seed=START_SEED; seed<=END_SEED; seed++)); do
    CURRENT_SEED="$seed"
    if [[ "$STAGE" != "evaluate" ]]; then
        CURRENT_PHASE="训练"
        export BATLINET_TASK="$TASK"
        export BATLINET_TRAINING_NUMBER=$((seed - START_SEED + 1))
        export BATLINET_TOTAL_TRAININGS="$TOTAL_RUNS"
        echo "[训练开始] 模型=$MODEL_NAME | 任务=$TASK | 种子=$seed | 训练=$BATLINET_TRAINING_NUMBER/$TOTAL_RUNS | 轮次=0/1000 | 当前训练预计剩余=估算中 | 后续训练=$((END_SEED - seed))次 | 正在准备数据与模型"
        python -u -B scripts/pipeline.py "$CONFIG" \
            --train True --evaluate False --device "$DEVICE" \
            --workspace "$TRAIN_WS" --seed "$seed" \
            2>&1 | tee "$TRAIN_WS/log.$seed"
        echo "[训练结束] 模型=$MODEL_NAME | 任务=$TASK | 种子=$seed | 后续训练=$((END_SEED - seed))次"
    fi
    if [[ "$STAGE" != "train" ]]; then
        CURRENT_PHASE="固定协议评估"
        checkpoints=("$TRAIN_WS"/*_seed_"${seed}"_epoch_1000.ckpt)
        (( ${#checkpoints[@]} == 1 )) || {
            echo "Expected exactly one epoch-1000 checkpoint for seed $seed." >&2; exit 1;
        }
        echo "[评估开始] 模型=$MODEL_NAME | 任务=$TASK | 种子=$seed | 协议=$PROTOCOL_DIR/seed_$seed.pt"
        python -u -B scripts/pipeline.py "$CONFIG" \
            --train False --evaluate True --device "$DEVICE" \
            --workspace "$EVAL_WS" --seed "$seed" \
            --checkpoint "${checkpoints[0]}" \
            --fixed_test_support_index_path "$PROTOCOL_DIR/seed_$seed.pt" \
            2>&1 | tee "$EVAL_WS/log.$seed"
        predictions=("$EVAL_WS"/predictions_seed_"${seed}"_*.pkl)
        (( ${#predictions[@]} == 1 )) || {
            echo "Expected exactly one prediction file for seed $seed." >&2; exit 1;
        }
        echo "[评估结束] 模型=$MODEL_NAME | 任务=$TASK | 种子=$seed | 后续评估=$((END_SEED - seed))次"
    fi
done
if [[ "$STAGE" != "train" ]]; then
    CURRENT_PHASE="汇总结果"
    python -u -B scripts/summarize_latent_self_cross_attention.py "$TASK" \
        --workspace "$EVAL_WS" --start-seed "$START_SEED" --end-seed "$END_SEED" \
        --output "$EVAL_WS/summary_seed_${START_SEED}_${END_SEED}.json"
fi
echo "[全部完成] 模型=$MODEL_NAME | 任务=$TASK | 本次阶段=$STAGE | 种子=$START_SEED-$END_SEED"

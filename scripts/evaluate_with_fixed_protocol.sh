#!/bin/bash
SCRIPT=./scripts/pipeline.py
CONFIG=$1
CHECKPOINT_DIR=$2
PROTOCOL_DIR=$3
WORKSPACE=$4
TOTAL_SEEDS=$5

if [ -z "$CONFIG" ] || [ -z "$CHECKPOINT_DIR" ] || [ -z "$PROTOCOL_DIR" ] || [ -z "$WORKSPACE" ] || [ -z "$TOTAL_SEEDS" ]; then
    echo "Usage: $0 CONFIG CHECKPOINT_DIR PROTOCOL_DIR WORKSPACE TOTAL_SEEDS"
    exit 1
fi

if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    USE_GPU=1
else
    NUM_GPUS=1
    USE_GPU=0
fi

mkdir -p "$WORKSPACE"
export PYTHONPATH=$PYTHONPATH:.

declare -A device_pids
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

for (( seed=0; seed<TOTAL_SEEDS; seed++ )); do
    device_id=$((seed % NUM_GPUS))
    if [ $USE_GPU -eq 1 ]; then
        actual_device="cuda:$device_id"
    else
        actual_device="cpu"
    fi

    if [[ -n ${device_pids[$device_id]} ]]; then
        wait ${device_pids[$device_id]}
    fi

    checkpoint=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name "*seed_${seed}*.ckpt" | head -n 1)
    protocol="$PROTOCOL_DIR/seed_${seed}.pt"

    if [ ! -f "$checkpoint" ]; then
        echo "Missing checkpoint for seed ${seed} in $CHECKPOINT_DIR"
        exit 1
    fi

    if [ ! -f "$protocol" ]; then
        echo "Missing fixed support protocol for seed ${seed}: $protocol"
        exit 1
    fi

    python "$SCRIPT" "$CONFIG" \
        --train False \
        --evaluate True \
        --device "$actual_device" \
        --workspace "$WORKSPACE" \
        --seed "$seed" \
        --checkpoint "$checkpoint" \
        --fixed_test_support_index_path "$protocol" | tee "$WORKSPACE/log.$seed" &
    device_pids[$device_id]=$!
done

for pid in ${device_pids[@]}; do
    wait $pid
done

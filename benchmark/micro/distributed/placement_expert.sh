#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
export BRT_CACHE_PATH=$SCRIPT_DIR/../../../.cache

GPU_SETTINGS=(
    "0,1"
    "0,1,2,3"
    "0,1,2,3,4,5,6,7"
)
EXPERTS=(
    1
    2
    4
    8
    16
    32
    64
)
CELL_SIZE=512
LOAD=1024
BENCH_ITEMS=(
    "imbalance"
    "balance"
)

for benchmark in "${BENCH_ITEMS[@]}"; do
    for GPUS in "${GPU_SETTINGS[@]}"; do
        GPU_NUM=$(($(echo "$GPUS" | tr -cd , | wc -c) + 1))
        PROC=$GPU_NUM
        export CUDA_VISIBLE_DEVICES=$GPUS
        for EXPERT_NUM in "${EXPERTS[@]}"; do
            LAUNCH_ARGS=()
            LAUNCH_ARGS+=(--benchmark "$benchmark")
            LAUNCH_ARGS+=(--local-expert "$EXPERT_NUM")
            LAUNCH_ARGS+=(--cell-size "$CELL_SIZE")
            echo "Running with $GPU_NUM GPUs, $EXPERT_NUM experts, $CELL_SIZE cell size, $LOAD load"
            echo "Launching with args: " "${LAUNCH_ARGS[@]}"
            torchrun --nproc_per_node="$PROC" \
                --nnode=1 --node_rank=0 \
                --master_addr=127.0.0.1 --master_port=6500 \
                placement_a2a.py "${LAUNCH_ARGS[@]}"
        done
    done
done


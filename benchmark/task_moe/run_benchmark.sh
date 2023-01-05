#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
BRT_DIR=$(cd "${script_dir}/../../" && pwd)

export BRT_CACHE_PATH=$BRT_DIR/.cache
export BRT_CAPTURED_FABRIC_TYPE=dispatch

LAUNCH_ARGS=()

ARGUMENT_LIST=(
    "gpus:"
    "opt:"
    "mode:"
)

# read arguments
opts=$(
    getopt \
        --longoptions "$(printf "%s," "${ARGUMENT_LIST[@]}")" \
        --name "$(basename "$0")" \
        --options "" \
        -- "$@"
)

eval set -- "$opts"

while [[ $# -gt 0 ]]; do
    case "$1" in
    --gpus)
        GPUS=$2
        GPU_NUM=$(($(echo "$GPUS" | tr -cd , | wc -c) + 1))
        shift 2
        ;;
    --mode)
        MODE=$2
        LAUNCH_ARGS+=(--mode "$MODE")
        shift 2
        ;;
    --opt)
        OPT=$2
        LAUNCH_ARGS+=(--opt "$OPT")
        shift 2
        ;;
    *)
        break
        ;;
    esac
done

# check if the number of GPUs is specified
if [ -z "$GPUS" ]; then
    echo "Please specify the number of GPUs to use with --gpus"
    exit 1
fi

PROC=$GPU_NUM
export CUDA_VISIBLE_DEVICES=$GPUS

echo "Final launch args:" "${LAUNCH_ARGS[@]}"

torchrun --nproc_per_node="$PROC" \
    --nnode=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port=6500 \
    benchmark.py "${LAUNCH_ARGS[@]}"

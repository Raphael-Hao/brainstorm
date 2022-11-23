#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
BRT_DIR=$(cd "${script_dir}/../../" && pwd)

export BRT_CACHE_PATH=$BRT_DIR/.cache
export BRT_CAPTURE_STATS=False # should be False for brt_dist or tutel
export BRT_CAPTURED_FABRIC_TYPE=dispatch

LAUNCH_ARGS=()

ARGUMENT_LIST=(
    "gpus:"
    "locality"
    "placement:"
    "port:"
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

    --locality)
        LAUNCH_ARGS+=(--locality)
        shift 1
        ;;

    --placement)
        PLACEMENT=$2
        LAUNCH_ARGS+=(--placement "$PLACEMENT")
        shift 2
        ;;

    --port)
        PORT=$2
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

EXPERT_NUM=$((16 / GPU_NUM))

if ((GPU_NUM == 1)); then
    export MOE_LAYER_VENDOR=brt
else
    export MOE_LAYER_VENDOR=brt_dist
fi

if [ -z "$PORT" ]; then
    PORT=$(((RANDOM % 10) + 6500))
fi

LAUNCH_ARGS+=(
    --cfg configs/"${EXPERT_NUM}"expert_"${GPU_NUM}"GPU.yaml
    --batch-size 128 --data-path "${BRT_CACHE_PATH}"/datasets/imagenet22k
    --output ./results/MoE/
    --eval --resume "${BRT_CACHE_PATH}"/ckpt/swin_moe/small_swin_moe_32GPU_16expert/model.pth
    --throughput
)

torchrun --nproc_per_node="$PROC" \
    --nnode=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port="$PORT" \
    benchmark.py "${LAUNCH_ARGS[@]}"

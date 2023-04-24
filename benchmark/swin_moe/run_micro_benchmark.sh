#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
BRT_DIR=$(cd "${script_dir}/../../" && pwd)

export BRT_CACHE_PATH=$BRT_DIR/.cache
export BRT_CAPTURED_FABRIC_TYPE=dispatch

LAUNCH_ARGS=()

ARGUMENT_LIST=(
    "vendor:"
    "gpus:"
    "locality"
    "placement:"
    "port:"
    "mode:"
    "blocking"
    "bs:"
    "capacity:"
    "moe-id:"
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
    --vendor)
        VENDOR=$2
        shift 2
        ;;
    --gpus)
        GPUS=$2
        GPU_NUM=$(($(echo "$GPUS" | tr -cd , | wc -c) + 1))
        shift 2
        ;;
    --locality)
        LOCALITY=1
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
    --mode)
        MODE=$2
        LAUNCH_ARGS+=(--mode "$MODE")
        shift 2
        ;;
    --blocking)
        export CUDA_LAUNCH_BLOCKING=1
        shift 1
        ;;
    --bs)
        BS=$2
        shift 2
        ;;
    --capacity)
        CAPACITY=$2
        LAUNCH_ARGS+=(--capacity "$CAPACITY")
        shift 2
        ;;
    --moe-id)
        MOE_ID=$2
        LAUNCH_ARGS+=(--moe-id "$MOE_ID")
        shift 2
        ;;
    *)
        break
        ;;
    esac
done

if [[ "${MODE}" == "trace" ]]; then
    export BRT_CAPTURE_STATS=True # should be False for brt_dist or tutel
else
    export BRT_CAPTURE_STATS=False # should be False for brt_dist or tutel
fi

# check if the number of GPUs is specified
if [ -z "$GPUS" ]; then
    echo "Please specify the number of GPUs to use with --gpus"
    exit 1
fi

PROC=$GPU_NUM
export CUDA_VISIBLE_DEVICES=$GPUS

EXPERT_NUM=$((16 / GPU_NUM))

if [[ "${VENDOR}" == "brt" ]]; then
    if ((GPU_NUM == 1)); then
        export MOE_LAYER_VENDOR=brt
    else
        export MOE_LAYER_VENDOR=brt_dist
    fi
else
    if [ -n "$LOCALITY" ] || [ -n "$PLACEMENT" ]; then
        echo "locality should not be specified for non-brt vendor"
        exit 1
    fi
    export MOE_LAYER_VENDOR=$VENDOR
fi

if [ -z "$BS" ]; then
    BS=128
    echo "Batch size is not specified, using default value $BS"
fi

if [ -z "$PORT" ]; then
    PORT=$(((RANDOM % 10) + 6500))
    echo "Port is not specified, using random generated port: $PORT"
fi

LAUNCH_ARGS+=(
    --cfg configs/"${EXPERT_NUM}"expert_"${GPU_NUM}"GPU.yaml
    --batch-size "$BS"
    --data-path "${BRT_CACHE_PATH}"/dataset/imagenet22k
    --output ./results/MoE/
    --resume "${BRT_CACHE_PATH}"/ckpt/swinv2_moe_small/model.pth
    --seed "$RANDOM"

)

echo "Final launch args:" "${LAUNCH_ARGS[@]}"

torchrun --nproc_per_node="$PROC" \
    --nnode=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port="$PORT" \
    micro_benchmark.py "${LAUNCH_ARGS[@]}"

#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
BRT_DIR=$(cd "${script_dir}/../../" && pwd)

export BRT_CACHE_PATH=$BRT_DIR/.cache
export BRT_CAPTURE_STATS=False # should be False for brt_dist or tutel
export BRT_CAPTURED_FABRIC_TYPE=dispatch
export MOE_LAYER_VENDOR=tutel #_dist #_dist # tutel, brt, or brt_dist

if [[ $1 == "--gpus" ]]; then
    GPUS=$2
    GPU_NUM=$(($(echo "$GPUS" | tr -cd , | wc -c) + 1))
    PROC=$GPU_NUM
    shift 2
else
    echo "No GPU specified, Please specify GPUs like --gpus 0,1,2,3"
fi

export CUDA_VISIBLE_DEVICES=$GPUS

EXPERT_NUM=$((16 / GPU_NUM))

if [[ $1 == "--port" ]]; then
    PORT=$2
    shift 2
else
    PORT=$(((RANDOM % 10) + 6500))
fi


torchrun --nproc_per_node="$PROC" \
    --nnode=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port="$PORT" \
    benchmark.py --cfg configs/"${EXPERT_NUM}"expert_"${GPU_NUM}"GPU.yaml \
    --batch-size 256 --data-path "${BRT_CACHE_PATH}"/datasets/imagenet22k --output ./results/MoE/ \
    --eval --resume "${BRT_CACHE_PATH}"/ckpt/swin_moe/small_swin_moe_32GPU_16expert/model.pth \
    --throughput

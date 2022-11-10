#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BRT_DIR=$(cd "${script_dir}/../../" && pwd)
export BRT_CACHE_PATH=$BRT_DIR/.cache
export BRT_CAPTURE_STATS=False # should be False for brt_dist or tutel
export BRT_CAPTURED_FABRIC_TYPE=dispatch
export MOE_LAYER_VENDOR=brt_dist # tutel, brt, or brt_dist

torchrun --nproc_per_node=2 \
    --nnode=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port=6500 \
    benchmark.py --cfg configs/moe.yaml \
    --batch-size 128 --data-path "${BRT_CACHE_PATH}"/datasets/imagenet22k --output ./results/MoE/ \
    --eval --single-gpu-eval --resume "${BRT_CACHE_PATH}"/ckpt/swin_moe/small_swin_moe_32GPU_16expert/model.pth \
    --debug

#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

export BRT_CACHE_PATH=$HOME/brainstorm_project/brainstorm/.cache
export BRT_CAPTURE_STATS=True
export BRT_CAPTURED_FABRIC_TYPE=dispatch
export MOE_LAYER_VENDOR=brt

torchrun --nproc_per_node=1 \
    --nnode=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port=6500 \
    benchmark.py --cfg configs/swinv2_moe_small_pre_nattn_cpb_patch4_window12_192_s2it2_s3b1_top1_vitmoeloss_GwoN_bpr_cap125_moedrop01_nobias_22k_32gpu_16exp.yaml \
    --batch-size 128 --data-path ../../.cache/datasets/imagenet22k --output ./results/MoE/ \
    --eval --single-gpu-eval --resume ../../.cache/ckpt/swin_moe/small_swin_moe_32GPU_16expert/model.pth \
    --trace

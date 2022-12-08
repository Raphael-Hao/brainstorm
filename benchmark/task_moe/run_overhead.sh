#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
BRT_DIR=$(cd "${script_dir}/../../" && pwd)

export BRT_CACHE_PATH=$BRT_DIR/.cache
export BRT_CAPTURED_FABRIC_TYPE=dispatch

export CUDA_VISIBLE_DEVICES=0

echo "Final launch args:" "${LAUNCH_ARGS[@]}"
export CUDA_LAUNCH_BLOCKING=1

torchrun --nproc_per_node=1 \
    --nnode=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port=6500 \
    overhead.py "${LAUNCH_ARGS[@]}"

#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
BRT_DIR=$(cd "${script_dir}/../../" && pwd)

export BRT_CACHE_PATH=$BRT_DIR/.cache
rm -rf "$BRT_CACHE_PATH"/results/task_moe/throughput.csv

vendors=(
    pytorch
    None
    placement
)

GPUS=(
    "0,1"
    "0,1,2,3"
    "0,1,2,3,4,5,6,7"
)

for vendor in "${vendors[@]}"; do
    for gpu in "${GPUS[@]}"; do
        bash run_benchmark.sh --mode throughput --opt "$vendor" --gpus "$gpu" --seq 256 --token 32
        bash run_benchmark.sh --mode throughput --opt "$vendor" --gpus "$gpu" --seq 256 --token 64
        bash run_benchmark.sh --mode throughput --opt "$vendor" --gpus "$gpu" --seq 512 --token 32
    done
done

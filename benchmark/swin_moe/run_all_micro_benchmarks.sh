#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

BRT_DIR=$(cd "${script_dir}/../../" && pwd)
export BRT_CACHE_PATH=$BRT_DIR/.cache
rm "$BRT_CACHE_PATH"/results/swin_moe/micro_throughput.csv

CAPACITIES=(
    1.25
    2
    3
    4
)
GPUS="0,1,2,3,4,5,6,7"
# GPUS="0,1"

for capacity in "${CAPACITIES[@]}"; do
    for layer_id in {0..9}; do
        echo "Running benchmark for capacity: $capacity, layer_id: $layer_id"
        bash run_micro_benchmark.sh --gpus $GPUS --vendor brt_dist --mode bench-searched \
            --moe-id $layer_id --capacity $capacity
    done
done

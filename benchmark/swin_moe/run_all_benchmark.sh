#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.


CAPACITIES=(
    1.25
    # 2
    # 3
    # 4
)
VENDORS=(
    # pt
    # tutel
    brt_dist
)
GPUS=(
    "0,1"
    # "0,1,2,3"
    # "0,1,2,3,4,5,6,7"
)
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
BRT_DIR=$(cd "${script_dir}/../../" && pwd)
export BRT_CACHE_PATH=$BRT_DIR/.cache
rm "$BRT_CACHE_PATH"/results/swin_moe/e2e.csv

for vendor in "${VENDORS[@]}"; do
    for capacity in "${CAPACITIES[@]}"; do
        for gpus in "${GPUS[@]}"; do
            echo "Running benchmark for vendor: $vendor, capacity: $capacity, gpus: $gpus"
            bash run_benchmark.sh --vendor "$vendor" --capacity "$capacity" --gpus "$gpus" --mode throughput
            # bash run_benchmark.sh --vendor "$vendor" --capacity "$capacity" --gpus "$gpus" --mode correctness
        done
    done
    if [[ "${vendor}" == "brt_dist" ]]; then
        for capacity in "${CAPACITIES[@]}"; do
            for gpus in "${GPUS[@]}"; do
                echo "Running benchmark for vendor: $vendor, capacity: $capacity, gpus: $gpus"
                bash run_benchmark.sh --vendor "$vendor" --capacity "$capacity" --gpus "$gpus" --mode throughput --placement
            done
        done
    fi
done
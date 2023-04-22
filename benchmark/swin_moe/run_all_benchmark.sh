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
    # tutel
    # pt
    brt_dist
)
GPUS=(
    "0,1"
    # "0,1,2,3"
    # "0,1,2,3,4,5,6,7"
)

for vendor in "${VENDORS[@]}"; do
    for capacity in "${CAPACITIES[@]}"; do
        for gpus in "${GPUS[@]}"; do
            echo "Running benchmark for vendor: $vendor, capacity: $capacity, gpus: $gpus"
            bash run_benchmark.sh --vendor "$vendor" --capacity "$capacity" --gpus "$gpus" --mode correctness
        done
    done
done
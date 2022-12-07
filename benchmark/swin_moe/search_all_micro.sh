#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

gpus_settings=(
    # "0,1"
    # "0,1,2,3"
    "0,1,2,3,4,5,6,7"
)
capacity_factors=(
    1.25
    2.0
    3.0
    4.0
)
moe_layers=(
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
)

for gpus in "${gpus_settings[@]}"; do
    export CUDA_VISIBLE_DEVICES="$gpus"
    for cap_f in "${capacity_factors[@]}"; do
        for moe_id in "${moe_layers[@]}"; do
            bash run_micro_benchmark.sh --gpus "$gpus" --mode search-end --capacity "$cap_f" --vendor brt --moe-id "$moe_id"
        done
    done
done

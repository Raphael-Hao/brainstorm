#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
export BRT_CACHE_PATH=$SCRIPT_DIR/../../../.cache
rm -rf $BRT_CACHE_PATH/results/micro/speculative/route_e2e.csv

BRANCHES=(
    # 2
    # 3
    # 4
    # 5
    # 6
    # 7
    8
    # 9
    # 10
    # 11
    # 12
    # 13
    # 14
    # 15
    # 16
)

CELL_SIZES=(
    # 128
    # 256
    # 384
    # 512
    # 640
    # 768
    # 896
    1024
    # 2048
)

UNROLL_INDICES=(
    0.000087 #4
    0.000174 #8
    0.000261 #12
    0.000348 #16
    0.000435 #20
)

for cell_size in "${CELL_SIZES[@]}"; do
    for branch in "${BRANCHES[@]}"; do
        for unroll_index in "${UNROLL_INDICES[@]}"; do
            echo "Running branch $branch, cell_size $cell_size, unroll_index $unroll_index"
            python route.py --path-num "$branch" --cell-size "$cell_size" --time "$unroll_index"
        done
    done
done

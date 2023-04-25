#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
export BRT_CACHE_PATH=$SCRIPT_DIR/../../../.cache

rm -rf $BRT_CACHE_PATH/results/micro/speculative/load_e2e.csv

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
    64
    128
    192
    256
    320
    384
    448
    512
    576
    640
)

for branch in "${BRANCHES[@]}"; do
    for cell_size in "${CELL_SIZES[@]}"; do
        echo "Running branch $branch, cell_size $cell_size"
        python load.py --path-num "$branch" --cell-size "$cell_size"
    done
done

#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
export BRT_CACHE_PATH=$SCRIPT_DIR/../../../.cache
BRANCHES=(
    # 2
    # 3
    # 4
    # 5
    # 6
    # 7
    # 8
    # 9
    # 10
    # 11
    12
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
)

UNROLL_INDICES=(
    # 2
    4
    # 6
    # 8
    # 10
    # 12
    # 14
    # 16
    # 18
    # 20
)

for cell_size in "${CELL_SIZES[@]}"; do
    for branch in "${BRANCHES[@]}"; do
        for unroll_index in "${UNROLL_INDICES[@]}"; do
            echo "Running branch $branch, cell_size $cell_size, unroll_index $unroll_index"
            python route.py --path-num "$branch" --cell-size "$cell_size" --unroll-index "$unroll_index"
        done
    done
done

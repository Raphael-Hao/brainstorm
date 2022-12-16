#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
export BRT_CACHE_PATH=$SCRIPT_DIR/../../../.cache

EXPERTS=(
    # 1
    2
    3
    4
    # 5
    # 6
    # 7
    # 8
    # 9
    # 10
    # 11
    # 12
    # 14
    # 15
    # 16
)
TOKENS=(
    # 1
    # 2
    # 4
    # 8
    # 16
    # 32
    # 64
    128
    # 256
    # 512
    # 1024
)
BENCH_ITEMS=(
    # brt
    brt_homo
    # matmul
)
for bench in "${BENCH_ITEMS[@]}"; do
    for expert in "${EXPERTS[@]}"; do
        for token in "${TOKENS[@]}"; do
            echo "Running benchmark for $bench with $expert experts and $token tokens"
            python micro_thor.py --bench "$bench" --expert "$expert" --token "$token"
        done
    done
done

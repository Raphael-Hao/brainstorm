#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

EXPERTS=(
    2
    4
    8
    16
)
TOKENS=(
    32
    64
    96
    128
    160
    192
    224
    256
)
BENCH_ITEMS=(
    brt
    brt_homo
)
for bench in "${BENCH_ITEMS[@]}"; do
    for expert in "${EXPERTS[@]}"; do
        for token in "${TOKENS[@]}"; do
            echo "Running benchmark for $bench with $expert experts and $token tokens"
            bash run_benchmark.sh --benchmark "$bench" --expert "$expert" --token "$token"
        done
    done
done

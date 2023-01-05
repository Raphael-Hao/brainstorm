#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

vendors=(
    brt
    torch
    batchmatmul
)

experts=(
    8
    16
    32
    64
    128
    256
)

for vendor in "${vendors[@]}"; do
    for expert in "${experts[@]}"; do
        bash run_benchmark.sh --mode throughput --vendor "$vendor" --expert "$expert"
    done
done

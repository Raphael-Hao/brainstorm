#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
BRT_DIR=$(cd "${script_dir}/../../" && pwd)

export BRT_CACHE_PATH=$BRT_DIR/.cache
rm -rf "$BRT_CACHE_PATH"/results/switch_moe/e2e.csv

vendors=(
    torch
    batchmatmul
    brt
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

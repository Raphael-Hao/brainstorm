#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export BRT_CACHE_PATH="$script_dir/../../.cache"
rm -rf "$BRT_CACHE_PATH"/results/dynamic_routing/speculative_load*.csv

ARCHS=(
    A
    B
    C
    Raw
)

for arch in "${ARCHS[@]}"; do
    echo "Running benchmark for arch $arch"
    python benchmark.py --arch "$arch" --benchmark=dce_memory_plan --memory-mode=predict
    python benchmark.py --arch "$arch" --benchmark=dce_memory_plan --memory-mode=on_demand
done

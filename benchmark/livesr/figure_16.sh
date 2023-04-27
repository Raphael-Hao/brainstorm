#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
export BRT_CACHE_PATH=$SCRIPT_DIR/../../.cache
rm -rf $BRT_CACHE_PATH/results/benchmark_livesr.csv

. ./benchmark.sh
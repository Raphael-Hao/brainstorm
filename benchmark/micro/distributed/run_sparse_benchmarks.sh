#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
export BRT_CACHE_PATH=$SCRIPT_DIR/../../../.cache
rm -rf $BRT_CACHE_PATH/results/micro/distributed/sparse*.csv

bash bench_expert.sh
bash bench_cellsize.sh
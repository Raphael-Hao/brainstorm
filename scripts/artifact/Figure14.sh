#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# run_all_benchmark.sh
cd $SCRIPT_DIR/../../benchmark/micro/speculative || exit
bash run_route_benchmarks.sh
bash run_load_benchmarks.sh
cd $SCRIPT_DIR
# visualize the results
python visualize/figure15.py
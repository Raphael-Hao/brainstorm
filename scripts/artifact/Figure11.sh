#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# run_all_benchmark.sh
cd $SCRIPT_DIR/../../benchmark/micro/distributed || exit
bash run_sparse_benchmarks.sh
cd $SCRIPT_DIR
# visualize the results
python visualize/figure11.py
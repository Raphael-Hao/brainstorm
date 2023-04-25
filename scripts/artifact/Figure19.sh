#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

# run_all_benchmark.sh

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

cd $SCRIPT_DIR/../../benchmark/swin_moe || exit
bash run_all_micro_benchmarks.sh

cd $SCRIPT_DIR
# visualize the results
python visualize/figure19.py
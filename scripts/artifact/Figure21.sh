#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# run_all_benchmark.sh
cd $SCRIPT_DIR/../../benchmark/dynamic_routing
bash run_all_route_benchmark.sh

cd $SCRIPT_DIR
python visualize/figure21.py


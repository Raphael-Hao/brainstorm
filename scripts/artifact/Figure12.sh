#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

cd $SCRIPT_DIR/../../benchmark/micro/homo_conv || exit
bash figure_12.sh
cd $SCRIPT_DIR

python visualize/figure12.py

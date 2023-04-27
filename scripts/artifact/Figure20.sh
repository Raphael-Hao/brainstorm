#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

cd $SCRIPT_DIR/../../benchmark/msdnet || exit
bash figure_20.sh
cd $SCRIPT_DIR

python visualize/figure20.py

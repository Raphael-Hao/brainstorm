#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

cd $SCRIPT_DIR/../../benchmark/livesr || exit
bash figure_16.sh
cd $SCRIPT_DIR

python visualize/figure16.py
#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

rm -rf "$SCRIPT_DIR"/../../.cache/results

bash Figure11.sh
bash Figure13.sh
bash Figure17.sh
bash Figure18.sh
bash Figure19.sh

cd "$SCRIPT_DIR"/../../.cache/results || exit
# compress figures and show md5sum
tar -czf figures_multi.tar.gz figures
md5sum figures_multi.tar.gz

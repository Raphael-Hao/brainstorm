#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

rm -rf "$SCRIPT_DIR"/../../.cache/results

bash Figure12.sh
bash Figure14.sh
bash Figure15.sh
bash Figure16.sh
bash Figure20.sh
bash Figure21.sh
bash Figure22.sh

cd "$SCRIPT_DIR"/../../.cache/results || exit
# compress figures and show md5sum
tar -czf figures_single.tar.gz figures
md5sum figures_single.tar.gz

#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.


script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
BRT_DIR=$(cd "${script_dir}/../../" && pwd)

export BRT_CACHE_PATH=$BRT_DIR/.cache

LAUNCH_ARGS=()

ARGUMENT_LIST=(
    "vendor:"
    "mode:"
    "expert:"
)

# read arguments
opts=$(
    getopt \
        --longoptions "$(printf "%s," "${ARGUMENT_LIST[@]}")" \
        --name "$(basename "$0")" \
        --options "" \
        -- "$@"
)

eval set -- "$opts"

while [[ $# -gt 0 ]]; do
    case "$1" in
    --mode)
        MODE=$2
        LAUNCH_ARGS+=(--mode "$MODE")
        shift 2
        ;;
    --vendor)
        VENDOR=$2
        LAUNCH_ARGS+=(--vendor "$VENDOR")
        shift 2
        ;;
    --expert)
        EXPERT=$2
        LAUNCH_ARGS+=(--expert "$EXPERT")
        shift 2
        ;;
    *)
        break
        ;;
    esac
done

echo "Final launch args:" "${LAUNCH_ARGS[@]}"

python benchmark.py "${LAUNCH_ARGS[@]}"

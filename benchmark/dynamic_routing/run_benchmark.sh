#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

ARGUMENT_LIST=(
    "arch:"        # an option with a required argument
    "benchmark:"   # an option with a required argument
    "memory-mode:" # an option with a required argument
)

# read arguments
opts=$(
    getopt \
        --longoptions "$(printf "%s," "${ARGUMENT_LIST[@]}")" \
        --name "$(basename "$0")" \
        --options "" \
        -- "$@"
)

eval set --"$opts"

while [[ $# -gt 0 ]]; do
    case "$1" in
    --arch)
        ARCH=$2
        shift 2
        ;;
    --benchmark)
        BENCHMARK=$2
        shift 2
        ;;
    --memory-mode)
        MEMORY_MODE=$2
        shift 2
        ;;
    *)
        break
        ;;
    esac
done

LAUNCH_ARGS=(
    --arch "$ARCH"
    --benchmark "$BENCHMARK"
)

if [[ "${BENCHMARK}" == "memory_plan" ]]; then
    if [[ -n "${MEMORY_MODE}" ]]; then
        LAUNCH_ARGS+=(
            --memory-mode "$MEMORY_MODE"
        )
    fi
fi

export BRT_CACHE_PATH="$script_dir/../../.cache"
export BRT_CAPTURE_STATS=True
export BRT_CAPTURED_FABRIC_TYPE=dispatch,combine
python benchmark.py "${LAUNCH_ARGS[@]}"

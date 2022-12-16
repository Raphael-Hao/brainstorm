#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
export BRT_CACHE_PATH=$SCRIPT_DIR/../../.cache

ARGUMENT_LIST=(
    "benchmark:"
    "token:"  # an option with a required argument
    "expert:" # an option with a required argument
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

LAUNCH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
    --benchmark)
        BENCH=$2
        shift 2
        ;;
    --token)
        TOKEN=$2
        LAUNCH_ARGS+=("--token" "$TOKEN")
        shift 2
        ;;

    --expert)
        EXPERT=$2
        LAUNCH_ARGS+=("--expert" "$EXPERT")
        shift 2
        ;;
    *)
        break
        ;;
    esac
done

if [[ "${BENCH}" == "brt" ]]; then
    python brt_no_opt_thor.py "${LAUNCH_ARGS[@]}"
elif [[ "${BENCH}" == "brt_homo" ]]; then
    python brt_homo_thor.py "${LAUNCH_ARGS[@]}"
elif [[ "${BENCH}" == "brt_bmm" ]]; then
    python brt_bmm_thor.py "${LAUNCH_ARGS[@]}"
fi

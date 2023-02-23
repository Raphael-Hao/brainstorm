#!/usr/bin/env bash

export BRT_CAPTURE_STATS=True
export BRT_CAPTURED_FABRIC_TYPE=dispatch,combine
export BRT_CACHE_PATH="${HOME}/brainstorm_project/brainstorm/.cache"

python benchmark.py
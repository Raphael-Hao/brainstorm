#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

export BRT_CACHE_PATH=/home/whcui/brainstorm_project/brainstorm/.cache
export BRT_CAPTURE_STATS=True
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.3

python benchmark.py --debug
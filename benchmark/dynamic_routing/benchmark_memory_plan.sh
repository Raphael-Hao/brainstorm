#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

export BRT_CACHE_PATH=$HOME/brainstorm_project/brainstorm/.cache
export BRT_CAPTURE_STATS=True
python benchmark.py --arch A --benchmark memory_plan
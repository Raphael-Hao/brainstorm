#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.


BENCH_ITEMS=(
    liveness
    memory_plan
)

ARCHS=(
    raw
    A
    B
    C
)
LIVENESS_OPTS=(
)

MEMORY_PLAN_OPTS=(
    --memory-mode "on_demand"
    --memory-mode "predict"
)


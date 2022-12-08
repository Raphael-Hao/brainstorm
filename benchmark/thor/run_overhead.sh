#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.



export BRT_CAPTURE_STATS=True
# export BRT_CAPTURED_FABRIC_TYPE=dispatch,combine
# export BRT_CAPTURED_FABRIC_TYPE=combine
export BRT_CAPTURED_FABRIC_TYPE=dispatch
python overhead.py --expert 2 --token 64

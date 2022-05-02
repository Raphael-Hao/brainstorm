#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

# rename *_tune.log to *.log
rename 's/_tune.log/\.log/' ./*_tune.log
#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /init.sh
# \brief:
# Author: raphael hao

set -e

cp -r ./vscode ../.vscode
bash blob/mount.sh
bash blob/prepare_dataset_ckpt.sh
#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /download_image.sh
# \brief:
# Author: raphael hao

ds_ckpt_script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)
azcopy copy "https://projectbrainstorm.blob.core.windows.net/artifact?sp=rl&st=2023-11-13T04:54:53Z&se=2030-07-30T12:54:53Z&spr=https&sv=2022-11-02&sr=c&sig=3VAyGJlCJdazyEekOM6sLS2MZ45TKIIv29SMxvsc3jw%3D" "$HOME/artifact" --recursive
ln -s "$HOME/artifact/dataset" "$ds_ckpt_script_dir"/../.cache/dataset
ln -s "$HOME/artifact/dataset/cityscapes" "$ds_ckpt_script_dir"/../3rdparty/dynamic_routing/datasets/cityscapes
ln -s "$HOME/artifact/ckpt" "$ds_ckpt_script_dir"/../.cache/ckpt
ln -s "$HOME/artifact/kernel_db.sqlite" "$ds_ckpt_script_dir"/../.cache/kernel_db.sqlite

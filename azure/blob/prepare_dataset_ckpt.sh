#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /download_image.sh
# \brief:
# Author: raphael hao

ds_ckpt_script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)
azcopy copy "https://projectbrainstorm.blob.core.windows.net/largedata/dataset/?sp=racwdli&st=2022-09-23T02:27:48Z&se=2023-02-01T10:27:48Z&spr=https&sv=2021-06-08&sr=c&sig=e%2B%2F9PGQi5%2B8g4bXsuJ14AHrk98RohKx51QFBlAWwIWg%3D" "$HOME/" --recursive
ln -s "$HOME/dataset" "$ds_ckpt_script_dir"/../../.cache/dataset
azcopy copy "https://projectbrainstorm.blob.core.windows.net/largedata/ckpt/?sp=racwdli&st=2022-09-23T02:27:48Z&se=2023-02-01T10:27:48Z&spr=https&sv=2021-06-08&sr=c&sig=e%2B%2F9PGQi5%2B8g4bXsuJ14AHrk98RohKx51QFBlAWwIWg%3D" "$HOME/" --recursive
ln -s "$HOME/ckpt" "$ds_ckpt_script_dir"/../../.cache/ckpt
ln -s "$ds_ckpt_script_dir"/../../.cache/dataset/cityscapes "$ds_ckpt_script_dir"/../../3rdparty/dynamic_routing/datasets/cityscapes
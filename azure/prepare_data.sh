#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /download_image.sh
# \brief:
# Author: raphael hao

ds_ckpt_script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)
azcopy copy "https://projectbrainstorm.blob.core.windows.net/artifact/dataset/?sp=rl&st=2023-04-18T04:28:14Z&se=2023-09-01T12:28:14Z&spr=https&sv=2021-12-02&sr=c&sig=ORudu98c5ghmoRBD2aiVjwKbBF3h4CxnnuZis%2FgHKo4%3D" "$HOME/" --recursive
ln -s "$HOME/dataset" "$ds_ckpt_script_dir"/../../.cache/dataset
azcopy copy "https://projectbrainstorm.blob.core.windows.net/artifact/ckpt/?sp=rl&st=2023-04-18T04:28:14Z&se=2023-09-01T12:28:14Z&spr=https&sv=2021-12-02&sr=c&sig=ORudu98c5ghmoRBD2aiVjwKbBF3h4CxnnuZis%2FgHKo4%3D" "$HOME/" --recursive
ln -s "$HOME/ckpt" "$ds_ckpt_script_dir"/../../.cache/ckpt
ln -s "$ds_ckpt_script_dir"/../../.cache/dataset/cityscapes "$ds_ckpt_script_dir"/../../3rdparty/dynamic_routing/datasets/cityscapes
azcopy copy  "https://projectbrainstorm.blob.core.windows.net/artifact/kernel_db.sqlite/?sp=rl&st=2023-04-18T04:28:14Z&se=2023-09-01T12:28:14Z&spr=https&sv=2021-12-02&sr=c&sig=ORudu98c5ghmoRBD2aiVjwKbBF3h4CxnnuZis%2FgHKo4%3D" "$$HOME/kernel_db.sqlite"
ln -s "$HOME/kernel_db.sqlite" "$ds_ckpt_script_dir"/../../.cache/kernel_db.sqlite
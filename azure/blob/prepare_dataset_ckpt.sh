#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /download_image.sh
# \brief:
# Author: raphael hao

ds_ckpt_script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)
mkdir -p ~/dataset
azcopy copy "https://projectbrainstorm.blob.core.windows.net/largedata/imagenet22k/?sp=racwdli&st=2022-09-23T02:27:48Z&se=2023-02-01T10:27:48Z&spr=https&sv=2021-06-08&sr=c&sig=e%2B%2F9PGQi5%2B8g4bXsuJ14AHrk98RohKx51QFBlAWwIWg%3D" "$HOME/dataset/" --recursive
mkdir -p "$ds_ckpt_script_dir"/../../.cache/datasets
ln -s ~/dataset/imagenet22k/ "$ds_ckpt_script_dir"/../../.cache/datasets/imagenet22k
mkdir -p ~/ckpt
azcopy copy "https://projectbrainstorm.blob.core.windows.net/largedata/swinv2_moe_small_pre_nattn_cpb_patch4_window12_192_s2it2_s3b1_top1_vitmoeloss_GwoN_bpr_cap125_moedrop01_nobias_22k_32gpu_16exp/?sp=racwdli&st=2022-09-23T02:27:48Z&se=2023-02-01T10:27:48Z&spr=https&sv=2021-06-08&sr=c&sig=e%2B%2F9PGQi5%2B8g4bXsuJ14AHrk98RohKx51QFBlAWwIWg%3D" "$HOME/ckpt/" --recursive
mkdir -p "$ds_ckpt_script_dir"/../../.cache/ckpt/swin_moe/
ln -s ~/ckpt/swinv2_moe_small_pre_nattn_cpb_patch4_window12_192_s2it2_s3b1_top1_vitmoeloss_GwoN_bpr_cap125_moedrop01_nobias_22k_32gpu_16exp/ "$ds_ckpt_script_dir"/../../.cache/ckpt/swin_moe/small_swin_moe_32GPU_16expert
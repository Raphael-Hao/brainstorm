#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /mount.sh
# \brief:
# Author: raphael hao

sudo mkdir /mnt/ramdisk
sudo mount -t tmpfs -o size=16g tmpfs /mnt/ramdisk
sudo mkdir /mnt/ramdisk/blobfusetmp
blobfuse ~/datasets/swin_moe --tmp-path=/mnt/ramdisk/blobfusetmp  --config-file=~/dotfile/blobfuse.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120

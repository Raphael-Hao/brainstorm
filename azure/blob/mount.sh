#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /mount.sh
# \brief:
# Author: raphael hao

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)
mkdir /mnt/ramdisk
mount -t tmpfs -o size=16g tmpfs /mnt/ramdisk
mkdir /mnt/ramdisk/blobfusetmp
chown "$(whoami)" /mnt/ramdisk/blobfusetmp
chmod 600 "$script_dir/blobfuse.cfg"
mkdir ~/largedata
blobfuse ~/largedata --tmp-path=/mnt/ramdisk/blobfusetmp --config-file="$script_dir/blobfuse.cfg" -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120

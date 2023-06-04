#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /init.sh
# \brief:
# Author: raphael hao

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

wget -O azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux -O azcopy.tar.gz
mkdir -p azcopy && tar -xzvf azcopy.tar.gz -C "azcopy" --strip-components=1
mv azcopy/azcopy "$HOME"/.local/bin/azcopy && rm -rf azcopy.tar.gz azcopy

export PATH="$HOME"/.local/bin:"$PATH"

bash "$script_dir"/prepare_data.sh

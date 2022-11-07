#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /install.sh
# \brief:
# Author: raphael hao
set -e
set -u
set -o pipefail

if [[ "$1" == "--branch" ]]; then
    BRT_BRANCH="$2"
    shift 2
fi

if [[ "$1" == "--brt_only" ]]; then
    BRT_ONLY="$2"
    shift 2
fi

cd /brainstorm_project/brainstorm
git fetch
git checkout -b "${BRT_BRANCH:-main}"
git submodule update --init --recursive

if [[ "$BRT_ONLY" = "True" ]]; then
    cd 3rdparty/tvm || exit
    mkdir -p build && cd build || exit
    cp ../../../cmake/config/tvm.cmake config.cmake
    cmake ..
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH &&
        ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 &&
        make install -j &&
        rm -f /usr/local/cuda/lib64/stubs/libcuda.so.1
    cd ../python && pip install .
fi

cd /brainstorm_project/brainstorm || exit
pip install -v --editable .

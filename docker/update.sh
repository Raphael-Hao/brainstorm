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

cd /root/brainstorm
git checkout "${BRT_BRANCH:-main}"
git pull --rebase
git submodule update --init --recursive

if (("$BRT_ONLY" == 0)); then
    cd 3rdparty/tvm || exit
    mkdir -p build && cd build || exit
    cp ../../../cmake/config/tvm.cmake config.cmake
    cmake ..
    make install -j
    cd ../python && pip install .
fi

cd /root/brainstorm || exit
pip install -v --editable .

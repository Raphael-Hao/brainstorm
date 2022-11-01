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

apt-get update && apt-get install -y python3 python3-dev python3-setuptools \
    gcc libtinfo-dev zlib1g-dev build-essential \
    cmake libedit-dev libxml2-dev llvm tmux

mkdir -p /root/brainstorm_project
cd /root/brainstorm_project
git clone git@github.com:Raphael-Hao/brainstorm.git \
    -b "${BRT_BRANCH:-main}" \
    --recursive --depth 1
cd brainstorm

pip3 install -r requirements.txt

cd 3rdparty/tvm || exit

mkdir -p build && cd build || exit
cp ../../../cmake/config/tvm.cmake config.cmake
cmake .. -DCMAKE_INSTALL_PREFIX=~/brainstorm_project/install
make install -j
cd ../python && pip install .

cd ../../../
pip3 install --editable .

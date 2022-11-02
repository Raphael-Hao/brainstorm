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

cd /root
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh &&
    bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b &&
    rm -f Miniconda3-py38_4.10.3-Linux-x86_64.sh

cd /root
git clone git@github.com:Raphael-Hao/brainstorm.git \
    -b "${BRT_BRANCH:-main}" \
    --recursive --depth 1
cd brainstorm

pip install --upgrade pip
conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt

cd 3rdparty/tvm || exit

mkdir -p build && cd build || exit
cp ../../../cmake/config/tvm.cmake config.cmake
cmake ..
make install -j
cd ../python && pip install .

cd /root/brainstorm || exit
pip install -v --editable .

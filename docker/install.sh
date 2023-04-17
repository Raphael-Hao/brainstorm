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
    bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b -p /opt/miniconda3 &&
    rm -f Miniconda3-py38_4.10.3-Linux-x86_64.sh

cd /root

echo 'export PATH=/opt/miniconda3/bin:$PATH' >>/etc/profile

BRT_DIR=/brainstorm_project/brainstorm
mkdir -p /brainstorm_project && cd /brainstorm_project

git clone https://Raphael-Hao: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@github.com/Raphael-Hao/brainstorm.git \
    -b "${BRT_BRANCH:-main}" \
    --recursive

cd "$BRT_DIR" || exit

pip config set global.index-url https://mirror.sjtu.edu.cn/pypi/web/simple
pip install --upgrade pip
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html/cu113
pip install -r requirements.txt
pip install -v --editable .

cd 3rdparty/tvm || exit
mkdir -p build && cd build || exit
cp ../../../cmake/config/tvm.cmake config.cmake
cmake ..
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH &&
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 &&
    make install -j &&
    rm -f /usr/local/cuda/lib64/stubs/libcuda.so.1
cd ../python && pip install .
cd "$BRT_DIR" || exit

cd 3rdparty/tutel && pip install -v -e .
cd "$BRT_DIR" || exit 1

cd benchmark/swin_moe && pip install -r requirements.txt
cd "$BRT_DIR" || exit 1

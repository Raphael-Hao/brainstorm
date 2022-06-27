#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

cd "$HOME" || exit

wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
wget https://github.com/Kitware/CMake/releases/download/v3.23.2/cmake-3.23.2-linux-x86_64.sh

chmod +x Miniconda3-py38_4.12.0-Linux-x86_64.sh
./Miniconda3-py38_4.12.0-Linux-x86_64.sh -b -p "$HOME"/miniconda3
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

cd "$HOME"/brainstorm_project/brainstorm || exit

# install trasformer
cd 3rdparty/transformers || exit
pip install --editable ./

# install tvm
cd ../tvm || exit
mkdir -p build && cd build || exit
cp ../../../cmake/config/tvm.cmake config.cmake
cmake .. -DCMAKE_INSTALL_PREFIX=~/brainstorm_project/install
make cpptest -j
./cpptest
make install -j

# build brt
cd ../../../
mkdir build && cd build || exit
cmake .. -DCMAKE_INSTALL_PREFIX=~/brainstorm_project/install -DCMAKE_CUDA_COMPILER="/usr/local/cuda/bin/nvcc"
make brt_torchscript -j
cd ../
pip install -r requirements.txt
pip install --editable .

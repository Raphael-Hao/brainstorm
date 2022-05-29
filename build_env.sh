#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

curl https://pyenv.run | bash
pyenv install miniconda3-3.8-4.10.3
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

cd 3rdparty/transformers || exit
pip install --editable ./

cd ../tvm || exit
mkdir -p build && cd build || exit
cp ../../../cmake/config/tvm.cmake config.cmake
cmake .. -DCMAKE_INSTALL_PREFIX=~/brainstorm_project/install
make cpptest -j
./cpptest
make install -j
cd ../../../
pip install -r requirements.txt
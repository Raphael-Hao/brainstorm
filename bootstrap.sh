#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

set -e
set -u
set -o pipefail

if [[ "$1" == "--branch" ]]; then
    BRT_BRANCH="$2"
    shift 2
fi

cd "$HOME"
rm -rf /opt/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh &&
    bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b -p /opt/miniconda3 &&
    rm -f Miniconda3-py38_4.10.3-Linux-x86_64.sh
echo 'export PATH=/opt/miniconda3/bin:$PATH' >>~/.bashrc
echo 'export PATH=/opt/miniconda3/bin:$PATH' >>~/.zshrc

export PATH=/opt/miniconda3/bin:$PATH

wget https://azcopyvnext.azureedge.net/release20221005/azcopy_linux_amd64_10.16.1.tar.gz -O azcopy.tar.gz
mkdir azcopy && tar -xzvf azcopy.tar.gz -C "$HOME/azcopy" --strip-components=1
mv azcopy/azcopy /usr/bin/azcopy && rm -rf azcopy.tar.gz azcopy

UBUNTU_DIST=$(lsb_release -sr)
wget https://packages.microsoft.com/config/ubuntu/"${UBUNTU_DIST}"/packages-microsoft-prod.deb
dpkg -i packages-microsoft-prod.deb && rm -f packages-microsoft-prod.deb
apt-get update
apt-get install -y \
    blobfuse gcc libtinfo-dev zlib1g-dev build-essential \
    libedit-dev libxml2-dev llvm wget git

git config --global user.name raphael
git config --global user.email raphaelhao@outlook.com

mkdir -p ~/brainstorm_project && cd ~/brainstorm_project
git clone git@github.com:Raphael-Hao/brainstorm.git \
    -b "${BRT_BRANCH:-main}" \
    --recursive
cd brainstorm

pip install --upgrade pip
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt

cd 3rdparty/tvm || exit

mkdir -p build && cd build || exit
cp ../../../cmake/config/tvm.cmake config.cmake
cmake ..
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH &&
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 &&
    make install -j &&
    rm -f /usr/local/cuda/lib64/stubs/libcuda.so.1
cd ../python && pip install .

cd ~/brainstorm_project/brainstorm || exit

pip install -v --editable .

cd ~/brainstorm_project/brainstorm/azure/blob || exit
bash mount.sh
bash download_image.sh

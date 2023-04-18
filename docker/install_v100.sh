#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /install.sh
# \brief:
# Author: raphael hao

if [[ "$1" == "--branch" ]]; then
    BRT_BRANCH="$2"
    shift 2
fi

cd /root

apt-get -y update && apt-get install -y \
    ssh openssh-server gcc libtinfo-dev zlib1g-dev build-essential \
    cmake libedit-dev libxml2-dev llvm tmux wget curl git vim zsh

wget https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh
sed '/exec zsh -l/d' ./install.sh >./install_wo_exec.sh
sh install_wo_exec.sh
rm install.sh install_wo_exec.sh

git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
git clone https://github.com/zsh-users/zsh-history-substring-search ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-history-substring-search
sed "s/plugins=(git)/plugins=(git extract zsh-autosuggestions zsh-history-substring-search zsh-syntax-highlighting)/g" "${HOME}/.zshrc" >"${HOME}/.tmp_zshrc" && mv "${HOME}/.tmp_zshrc" "${HOME}/.zshrc"



wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh &&
    bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b -p /opt/miniconda3 &&
    rm -f Miniconda3-py38_4.10.3-Linux-x86_64.sh

cd /root

echo 'export PATH=/opt/miniconda3/bin:$PATH' >>/root/.zshrc
echo 'export PATH=/opt/miniconda3/bin:$PATH' >>/root/.bashrc

source /root/.bashrc

pip install --upgrade pip
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113


BRT_DIR=/root/brainstorm_project/brainstorm
mkdir -p /root/brainstorm_project && cd /root/brainstorm_project

git clone https://Raphael-Hao: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@github.com/Raphael-Hao/brainstorm.git \
    -b "${BRT_BRANCH:-main}" \
    --recursive

cd "$BRT_DIR" || exit

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

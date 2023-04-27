#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /install.sh
# \brief:
# Author: raphael hao

if [[ "$1" == "--branch" ]]; then
    BRT_BRANCH="$2"
    shift 2
fi


is_root() {
    return "$(id -u)"
}

has_sudo() {
    local prompt
    prompt=$(sudo -nv 2>&1)
    if [ $? -eq 0 ]; then
        echo "has_sudo__pass_set"
    elif echo "$prompt" | grep -q '^sudo:'; then
        echo "has_sudo__needs_pass"
    else
        echo "no_sudo"
    fi
}

if is_root; then
    sudo_cmd=""
else
    HAS_SUDO=$(has_sudo)
    sudo_cmd="sudo"
    if [ "$HAS_SUDO" == "has_sudo__needs_pass" ]; then
        echo "You need to supply the password to use sudo."
        sudo -v
    elif [ "$HAS_SUDO" == "has_sudo__pass_set" ]; then
        echo "You already have sudo privileges."
    else
        echo "You need to have sudo privileges to run this script for some packages."
        exit 1
    fi
fi

cd "$HOME" || exit

$sudo_cmd apt-get -y update && $sudo_cmd apt-get install -y \
    gcc libtinfo-dev zlib1g-dev build-essential \
    cmake libedit-dev libxml2-dev llvm wget curl git vim

wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh &&
    bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b -p "$HOME"/miniconda3 &&
    rm -f Miniconda3-py38_4.10.3-Linux-x86_64.sh

# shellcheck disable=SC2016
{
    echo 'export PATH=/usr/local/cuda/bin:$HOME/miniconda3/bin:$PATH'
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH'
    echo 'export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH'
} >>"$HOME/.bashrc"

export PATH=/usr/local/cuda/bin:$HOME/miniconda3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH

export PATH="/usr/local/cuda/bin:$HOME"/miniconda3/bin:$PATH
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH

# shellcheck disable=SC1090,SC1091
source "$HOME/.bashrc"

pip install --upgrade pip
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

BRT_DIR="$HOME/brainstorm_project/brainstorm"
mkdir -p "$HOME/brainstorm_project" && cd "$HOME/brainstorm_project" || exit

export TORCH_CUDA_ARCH_LIST="7.0;7.2;7.5;8.0;8.6+PTX"

git clone https://github.com/Raphael-Hao/brainstorm.git \
    -b "${BRT_BRANCH:-main}" \
    --recursive

cd "$BRT_DIR" || exit

pip install -r requirements.txt
pip install -v --editable .

cd 3rdparty/tvm || exit
mkdir -p build && cd build || exit
cp ../../../cmake/config/tvm.cmake config.cmake
cmake ..
make install -j24
cd ../python && pip install .
cd "$BRT_DIR" || exit

cd 3rdparty/tutel && pip install -v -e .
cd "$BRT_DIR" || exit

cd 3rdparty/dynamic_routing && pip install -v -e .
cd "$BRT_DIR" || exit

cd benchmark/swin_moe && pip install -r requirements.txt
cd "$BRT_DIR" || exit

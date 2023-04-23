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
    ssh openssh-server gcc libtinfo-dev zlib1g-dev build-essential \
    cmake libedit-dev libxml2-dev llvm tmux wget curl git vim zsh

wget https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh
sed '/exec zsh -l/d' ./install.sh >./install_wo_exec.sh
sh install_wo_exec.sh
rm install.sh install_wo_exec.sh

git clone https://github.com/zsh-users/zsh-syntax-highlighting.git "${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting"
git clone https://github.com/zsh-users/zsh-autosuggestions "${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions"
git clone https://github.com/zsh-users/zsh-history-substring-search "${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-history-substring-search"
sed "s/plugins=(git)/plugins=(git extract zsh-autosuggestions zsh-history-substring-search zsh-syntax-highlighting)/g" "${HOME}/.zshrc" >"${HOME}/.tmp_zshrc" && mv "${HOME}/.tmp_zshrc" "${HOME}/.zshrc"

wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh &&
    bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b -p "$HOME"/miniconda3 &&
    rm -f Miniconda3-py38_4.10.3-Linux-x86_64.sh

# shellcheck disable=SC2016
echo 'export PATH="$HOME"/miniconda3/bin:$PATH' >>"$HOME/.zshrc"
# shellcheck disable=SC2016
echo 'export PATH="$HOME"/miniconda3/bin:$PATH' >>"$HOME/.bashrc"

# shellcheck disable=SC1090,SC1091
source "$HOME/.bashrc"

pip install --upgrade pip
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

BRT_DIR="$HOME/brainstorm_project/brainstorm"
mkdir -p "$HOME/brainstorm_project" && cd "$HOME/brainstorm_project" || exit

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
make install -j24
cd ../python && pip install .
cd "$BRT_DIR" || exit

cd 3rdparty/tutel && pip install -v -e .
cd "$BRT_DIR" || exit 1

cd benchmark/swin_moe && pip install -r requirements.txt
cd "$BRT_DIR" || exit 1

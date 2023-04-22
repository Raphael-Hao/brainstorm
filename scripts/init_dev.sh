#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /init.sh
# \brief:
# Author: raphael hao

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

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
    use_sudo=1
    sudo_cmd=""
else
    HAS_SUDO=$(has_sudo)
    sudo_cmd="sudo"
    if [ "$HAS_SUDO" == "has_sudo__needs_pass" ]; then
        use_sudo=1
        echo "You need to supply the password to use sudo."
        sudo -v
    elif [ "$HAS_SUDO" == "has_sudo__pass_set" ]; then
        use_sudo=1
    else
        echo "You need to have sudo privileges to run this script for some packages."
        use_sudo=0
        exit 1
    fi
fi

git config --global user.name raphael
git config --global user.email raphaelhao@outlook.com

curl -sL https://aka.ms/InstallAzureCLIDeb | $sudo_cmd bash

wget -O azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux -O azcopy.tar.gz
mkdir -p azcopy && tar -xzvf azcopy.tar.gz -C "azcopy" --strip-components=1
$sudo_cmd mv azcopy/azcopy /usr/bin/azcopy && rm -rf azcopy.tar.gz azcopy

mkdir -p ../.vscode
cp ./vscode/c_cpp_properties.json ../.vscode/c_cpp_properties.json
cp ./vscode/settings_workspace.json ../.vscode/settings.json
mkdir -p ~/.vscode-server/data/Machine/
cp ./vscode/settings_global.json ~/.vscode-server/data/Machine/settings.json
bash prepare_data.sh

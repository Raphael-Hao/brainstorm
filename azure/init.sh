#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /init.sh
# \brief:
# Author: raphael hao

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
BRT_DIR=$(cd "${script_dir}/../" && pwd)

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

wget https://azcopyvnext.azureedge.net/release20221005/azcopy_linux_amd64_10.16.1.tar.gz -O azcopy.tar.gz
mkdir -p azcopy && tar -xzvf azcopy.tar.gz -C "azcopy" --strip-components=1
$sudo_cmd mv azcopy/azcopy /usr/bin/azcopy && rm -rf azcopy.tar.gz azcopy

UBUNTU_DIST=$(lsb_release -sr)
wget https://packages.microsoft.com/config/ubuntu/"${UBUNTU_DIST}"/packages-microsoft-prod.deb
$sudo_cmd dpkg -i packages-microsoft-prod.deb && rm -f packages-microsoft-prod.deb
$sudo_cmd apt-get update && $sudo_cmd apt-get install -y blobfuse dotnet-sdk-6.0

mkdir -p ../.vscode
cp -r ./vscode/* ../.vscode/
bash blob/mount.sh
bash blob/prepare_dataset_ckpt.sh
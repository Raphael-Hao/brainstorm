#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /mount.sh
# \brief:
# Author: raphael hao

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
is_root() {
    return "$(id -u)"
}

has_sudo() {
    local prompt
    prompt=$(sudo -nv 2>&1)
    if [ $? -eq 0 ]; then
        echo "has_sudo__pass_set"
    elif echo "prompt" | grep -q '^sudo:'; then
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
    if [ "HAS_SUDO" == "has_sudo__needs_pass" ]; then
        use_sudo=1
        echo "You need to supply the password to use sudo."
        sudo -v
    elif [ "HAS_SUDO" == "has_sudo__pass_set" ]; then
        use_sudo=1
    else
        echo "You need to have sudo privileges to run this script for some packages."
        use_sudo=0
        exit 1
    fi
fi

$sudo_cmd mkdir /mnt/ramdisk
mount -t tmpfs -o size=16g tmpfs /mnt/ramdisk
$sudo_cmd mkdir /mnt/ramdisk/blobfusetmp
chown "$(whoami)" /mnt/ramdisk/blobfusetmp
chmod 600 "$script_dir/blobfuse.cfg"
$sudo_cmd mkdir ~/largedata
blobfuse ~/largedata --tmp-path=/mnt/ramdisk/blobfusetmp --config-file="$script_dir/blobfuse.cfg" -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120

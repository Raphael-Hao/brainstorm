#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /az_show.sh
# \brief:
# Author: raphael hao

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$1" == "--registry" ]]; then
    REGISTRY="$2"
    shift 2
else
    echo "Please specify the image to upload"
    exit 1
fi
# shellcheck disable=SC1091
source "$SCRIPT_DIR/registry.config"

# view containers in the repository:
az acr repository list --name "$REGISTRY" -o tsv

# get details about a repository:
az acr repository show -n "$REGISTRY" --repository "raphael/brt"

# get details about a container image version:
az acr repository show -n "$REGISTRY" --image "raphael/brt:main"

# delete the whole repository:
az acr repository delete -n "$REGISTRY" --repository "raphael/brt"

# delete a image:
az acr repository delete -n "$REGISTRY" --image "raphael/brt:main"

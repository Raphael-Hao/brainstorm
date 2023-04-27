#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /az_upload.sh
# \brief:
# Author: raphael hao

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$1" == "--image" ]]; then
    DOCKER_IMG_SPEC="$2"
    shift 2
else
    echo "Please specify the image to upload"
    exit 1
fi

if [[ "$1" == "--registry" ]]; then
    REGISTRY="$2"
    shift 2
else
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/registry.config"
    shift 1
fi

az acr login --name "$REGISTRY"
echo "Uploading Docker image to Azure GCR ..."
docker tag "$DOCKER_IMG_SPEC" "$REGISTRY.azurecr.io/raphael/$DOCKER_IMG_SPEC"
docker push "$REGISTRY.azurecr.io/raphael/$DOCKER_IMG_SPEC"
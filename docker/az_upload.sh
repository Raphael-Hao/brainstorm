#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /az_upload.sh
# \brief:
# Author: raphael hao
if [[ "$1" == "--image" ]]; then
    DOCKER_IMG_SPEC="$2"
    shift 2
else
    echo "Please specify the image to upload"
    exit 1
fi

az acr login --name gcrmembers
echo "Uploading Docker image to Azure GCR ..."
docker tag "$DOCKER_IMG_SPEC" "gcrmembers.azurecr.io/v-weihaocui/$DOCKER_IMG_SPEC"
docker push "gcrmembers.azurecr.io/v-weihaocui/$DOCKER_IMG_SPEC"
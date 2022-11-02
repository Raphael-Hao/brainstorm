#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /build.sh
# \brief:
# Author: raphael hao
# set -e
# set -u
# set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONTAINER_TYPE=$(echo "$1" | tr '[:upper:]' '[:lower:]')
shift 1

DOCKERFILE_PATH="${SCRIPT_DIR}/Dockerfile.${CONTAINER_TYPE}"

if [[ "$1" == "--tag" ]]; then
    DOCKER_IMAGE_TAG="$2"
    echo "Using custom Docker tag: ${DOCKER_IMAGE_TAG}"
    shift 2
fi

if [[ "$1" == "--context-path" ]]; then
    DOCKER_CONTEXT_PATH="$2"
    echo "Using custom context path: ${DOCKER_CONTEXT_PATH}"
    shift 2
else
    DOCKER_CONTEXT_PATH=$(dirname "${DOCKERFILE_PATH}")
    echo "Using default context path: ${DOCKER_CONTEXT_PATH}"
fi

if [[ "$1" == "--branch" ]]; then
    BRT_BRANCH="$2"
    echo "Using custom branch: ${BRT_BRANCH}"
    shift 2
else
    BRT_BRANCH="main"
    echo "Using default branch: ${BRT_BRANCH}"
fi

if [[ ! -f "${DOCKERFILE_PATH}" ]]; then
    echo "Invalid Dockerfile path: \"${DOCKERFILE_PATH}\""
    exit 1
fi

SSH_KEY_FILE="${DOCKER_CONTEXT_PATH}/id_ed25519"

if [[ ! -f "${SSH_KEY_FILE}" ]]; then
    echo "Invalid SSH Key file path:\"${SSH_KEY_FILE}\""
    echo "Currently, a valid SSH Key file is required to clone the BRT repo."
    exit 1
fi

COMMAND=("$@")

BUILD_TAG="${BUILD_TAG:-brt}"
DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-latest}"

DOCKER_IMG_NAME="${BUILD_TAG}.${CONTAINER_TYPE}"

DOCKER_IMG_NAME=$(echo "${DOCKER_IMG_NAME}" | sed -e 's/=/_/g' -e 's/,/-/g')

DOCKER_IMG_NAME=$(echo "${DOCKER_IMG_NAME}" | tr '[:upper:]' '[:lower:]')

DOCKER_IMG_SPEC="${DOCKER_IMG_NAME}:${DOCKER_IMAGE_TAG}"

echo "Building Docker image: ${DOCKER_IMG_SPEC}..."
echo "Docker image context path: ${DOCKER_CONTEXT_PATH}"
echo "Docker image Dockerfile path: ${DOCKERFILE_PATH}"
echo "Using Branch of Brainstorm: ${BRT_BRANCH}"
echo "Using SSH Key file: ${SSH_KEY_FILE} for accessing private git repos"

docker build -t "$DOCKER_IMG_SPEC" \
    -f "$DOCKERFILE_PATH" \
    --build-arg SSH_KEY_FILE="$SSH_KEY_PATH" \
    --build-arg BRT_BRANCH="$BRT_BRANCH" \
    "$DOCKER_CONTEXT_PATH"
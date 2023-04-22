#!/bin/bash

CACHE_DATA_DIR=/home/whcui/brainstorm_artifact

docker run \
    -d --name brt \
    --mount type=bind,source=$CACHE_DATA_DIR/ckpt,target=/root/brainstorm_project/brainstorm/.cache/ckpt \
    --mount type=bind,source=$CACHE_DATA_DIR/dataset,target=/root/brainstorm_project/brainstorm/.cache/dataset \
    --mount type=bind,source=$CACHE_DATA_DIR/kernel_db.sqlite,target=/root/brainstorm_project/brainstorm/.cache/kernel_db.sqlite \
    --mount type=bind,source=$CACHE_DATA_DIR/results,target=/root/brainstorm_project/brainstorm/.cache/results \
    ghcr.io/raphael-hao/brt:latest

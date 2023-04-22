#!/bin/bash

if [[ "$1" == "--mount" ]]; then
    CACHE_DATA_DIR=$2
    shift 2
    echo "start brt docker with mount : $CACHE_DATA_DIR"
    docker run \
        -d --name brt \
        --mount type=bind,source=$CACHE_DATA_DIR/ckpt,target=/root/brainstorm_project/brainstorm/.cache/ckpt \
        --mount type=bind,source=$CACHE_DATA_DIR/dataset,target=/root/brainstorm_project/brainstorm/.cache/dataset \
        --mount type=bind,source=$CACHE_DATA_DIR/kernel_db.sqlite,target=/root/brainstorm_project/brainstorm/.cache/kernel_db.sqlite \
        --mount type=bind,source=$CACHE_DATA_DIR/results,target=/root/brainstorm_project/brainstorm/.cache/results \
        ghcr.io/raphael-hao/brt:latest
else
    echo "start brt docker without mount"
    docker run \
        -d --name brt \
        ghcr.io/raphael-hao/brt:latest
fi

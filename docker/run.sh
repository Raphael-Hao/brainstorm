#!/bin/bash

docker run \
    -d --name brt \
    --mount type=bind,source=/home/whcui/ckpt,target=/brainstorm_project/brainstorm/.cache/ckpt \
    --mount type=bind,source=/home/whcui/dataset,target=/brainstorm_project/brainstorm/.cache/dataset \
     ghcr.io/raphael-hao/brt:latest
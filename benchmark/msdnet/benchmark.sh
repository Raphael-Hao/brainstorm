#!/usr/bin/env bash

export BRT_CAPTURE_STATS=True
export BRT_CAPTURED_FABRIC_TYPE=dispatch,combine
export BRT_CACHE_PATH="${HOME}/brainstorm_project/brainstorm/.cache"

msdnet_path="${HOME}/brainstorm_project/brainstorm/benchmark/msdnet"
pth_path="${msdnet_path}/MSDNet.pth"

function run_benchmark() {
    echo "############################################################"
    echo "####################### Exit Portion #######################"
    echo "################## ${exit_portion} #################"
    echo "############################################################"
    
    echo ${thresholds}
    python3 main.py \
        --thresholds ${thresholds} \
        --data-root ~/dataset/imagenet/ \
        --data ImageNet \
        --save ${msdnet_path}/saveresult \
        --arch msdnet \
        --batch-size 256 \
        --epochs 90 \
        --nBlocks 5 \
        --stepmode even \
        --step 4 \
        --base 4 \
        --nChannels 32 \
        --growthRate 16 \
        --grFactor 1-2-4-4 \
        --bnFactor 1-2-4-4 \
        --evalmode threshold \
        --evaluate-from ${pth_path} \
        --benchmark all_opt \
        --use-valid \
        --gpu 0,1,2,3 -j 16 \
        --init_routers \
        --parallel
    echo "############################################################"
    echo "##################### Exit Portion Ends ####################"
    echo "################## ${exit_portion} #################"
    echo "############################################################"
    echo
}

thresholds="1000 1000 1000 1000"
exit_portion="0.0, 0.0, 0.0, 0.0, 1.0"
run_benchmark

thresholds="1000 1000 1000 0.83451331"
exit_portion="0.0, 0.0, 0.0, 0.4, 0.6"
run_benchmark

thresholds="1000 1000 0.90728849 0.57961094"
exit_portion="0.0, 0.0, 0.3, 0.3, 0.4"
run_benchmark

thresholds="0.96616900 0.95113075 0.80969042 0.45410264"
exit_portion="0.1, 0.1, 0.2, 0.3, 0.3"
run_benchmark

thresholds="0.44246864 0.39881980 0.19329087 -1"
exit_portion="0.5, 0.2, 0.2, 0.1, 0.0"
run_benchmark

thresholds="0.44246849 0.26682281 -1 -1"
exit_portion="0.5, 0.3, 0.2, 0.0, 0.0"
run_benchmark

thresholds="0.44246858 -1 -1 -1"
exit_portion="0.5, 0.5, 0.0, 0.0, 0.0"
run_benchmark

thresholds="-1 -1 -1 -1"
exit_portion="1.0, 0.0, 0.0, 0.0, 0.0"
run_benchmark



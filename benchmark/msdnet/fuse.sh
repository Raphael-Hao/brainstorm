export BRT_CACHE_PATH=$HOME/brainstorm_project/brainstorm/.cache
export BRT_CAPTURE_STATS=True
export BRT_CAPTURED_FABRIC_TYPE=dispatch,combine
rm -r saveresult/
rm msdnet.json
    ##0.5 0.5 0 0 0

CUDA_LAUNCH_BLOCKING=1 python3 main.py --data-root  ~/dataset/imagenet/ --data ImageNet --save /home/v-louyang/brainstorm_project/brainstorm/benchmark/msdnet/saveresult                 --arch msdnet --batch-size 256 --epochs 90 --nBlocks 5                 --stepmode even --step 4 --base 4 --nChannels 32 --growthRate 16                 --grFactor 1-2-4-4 --bnFactor 1-2-4-4                 --evalmode threshold  --evaluate-from /home/v-louyang/brainstorm_project/brainstorm/benchmark/msdnet/pth/msdnet-step=4-block=5.pth.tar  --benchmark vfuse             --use-valid --gpu 0,1,2,3 -j 16 --init_routers --parallel
##[0 0 0 0 0 1]
    # [1000000,100000,1000000,100000]

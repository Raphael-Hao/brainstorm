export BRT_CACHE_PATH=$HOME/brainstorm_project/brainstorm/.cache
export BRT_CAPTURE_STATS=True
export BRT_CAPTURED_FABRIC_TYPE=dispatch,combine
rm -r saveresult/
python3 main.py  --data-root  ~/imagenet/all --data ImageNet --save /home/v-weihaocui/brainstorm_project/brainstorm/benchmark/msdnet/saveresult                 --arch msdnet --batch-size 256 --epochs 90 --nBlocks 5                 --stepmode even --step 4 --base 4 --nChannels 32 --growthRate 16                 --grFactor 1-2-4-4 --bnFactor 1-2-4-4                 --evalmode threshold  --evaluate-from /home/v-weihaocui/pth/msdnet-step=4-block=5.pth.tar  --benchmark all_opt             --use-valid --gpu 0,1,2,3 -j 16 --init_routers --parallel

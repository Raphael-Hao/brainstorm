rm -r saveresult/
python3 main.py --data-root  /home/disk1/cwh/data/imagenet --data ImageNet --save /home/yichuanjiaoda/brainstorm_project/brainstorm/benchmark/msdnet/saveresult                 --arch msdnet --batch-size 256 --epochs 90 --nBlocks 5                 --stepmode even --step 4 --base 4 --nChannels 32 --growthRate 16                 --grFactor 1-2-4-4 --bnFactor 1-2-4-4                 --evalmode threshold  --evaluate-from /home/yichuanjiaoda/model/checkpoint/models/step=4/msdnet-step=4-block=5.pth.tar  --benchmark constant_propagation          --use-valid --gpu 0,1,2,3 -j 16 --init_routers --parallel
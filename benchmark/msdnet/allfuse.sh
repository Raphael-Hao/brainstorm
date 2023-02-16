rm -r saveresult/

export BRT_CAPTURE_STATS=True
export BRT_CAPTURED_FABRIC_TYPE=dispatch,combine
export BRT_CACHE_PATH=$HOME/brainstorm_project/brainstorm/.cache

rm msdnet.json

## 0.5 0.5 0 0 0
python3 main.py \
    --thresholds  0.44246858 -1 -1 -1 \
    --data-root  ~/dataset/imagenet/ \
    --data ImageNet \
    --save ~/brainstorm_project/brainstorm/benchmark/msdnet/saveresult \
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
    --evaluate-from ~/brainstorm_project/brainstorm/benchmark/msdnet/MSDNet.pth \
    --benchmark all_opt \
    --use-valid \
    --gpu 0,1,2,3 -j 16 \
    --init_routers \
    --parallel

## 0.5 0.5 0 0 0
python3 main.py \
    --thresholds  0.44246858 -1 -1 -1 \
    --data-root  ~/dataset/imagenet/ \
    --data ImageNet \
    --save ~/brainstorm_project/brainstorm/benchmark/msdnet/saveresult \
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
    --evaluate-from ~/brainstorm_project/brainstorm/benchmark/msdnet/msdnet-step=4-block=5.pth.tar \
    --benchmark hfuse \
    --use-valid \
    --gpu 0,1,2,3 -j 16 \
    --init_routers \
    --parallel

## [0 0 0 0 0 1]
python3 main.py \
    --thresholds 1000000 100000 1000000 100000 \
    --data-root  ~/dataset/imagenet/ \
    --data ImageNet \
    --save ~/brainstorm_project/brainstorm/benchmark/msdnet/saveresult \
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
    --evaluate-from ~/brainstorm_project/brainstorm/benchmark/msdnet/msdnet-step=4-block=5.pth.tar \
    --benchmark hfuse \
    --use-valid \
    --gpu 0,1,2,3 -j 16 \
    --init_routers \
    --parallel

## [0 0 0 0.4 0.6]
python3 main.py \
    --data-root ~/dataset/imagenet/ \
    --thresholds 100000000.00000000 100000000.00000000 100000000.00000000 0.83451331 \
    --data ImageNet \
    --save ~/brainstorm_project/brainstorm/benchmark/msdnet/saveresult \
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
    --evaluate-from ~/brainstorm_project/brainstorm/benchmark/msdnet/msdnet-step=4-block=5.pth.tar \
    --benchmark hfuse \
    --use-valid \
    --gpu 0,1,2,3 -j 16 \
    --init_routers \
    --parallel

## [0,0,0.3,0.3,0.4]
python3 main.py \
    --data-root ~/dataset/imagenet/ \
    --thresholds 100000000.00000000 100000000.00000000 0.90728849 0.57961094 \
    --data ImageNet \
    --save ~/brainstorm_project/brainstorm/benchmark/msdnet/saveresult \
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
    --evaluate-from ~/brainstorm_project/brainstorm/benchmark/msdnet/msdnet-step=4-block=5.pth.tar \
    --benchmark hfuse \
    --use-valid \
    --gpu 0,1,2,3 -j 16 \
    --init_routers \
    --parallel

## [0.1,0.1,0.2,0.3,0.3]
python3 main.py \
    --data-root ~/dataset/imagenet/ \
    --thresholds 0.96616900 0.95113075 0.80969042 0.45410264 \
    --data ImageNet \
    --save ~/brainstorm_project/brainstorm/benchmark/msdnet/saveresult \
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
    --evaluate-from ~/brainstorm_project/brainstorm/benchmark/msdnet/msdnet-step=4-block=5.pth.tar \
    --benchmark hfuse \
    --use-valid \
    --gpu 0,1,2,3 -j 16 \
    --init_routers \
    --parallel

## 0.6 0.1 0.1 0.1 0.1
python3 main.py \
    --data-root ~/dataset/imagenet/ \
    --thresholds      0.34071380 0.47392023 0.37517136 0.22579938 \
    --data ImageNet \
    --save ~/brainstorm_project/brainstorm/benchmark/msdnet/saveresult \
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
    --evaluate-from ~/brainstorm_project/brainstorm/benchmark/msdnet/msdnet-step=4-block=5.pth.tar \
    --benchmark hfuse \
    --use-valid \
    --gpu 0,1,2,3 -j 16 \
    --init_routers \
    --parallel

## 0.5 0.2 0.2 0.1
python3 main.py \
    --data-root  ~/dataset/imagenet/ \
    --thresholds     0.44246864 0.39881980 0.19329087 -1 \
    --data ImageNet \
    --save ~/brainstorm_project/brainstorm/benchmark/msdnet/saveresult \
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
    --evaluate-from ~/brainstorm_project/brainstorm/benchmark/msdnet/msdnet-step=4-block=5.pth.tar \
    --benchmark hfuse \
    --use-valid \
    --gpu 0,1,2,3 -j 16 \
    --init_routers \
    --parallel

## 0.5 0.3 0.2 0 0
python3 main.py \
    --data-root ~/dataset/imagenet/ \
    --thresholds 0.44246849  0.26682281 -1 -1 \
    --data ImageNet \
    --save ~/brainstorm_project/brainstorm/benchmark/msdnet/saveresult \
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
    --evaluate-from ~/brainstorm_project/brainstorm/benchmark/msdnet/msdnet-step=4-block=5.pth.tar \
    --benchmark hfuse \
    --use-valid \
    --gpu 0,1,2,3 -j 16 \
    --init_routers \
    --parallel
python3 main.py \
    --data-root ~/dataset/imagenet/ \
    --thresholds -1 -1 -1 -1 \
    --data ImageNet \
    --save ~/brainstorm_project/brainstorm/benchmark/msdnet/saveresult \
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
    --evaluate-from ~/brainstorm_project/brainstorm/benchmark/msdnet/msdnet-step=4-block=5.pth.tar \
    --benchmark hfuse \
    --use-valid \
    --gpu 0,1,2,3 -j 16 \
    --init_routers \
    --parallel

export BRT_CACHE_PATH=$HOME/brainstorm_project/brainstorm/.cache
export BRT_CAPTURE_STATS=True
export BRT_CAPTURED_FABRIC_TYPE=dispatch,combine

rm -r saveresult/
rm msdnet.json

## 0.5 0.5 0 0 0
python3 main.py \
    --thresholds 0.44246858 -1 -1 -1 \
    --data-root  ~/dataset/imagenet/ \
    --data ImageNet \
    --save ~/brainstorm_project/brainstorm/benchmark/msdnet/saveresult \
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
    --evaluate-from ~/brainstorm_project/brainstorm/benchmark/msdnet/msdnet-step=4-block=5.pth.tar \
    --benchmark vfuse \
    --use-valid \
    --gpu 0,1,2,3 -j 16 \
    --init_routers \
    --parallel

## [0 0 0 0 0 1]
python3 main.py \
    --thresholds 1000000 100000 1000000 100000 \
    --data-root  ~/dataset/imagenet/ \
    --data ImageNet \
    --save ~/brainstorm_project/brainstorm/benchmark/msdnet/saveresult \
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
    --evaluate-from ~/brainstorm_project/brainstorm/benchmark/msdnet/msdnet-step=4-block=5.pth.tar \
    --benchmark vfuse \
    --use-valid \
    --gpu 0,1,2,3 -j 16 \
    --init_routers \
    --parallel

## [0 0 0 0.4 0.6]
python3 main.py \
    --data-root ~/dataset/imagenet/ \
    --thresholds 100000000.00000000 100000000.00000000 100000000.00000000 0.83451331 \
    --data ImageNet \
    --save ~/brainstorm_project/brainstorm/benchmark/msdnet/saveresult \
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
    --evaluate-from ~/brainstorm_project/brainstorm/benchmark/msdnet/msdnet-step=4-block=5.pth.tar \
    --benchmark vfuse \
    --use-valid \
    --gpu 0,1,2,3 -j 16 \
    --init_routers \
    --parallel

## [0,0,0.3,0.3,0.4]
python3 main.py \
    --data-root ~/dataset/imagenet/ \
    --thresholds 100000000.00000000 100000000.00000000 0.90728849 0.57961094 \
    --data ImageNet \
    --save ~/brainstorm_project/brainstorm/benchmark/msdnet/saveresult \
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
    --evaluate-from ~/brainstorm_project/brainstorm/benchmark/msdnet/msdnet-step=4-block=5.pth.tar \
    --benchmark vfuse \
    --use-valid \
    --gpu 0,1,2,3 -j 16 \
    --init_routers \
    --parallel

## [0.1,0.1,0.2,0.3,0.3]
python3 main.py \
    --data-root ~/dataset/imagenet/ \
    --thresholds 0.96616900 0.95113075 0.80969042 0.45410264 \
    --data ImageNet \
    --save ~/brainstorm_project/brainstorm/benchmark/msdnet/saveresult \
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
    --evaluate-from ~/brainstorm_project/brainstorm/benchmark/msdnet/msdnet-step=4-block=5.pth.tar \
    --benchmark vfuse \
    --use-valid \
    --gpu 0,1,2,3 -j 16 \
    --init_routers \
    --parallel

## 0.6 0.1 0.1 0.1 0.1
python3 main.py \
    --data-root ~/dataset/imagenet/ \
    --thresholds      0.34071380 0.47392023 0.37517136 0.22579938 \
    --data ImageNet \
    --save ~/brainstorm_project/brainstorm/benchmark/msdnet/saveresult \
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
    --evaluate-from ~/brainstorm_project/brainstorm/benchmark/msdnet/msdnet-step=4-block=5.pth.tar \
    --benchmark vfuse \
    --use-valid \
    --gpu 0,1,2,3 -j 16 \
    --init_routers \
    --parallel

## 0.5 0.2 0.2 0.1
python3 main.py \
    --data-root  ~/dataset/imagenet/ \
    --thresholds     0.44246864 0.39881980 0.19329087 -1 \
    --data ImageNet \
    --save ~/brainstorm_project/brainstorm/benchmark/msdnet/saveresult \
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
    --evaluate-from ~/brainstorm_project/brainstorm/benchmark/msdnet/msdnet-step=4-block=5.pth.tar \
    --benchmark vfuse \
    --use-valid \
    --gpu 0,1,2,3 -j 16 \
    --init_routers \
    --parallel

## 0.5 0.3 0.2 0 0
python3 main.py \
    --data-root ~/dataset/imagenet/ \
    --thresholds 0.44246849  0.26682281 -1 -1 \
    --data ImageNet \
    --save ~/brainstorm_project/brainstorm/benchmark/msdnet/saveresult \
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
    --evaluate-from ~/brainstorm_project/brainstorm/benchmark/msdnet/msdnet-step=4-block=5.pth.tar \
    --benchmark vfuse \
    --use-valid \
    --gpu 0,1,2,3 -j 16 \
    --init_routers \
    --parallel

python3 main.py \
    --data-root ~/dataset/imagenet/ \
    --thresholds -1 -1 -1 -1 \
    --data ImageNet \
    --save ~/brainstorm_project/brainstorm/benchmark/msdnet/saveresult \
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
    --evaluate-from ~/brainstorm_project/brainstorm/benchmark/msdnet/msdnet-step=4-block=5.pth.tar \
    --benchmark vfuse \
    --use-valid \
    --gpu 0,1,2,3 -j 16 \
    --init_routers \
    --parallel

python3 main.py \
    --thresholds 0.44246858 -1 -1 -1 \
    --data-root  ~/dataset/imagenet/ \
    --data ImageNet \
    --save ~/brainstorm_project/brainstorm/benchmark/msdnet/saveresult \
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
    --evaluate-from ~/brainstorm_project/brainstorm/benchmark/msdnet/msdnet-step=4-block=5.pth.tar \
    --benchmark vfuse_trans \
    --use-valid \
    --gpu 0,1,2,3 -j 16 \
    --init_routers \
    --parallel

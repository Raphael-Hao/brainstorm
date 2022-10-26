sudo env BRT_CACHE_PATH=$BRT_CACHE_PATH PATH=$PATH ncu --target-processes all -k regex:Conv2d -o nsight_rcan_conv2d python benchmark.py

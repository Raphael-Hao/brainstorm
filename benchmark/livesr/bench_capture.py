import timeit
import logging
import sys
from pathlib import Path

import torch
from torch.utils.benchmark import Timer

from brt.router import switch_router_mode
from brt.runtime.benchmark import profile

from archs.livesr import LiveSR
from archs.vfused_livesr import vFusedLiveSR
from archs.hfused_livesr import hFusedLiveSR
from dataset import get_dataloader

DEFAULT_DATASET = Path(__file__).parent / "dataset/cam1/LQ"
ALL_SUBNET_BS = [
    [6, 7, 12, 27, 8, 8, 8, 12, 12, 4],
    # [21, 8, 16, 11, 20, 8, 7, 15],
    # [9, 32, 18, 16, 18, 7],
    # [19, 25, 18, 36],
]
ALL_NUM_CHANNELS = [
    8,
    # 12,
    # 16,
    # 20,
]
SUBNET_NUM_BLOCK = 80

logger = logging.getLogger("benchmark")
logger.setLevel(logging.DEBUG)
logger_handler = logging.StreamHandler(stream=sys.stdout)
logger_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logger.addHandler(logger_handler)

bench_module_list = [
    "LiveSR",
    # "vFusedLiveSR",
    # "hFusedLiveSR",
]

logger.info(f"{SUBNET_NUM_BLOCK = }")
logger.info(f"{ALL_SUBNET_BS = }")
logger.info(f"{ALL_NUM_CHANNELS = }")

module_dict = {}
logger.info("Start")


def build_module(subnet_bs, num_feature):
    global module_dict
    num_subnets = len(subnet_bs)
    try:
        if "LiveSR" in bench_module_list:
            module_dict[("LiveSR", num_feature, num_subnets)] = LiveSR(
                num_subnets, SUBNET_NUM_BLOCK, num_feature
            ).cuda()
            logger.info(f"LiveSR {num_feature} {num_subnets} builded")
        if "vFusedLiveSR" in bench_module_list:
            module_dict[("vFusedLiveSR", num_feature, num_subnets)] = vFusedLiveSR(
                module_dict[("LiveSR", num_feature, num_subnets)], subnet_bs
            )
            logger.info(f"vFusedLiveSR {num_feature} {num_subnets} builded")
        if "hFusedLiveSR" in bench_module_list:
            module_dict[("hFusedLiveSR", num_feature, num_subnets)] = hFusedLiveSR(
                module_dict[("LiveSR", num_feature, num_subnets)], subnet_bs
            )
            logger.info(f"hFusedLiveSR {num_feature} {num_subnets} builded")
    except Exception as e:
        logger.warning(str(e))


for subnet_bs in ALL_SUBNET_BS:
    build_module(subnet_bs, ALL_NUM_CHANNELS[0])
for num_channels in ALL_NUM_CHANNELS[1:]:
    build_module(ALL_SUBNET_BS[0], num_channels)

dataloader = get_dataloader(DEFAULT_DATASET)

for input_tensor in dataloader:
    break

print(module_dict.keys())
print(input_tensor.shape, input_tensor.device, input_tensor.dtype)

for n in [10, 100]:
    logger.info(f"* Start timeit: Run {n} times")
    for (module_type, num_feature, num_subnets), model in module_dict.items():
        x = input_tensor
        switch_router_mode(model, True)
        time = (
            Timer(
                f"model(x)",
                # f"model.classifier(x)",
                setup="import torch; torch.cuda.synchronize()",
                # setup="import torch; torch.cuda.synchronize(); torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False",
                globals={"model": model, "x": x},
            )
            .timeit(n)
            .mean
            * 10e6
        )
        logger.info(f"{module_type}, {num_feature}, {num_subnets}:\t\t {time} us/run")
        switch_router_mode(model, False)
        time = (
            Timer(
                f"model(x)",
                # f"model.classifier(x)",
                setup="import torch; torch.cuda.synchronize()",
                # setup="import torch; torch.cuda.synchronize(); torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False",
                globals={"model": model, "x": x},
            )
            .timeit(n)
            .mean
            * 10e6
        )
        logger.info(f"{module_type}, {num_feature}, {num_subnets}:\t\t {time} us/run (no capture)")


input("Press any key to start profiling")
for (module_type, num_feature, num_subnets), model in module_dict.items():
    logger.info(f"Profiling {module_type}")
    model = module_dict[(module_type, num_feature, num_subnets)]
    x = input_tensor
    switch_router_mode(model, True)
    profile(lambda: model(x))
    switch_router_mode(model, False)
    profile(lambda: model(x))

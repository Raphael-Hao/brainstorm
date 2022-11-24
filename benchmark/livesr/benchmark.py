import timeit
import logging
import sys
from pathlib import Path

import torch
from torch.utils.benchmark import Timer

from brt.runtime.benchmark import profile

from archs.livesr import LiveSR
from archs.vfused_livesr import vFusedLiveSR
from archs.hfused_livesr import hFusedLiveSR
from dataset import get_dataloader

DEFAULT_DATASET = Path(__file__).parent / "dataset/cam1/LQ"
SUBNET_BATCH_SIZE = [6, 7, 12, 27, 8, 8, 8, 12, 12, 4]
SUBNET_NUM_BLOCK = 80
NUM_FEATURE = 8

logger = logging.getLogger("benchmark")
logger.setLevel(logging.DEBUG)
logger_handler = logging.StreamHandler(stream=sys.stdout)
logger_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logger.addHandler(logger_handler)

bench_module_list = [
    "LiveSR",
    "vFusedLiveSR",
    "hFusedLiveSR",
]

module_dict = {}
logger.info("Start")
if "LiveSR" in bench_module_list:
    module_dict["LiveSR"] = LiveSR(10, SUBNET_NUM_BLOCK, NUM_FEATURE).cuda()
    logger.info("LiveSR builded")
if "vFusedLiveSR" in bench_module_list:
    module_dict["vFusedLiveSR"] = vFusedLiveSR(module_dict["LiveSR"], SUBNET_BATCH_SIZE)
    logger.info("vFusedLiveSR builded")
if "hFusedLiveSR" in bench_module_list:
    module_dict["hFusedLiveSR"] = hFusedLiveSR(module_dict["LiveSR"], SUBNET_BATCH_SIZE)
    logger.info("hFusedLiveSR builded")


dataloader = get_dataloader(DEFAULT_DATASET)

for input_tensor in dataloader:
    break

for n in [10, 100]:
    logger.info(f"* Start timeit: Run {n} times")
    for module_type in bench_module_list:
        model = module_dict[module_type]
        x = input_tensor
        # time = (
        #     Timer(
        #         f"model(x)",
        #         setup="from __main__ import model, x; import torch; torch.cuda.synchronize(); torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False",
        #     )
        #     .timeit(n)
        #     .mean
        #     * 10e6
        # )
        # logger.info(f"{module_type}:\t\t {time}s in {n} runs ({time/n}s/run) (torch tf32)")
        time = (
            Timer(
                f"model(x)",
                setup="from __main__ import model, x; import torch; torch.cuda.synchronize()",
            )
            .timeit(n)
            .mean
            * 10e6
        )
        logger.info(f"{module_type}:\t\t {time} us/run")

input("Press any key to start profiling")
for module_type in bench_module_list:
    logger.info(f"Profiling {module_type}")
    model = module_dict[module_type]
    x = input_tensor
    profile(lambda: model(x))

import timeit
import logging
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.benchmark import Timer

from brt.runtime.benchmark import profile, CUDATimer
from brt.runtime.log import get_logger, set_level_to_debug
from brt.passes import RouterFixPass, VerticalFusePass, HorizFusePass
from brt.router import switch_capture, RouterBase

from archs.livesr import LiveSR
from archs.vfused_livesr import vFusedLiveSR
from archs.hfused_livesr import hFusedLiveSR
from dataset import get_dataloader

DEFAULT_DATASET = Path(__file__).parent / "dataset/cam1/LQ"
ALL_SUBNET_BS = [
    [6, 7, 12, 27, 8, 8, 8, 12, 12, 4],
    [21, 8, 16, 11, 20, 8, 7, 15],
    [9, 32, 18, 16, 18, 7],
    [19, 25, 18, 36],
]
ALL_NUM_SUBNETS = [len(subnet_bs) for subnet_bs in ALL_SUBNET_BS]
ALL_NUM_CHANNELS = [8, 12, 16, 20]
SUBNET_NUM_BLOCK = 80


logger = get_logger(__file__)
logger.setLevel(logging.DEBUG)
for hdlr in logger.handlers:
    if isinstance(hdlr, logging.StreamHandler) and hdlr.stream is sys.stdout:
        hdlr.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        break
# logger_handler = logging.StreamHandler(stream=sys.stdout)
# logger_handler.setFormatter(
#     logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
# )
# logger.addHandler(logger_handler)

# set_level_to_debug()


timer = CUDATimer(loop=100, repeat=2, export_fname="benchmark_livesr")


def print_load_history(module: nn.Module):
    for subn, subm in module.named_modules():
        if isinstance(subm, RouterBase) and subm.fabric_type == "dispatch":
            load_history = getattr(subm.fabric, "load_history", "no load_history found")
            logger.debug(f"{subn}, {subm.fabric.capturing}, {load_history}")


def time_it(func, func_args, msg):
    # profile(lambda: func(func_args))

    USING_BRT_CUDA_TIMER = True
    # USING_BRT_CUDA_TIMER = False
    if USING_BRT_CUDA_TIMER:
        timer.execute(lambda: func(func_args), msg)
        return timer.avg
    else:
        return (
            Timer(
                f"model(x)",
                setup="import torch; torch.cuda.synchronize()",
                # setup="import torch; torch.cuda.synchronize(); torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False",
                globals={"model": func, "x": func_args},
            )
            .timeit(100)
            .mean
            * 1e3
        )


def benchmark(num_subnets: int, num_feature: int) -> None:
    livesr = LiveSR(num_subnets, SUBNET_NUM_BLOCK, num_feature).cuda()
    logger.info(f"LiveSR {num_feature} {num_subnets} builded")

    switch_capture(livesr, True, "max", "dispatch,combine")
    for input_tensor in dataloader:
        livesr(input_tensor)
    switch_capture(livesr, False)
    print_load_history(livesr)

    raw_time = time_it(livesr, input_tensor, "raw livesr")
    logger.info(
        f"Raw LiveSR,    {num_feature=}, {num_subnets=}: {raw_time:3.06} ms/run"
    )

    router_fix_pass = RouterFixPass(livesr)
    router_fix_pass.run_on_graph()
    livesr_rf = router_fix_pass.finalize()

    vfuse_pass = VerticalFusePass(livesr_rf, sample_inputs={"inputs": input_tensor})
    vfuse_pass.run_on_graph()
    livesr_vf = vfuse_pass.finalize()
    logger.info(f"vFusedLiveSR {num_feature} {num_subnets} builded")
    # logger.debug(f"livesr_vf = {livesr_vf.graph}")

    vfuse_time = time_it(livesr_vf, input_tensor, "vfused livesr")
    logger.info(
        f"vFused LiveSR, {num_feature=}, {num_subnets=}: {vfuse_time:3.06} ms/run"
    )

    hfuse_pass = HorizFusePass(livesr_rf, sample_inputs={"inputs": input_tensor})
    hfuse_pass.run_on_graph()
    livesr_hf = hfuse_pass.finalize()
    logger.info(f"hFusedLiveSR {num_feature} {num_subnets} builded")
    # logger.debug(f"livesr_hf = {livesr_hf.graph}")

    hfuse_time = time_it(livesr_hf, input_tensor, "hfused livesr")
    logger.info(
        f"hFused LiveSR, {num_feature=}, {num_subnets=}: {hfuse_time:3.06} ms/run"
    )


logger.info(f"{SUBNET_NUM_BLOCK = }")
logger.info(f"{ALL_NUM_SUBNETS = }")
logger.info(f"{ALL_NUM_CHANNELS = }")

logger.info("Starts")

dataloader = get_dataloader(DEFAULT_DATASET)
for input_tensor in dataloader:
    break
input_tensor = input_tensor.cuda()
logger.debug(f"{input_tensor.shape}, {input_tensor.device}, {input_tensor.dtype}")

for num_subnets in ALL_NUM_SUBNETS:
    benchmark(num_subnets, ALL_NUM_CHANNELS[0])
# for num_channels in [8]:
for num_channels in ALL_NUM_CHANNELS[1:]:
    benchmark(ALL_NUM_SUBNETS[0], num_channels)

# input("Press any key to start profiling")
# for (module_type, num_feature, num_subnets), model in module_dict.items():
#     logger.info(f"Profiling {module_type}")
#     model = module_dict[module_type]
#     x = input_tensor
#     profile(lambda: model(x))

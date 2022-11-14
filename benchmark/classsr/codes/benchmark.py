import timeit
import logging
import sys
from pathlib import Path

import torch
import cv2
import numpy as np

from models.archs.classSR_fsrcnn_arch import classSR_3class_fsrcnn_net as ClassSR_FSRCNN
from models.archs.classSR_carn_arch import ClassSR as ClassSR_CARN
from models.archs.classSR_srresnet_arch import ClassSR as ClassSR_SRResNet
from models.archs.classSR_rcan_arch import classSR_3class_rcan_net as ClassSR_RCAN

from models.archs.fused_classSR_fsrcnn_arch import (
    fused_classSR_3class_fsrcnn_net as Fused_ClassSR_FSRCNN,
)
from models.archs.classSR_fused_fsrcnn_arch import (
    classSR_3class_fused_fsrcnn_net as ClassSR_Fused_FSRCNN,
)
from models.archs.fused_classSR_rcan_arch import (
    fused_classSR_3class_rcan_net as Fused_ClassSR_RCAN,
)
from models.archs.classSR_fused_rcan_arch import (
    classSR_3class_fused_rcan_net as ClassSR_Fused_RCAN,
)

from brt.runtime.benchmark import profile

BRT_PROJECT_PATH = Path("/home/ycwang/brainstorm_project/brainstorm/")
# BRT_PROJECT_PATH = Path("/home/v-louyang/brainstorm_project/brainstorm/")
IMAGE_PATH = (
    BRT_PROJECT_PATH
    / "benchmark/classsr/datasets_2/1201.png"
)

logger = logging.getLogger("benchmark")
logger.setLevel(logging.DEBUG)
logger_handler = logging.StreamHandler(stream=sys.stdout)
logger_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logger.addHandler(logger_handler)


def crop_cpu(img, crop_sz, step):
    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError("Wrong image shape - {}".format(n_channels))
    h_space = np.arange(0, h - crop_sz + 1, step)
    w_space = np.arange(0, w - crop_sz + 1, step)
    index = 0
    num_h = 0
    lr_list = []
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            index += 1
            if n_channels == 2:
                crop_img = img[x : x + crop_sz, y : y + crop_sz]
            else:
                crop_img = img[x : x + crop_sz, y : y + crop_sz, :]
            lr_list.append(crop_img)
    h = x + crop_sz
    w = y + crop_sz
    return lr_list, num_h, num_w, h, w


n_resgroups = 10
n_resblocks = 20


class ModuleFactory:
    module_dict = {}

    def __class_getitem__(cls, module_type: str):
        module = cls.module_dict.get(module_type, None)
        if module is None:
            module = cls._build(module_type)
            cls.module_dict[module_type] = module
        return module

    @classmethod
    def _build(cls, module_type: str):
        if module_type == "Raw FSRCNN Model":
            model = ClassSR_FSRCNN().cuda().eval()
            full_state_dict = torch.load(
                BRT_PROJECT_PATH
                / "benchmark/classsr/experiments/pre_trained_models/ClassSR_FSRCNN.pth"
            )
            model.load_state_dict(full_state_dict)
        elif module_type == "Raw CARN Model":
            model = ClassSR_CARN().cuda().eval()
            full_state_dict = torch.load(
                BRT_PROJECT_PATH
                / "benchmark/classsr/experiments/pre_trained_models/ClassSR_CARN.pth"
            )
            model.load_state_dict(full_state_dict)
        elif module_type == "Raw SRResNet Model":
            model = ClassSR_SRResNet().cuda().eval()
            full_state_dict = torch.load(
                BRT_PROJECT_PATH
                / "benchmark/classsr/experiments/pre_trained_models/ClassSR_SRResNet.pth"
            )
            model.load_state_dict(full_state_dict)
        elif module_type == "Raw RCAN Model":
            model = (
                ClassSR_RCAN(n_resgroups=n_resgroups, n_resblocks=n_resblocks)
                .cuda()
                .eval()
            )
            full_state_dict = torch.load(
                BRT_PROJECT_PATH
                / "benchmark/classsr/experiments/pre_trained_models/ClassSR_RCAN.pth"
            )
            model.load_state_dict(full_state_dict, strict=False)
        elif module_type == "Horizontal Fused FSRCNN Model":
            model = (
                Fused_ClassSR_FSRCNN(cls["Raw FSRCNN Model"], (34, 38, 29))
                .cuda()
                .eval()
            )
        elif module_type == "Vertical Fused FSRCNN Model":
            model = (
                ClassSR_Fused_FSRCNN(cls["Raw FSRCNN Model"], (34, 38, 29))
                .cuda()
                .eval()
            )
        elif module_type == "Horizontal Fused RCAN Model":
            model = (
                Fused_ClassSR_RCAN(
                    cls["Raw RCAN Model"],
                    (27, 50, 28),
                    objective_func="most_efficient",
                    n_resgroups=n_resgroups,
                    n_resblocks=n_resblocks,
                )
                .cuda()
                .eval()
            )
        elif module_type == "Horizontal Fused RCAN Model (fastest)":
            import pdb; pdb.set_trace()
            model = (
                Fused_ClassSR_RCAN(
                    cls["Raw RCAN Model"],
                    (27, 50, 28),
                    n_resgroups=n_resgroups,
                    n_resblocks=n_resblocks,
                )
                .cuda()
                .eval()
            )
        elif module_type == "Vertical Fused RCAN Model":
            model = (
                ClassSR_Fused_RCAN(
                    cls["Raw RCAN Model"],
                    (27, 50, 28),
                    n_resgroups=n_resgroups,
                    n_resblocks=n_resblocks,
                )
                .cuda()
                .eval()
            )
        else:
            raise ValueError(f"Unsupported module type `{module_type}`")

        logger.info(f"{module_type} builded")
        return model


if __name__ == '__main__':
    image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_UNCHANGED)
    lr_list, num_h, num_w, h, w = crop_cpu(image, 32, 28)
    input_tensor = (
        torch.Tensor(np.array(lr_list))
        .cuda()
        .index_select(
            dim=3,
            index=torch.tensor([2, 1, 0], dtype=torch.int, device="cuda"),
        )
        .permute((0, 3, 1, 2))
        .contiguous()
    )
    input_tensor_div = input_tensor.div(255.0)

    logger.info(f"Start building module")
    module_type_list = [
        # "Raw FSRCNN Model",
        # "Horizontal Fused FSRCNN Model",
        # "Vertical Fused FSRCNN Model",
        # "Raw CARN Model",
        # "Raw SRResNet Model",
        # "Raw RCAN Model",
        # "Horizontal Fused RCAN Model",
        # "Horizontal Fused RCAN Model (fastest)",
        "Vertical Fused RCAN Model",
    ]
    for module_type in module_type_list:
        _ = ModuleFactory[module_type]

    # for module_type in module_type_list:
    #     logger.info(f"Profiling {module_type}")
    #     model = ModuleFactory[module_type]
    #     if "RCAN" not in module_type:
    #         x = input_tensor_div
    #     else:
    #         x = input_tensor
    #     profile(lambda: model(x))

    for n in [1, 100]:
        print(f"* Start timeit: Run {n} times")
        for module_type in module_type_list:
            model = ModuleFactory[module_type]
            if "RCAN" not in module_type:
                x = input_tensor_div
            else:
                x = input_tensor
            time = timeit.timeit(
                f"model(x)",
                setup="from __main__ import model, x; import torch; torch.cuda.synchronize(); torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False",
                number=n,
            )
            print(f"{module_type}:\t\t {time}s in {n} runs ({time/n}s/run)")

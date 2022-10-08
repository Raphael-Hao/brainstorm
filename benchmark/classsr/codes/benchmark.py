import torch
import cv2
import numpy as np
import timeit 
from pathlib import Path

from models.archs.classSR_fsrcnn_arch import classSR_3class_fsrcnn_net as ClassSR_FSRCNN
from models.archs.classSR_carn_arch import ClassSR as ClassSR_CARN
from models.archs.classSR_srresnet_arch import ClassSR as ClassSR_SRResNet 
from models.archs.classSR_rcan_arch import classSR_3class_rcan_net as ClassSR_RCAN 

from models.archs.fused_classSR_fsrcnn_arch import fused_classSR_3class_fsrcnn_net as Fused_ClassSR_FSRCNN
from models.archs.classSR_fused_fsrcnn_arch import classSR_3class_fused_fsrcnn_net as ClassSR_Fused_FSRCNN

from brt.runtime.benchmark import profile

BRT_PROJECT_PATH = Path("/home/ouyang/project/brainstorm/")
# BRT_PROJECT_PATH = Path("/home/v-louyang/brainstorm_project/brainstorm/")
IMAGE_PATH = BRT_PROJECT_PATH / "benchmark/classsr/datasets_2/AIC21_Track1_Vehicle_Counting/Dataset_A_Images/cam1/LQ/0001.png"

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
    .divide(255.0)
)

raw_fsrcnn = {}
raw_fsrcnn["name"] = "Raw FSRCNN Model"
raw_fsrcnn["model"] = ClassSR_FSRCNN().cuda().eval()
full_state_dict = torch.load(BRT_PROJECT_PATH / "benchmark/classsr/experiments/pre_trained_models/ClassSR_FSRCNN.pth")
raw_fsrcnn["model"].load_state_dict(full_state_dict)

raw_carn = {}
raw_carn["name"] = "Raw CARN Model"
raw_carn["model"] = ClassSR_CARN().cuda().eval()
full_state_dict = torch.load(BRT_PROJECT_PATH / "benchmark/classsr/experiments/pre_trained_models/ClassSR_CARN.pth")
raw_carn["model"].load_state_dict(full_state_dict)

raw_srresnet = {}
raw_srresnet["name"] = "Raw SRResNet Model"
raw_srresnet["model"] = ClassSR_SRResNet().cuda().eval()
full_state_dict = torch.load(BRT_PROJECT_PATH / "benchmark/classsr/experiments/pre_trained_models/ClassSR_SRResNet.pth")
raw_srresnet["model"].load_state_dict(full_state_dict)

raw_rcan = {}
raw_rcan["name"] = "Raw RCAN Model"
raw_rcan["model"] = ClassSR_RCAN().cuda().eval()
full_state_dict = torch.load(BRT_PROJECT_PATH / "benchmark/classsr/experiments/pre_trained_models/ClassSR_RCAN.pth")
raw_rcan["model"].load_state_dict(full_state_dict)

# horiz_fused_fsrcnn = {}
# horiz_fused_fsrcnn["name"] = "Horizontal Fused FSRCNN Model"
# horiz_fused_fsrcnn["model"] = Fused_ClassSR_FSRCNN(raw_fsrcnn["model"], (34, 38, 29)).cuda().eval()

# verti_fused_fsrcnn = {}
# verti_fused_fsrcnn["name"] = "Vertical Fused FSRCNN Model"
# verti_fused_fsrcnn["model"] = ClassSR_Fused_FSRCNN(raw_fsrcnn["model"], (34, 38, 29)).cuda().eval()

model_list = [
    raw_fsrcnn, 
    raw_carn, 
    raw_srresnet, 
    raw_rcan, 
    # horiz_fused_fsrcnn,
    # verti_fused_fsrcnn, 
]

# for model_info in model_list:
#     profile(lambda :model_info["model"](input_tensor))

for n in [1, 100]:
    print(f"* Start timeit: Run {n} times")
    for model_info in model_list:
        model = model_info["model"]
        time = timeit.timeit(
            f"model(input_tensor)",
            setup="from __main__ import model, input_tensor; import torch; torch.cuda.synchronize(); torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False",
            number=n,
            )
        print(
            f"{model_info['name']}:\t\t {time}s in {n} runs ({time/n}s/run)"
        )

import torch
import cv2
import numpy as np
import timeit

from models.archs.classSR_fused_fsrcnn_arch import fused_classSR_3class_fsrcnn_net
from models.archs.classSR_fsrcnn_arch import classSR_3class_fsrcnn_net

from brt.runtime.benchmark import profile

IMAGE_PATH = "/home/v-louyang/brainstorm_project/brainstorm/benchmark/classsr/datasets_2/AIC21_Track1_Vehicle_Counting/Dataset_A_Images/cam1/LQ/0001.png"
PTH_PATH = "/home/v-louyang/brainstorm_project/brainstorm/benchmark/classsr/experiments/pre_trained_models/ClassSR_FSRCNN.pth"


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


image = cv2.imread(IMAGE_PATH, cv2.IMREAD_UNCHANGED)
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

full_state_dict = torch.load(PTH_PATH)
# print(full_state_dict.keys())

raw_model = classSR_3class_fsrcnn_net().cuda().eval()
# print([x for x, _ in raw_model.named_parameters()])
raw_model.load_state_dict(full_state_dict)

fused_model = fused_classSR_3class_fsrcnn_net(raw_model, (34, 38, 29)).cuda().eval()

profile(lambda :print(raw_model(input_tensor)[1]))
profile(lambda :print(fused_model(input_tensor)[1]))

# for n in [1, 100, 1000, 10000]:
#     print(f"* Start timeit: Run {n} times")
#     raw_time = timeit.timeit(
#         "x = raw_model(input_tensor)",
#         setup="from __main__ import raw_model, input_tensor; import torch; torch.cuda.synchronize(); torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False",
#         number=n,
#         )
#     print(
#         f"Raw model:   {raw_time}s in {n} runs ({raw_time/n}s/run)"
#     )
#     fused_time = timeit.timeit(
#                 "x = fused_model(input_tensor)",
#                 setup="from __main__ import fused_model, input_tensor; import torch; torch.cuda.synchronize(); torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False",
#                 number=n,
#             )
#     print(
#         f"Fused model: {fused_time}s in {n} runs ({fused_time/n}s/run)"
#     )

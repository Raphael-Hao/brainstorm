from typing import Union
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision


def crop_image(image: torch.Tensor, size: int, stride: int):
    c, h, w = image.shape
    image = image.unsqueeze(0)
    unfolded = torch.nn.functional.unfold(image, size, stride=stride)
    unfolded = unfolded.permute(0, 2, 1)
    unfolded = unfolded.reshape(-1, c, size, size)
    return unfolded.contiguous()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, device: str):
        self.device = torch.device(device)
        self.imgs = list(path.iterdir())

    def __getitem__(self, index: int):
        image_path = self.imgs[index]
        # image = cv2.imread(str(image_path), cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        image = torchvision.io.read_image(str(image_path)).to(dtype=torch.float32)
        sub_images = crop_image(image, 32, 28)
        # print(f"Dataset.__getitem__() return tensor with shape = {sub_images.shape}")
        # print(f"{sub_images.dtype=}, {sub_images.shape=}")
        return sub_images.to(self.device)

    def __len__(self):
        return len(self.imgs)


class DataLoader:
    def __init__(self, *args, **kwargs):
        self.loader = torch.utils.data.DataLoader(*args, **kwargs)

    def __iter__(self):
        for data in self.loader:
            yield data[0]


def get_dataloader(
    dataset_path: Union[Path, str], num_workers: int = 0, device: str = "cuda"
):
    if isinstance(dataset_path, str):
        dataset_path = Path(dataset_path)
    dataset = Dataset(dataset_path, device)
    dataloader = DataLoader(dataset, num_workers=num_workers)
    return dataloader

import argparse
from typing import List, Dict, Type
from pathlib import Path

import torch
from torch import nn

from archs.livesr import LiveSR
from archs.vfused_livesr import vFusedLiveSR
from archs.hfused_livesr import hFusedLiveSR

from dataset import get_dataloader


DEFAULT_MODEL_TYPE = "LiveSR"
DEFAULT_DATASET = "./dataset/cam1/LQ/"

SUBNET_BATCH_SIZE = [6, 7, 12, 27, 8, 8, 8, 12, 12, 4]


def generate_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL_TYPE)
    parser.add_argument("-n", "--subnet-num-block", type=int, default=8)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--objective-func", choices=["fastest", "most-efficient"], default="fastest"
    )
    parser.add_argument("--rank", type=int, default=1)
    args = parser.parse_args()
    args.dataset = Path(args.dataset)
    return args


def build_model(model_type: str, subnet_num_block: int):
    if model_type == "LiveSR":
        model = LiveSR(10, subnet_num_block).cuda()
    elif model_type == "vFused":
        model = vFusedLiveSR(LiveSR(10, subnet_num_block).cuda(), SUBNET_BATCH_SIZE)
    elif model_type == "hFused":
        model = hFusedLiveSR(LiveSR(10, subnet_num_block).cuda(), SUBNET_BATCH_SIZE)
    else:
        raise ValueError
    return model


def main():
    args = generate_args()
    model = build_model(args.model, args.subnet_num_block)
    print(model)
    dataloader = get_dataloader(args.dataset, device="cuda")
    for data in dataloader:
        output = model(data)
        # input()


if __name__ == "__main__":
    main()

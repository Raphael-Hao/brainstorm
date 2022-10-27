import argparse
from typing import List, Dict, Type
from pathlib import Path

from livesr import LiveSR

import torch
from torch import nn

from dataset import get_dataloader

str_to_model_type = {"LiveSR": LiveSR}

DEFAULT_MODEL_TYPE = "LiveSR"
DEFAULT_DATASET = "./dataset/cam1/LQ/"


def generate_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=str_to_model_type.keys(), default=DEFAULT_MODEL_TYPE
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--objective-func", choices=["fastest", "most-efficient"], default="fastest"
    )
    parser.add_argument("--rank", type=int, default=1)
    args = parser.parse_args()
    args.dataset = Path(args.dataset)
    return args


def build_model(model_type: Type[nn.Module]):
    model = model_type().eval()
    return model


def main():
    args = generate_args()
    model_type = str_to_model_type[args.model]
    model = build_model(model_type)
    print(model)
    dataloader = get_dataloader(args.dataset, device="cpu")
    for data in dataloader:
        print(f"input: {data.shape}")
        output = model(data)
        print(f"output: {output.shape}")
        input()


if __name__ == "__main__":
    main()

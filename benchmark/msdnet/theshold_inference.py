import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import math
from msdnet import MSDNet

def threshold_evaluate(model: MSDNet, val_loader: DataLoader, args):
    model.build_routers(thresholds=args.thresholds)
    model.eval()
    acc = 0
    for i, (input, target) in enumerate(val_loader):
        output = model(input)
        pred = output.max(1, keepdim=True)[1]
        acc += pred.eq(target.view_as(pred)).sum().item()

    return acc * 100.0 / len(val_loader)
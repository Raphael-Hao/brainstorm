import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import math
from msdnet import MSDNet
from adaptive_inference import Tester

def threshold_evaluate(model1: MSDNet, val_loader: DataLoader, args):
    print("threshold_evaluate  {}",args.thresholds)
    model1.build_routers(thresholds=args.thresholds)
    model1.eval()
    acc = 0
    for i, (input, target) in enumerate(val_loader):
        output = model1(input)
        pred = output.max(1, keepdim=True)[1]
        acc += pred.eq(target.view_as(pred)).sum().item()

    return acc * 100.0 / len(val_loader)


def threshold_dynamic_evaluate(model1: MSDNet, val_loader: DataLoader, args):
    tester = Tester(model1, args)
    if os.path.exists(os.path.join(args.save, 'logits_single.pth')):
        val_pred, val_target, test_pred, test_target = \
            torch.load(os.path.join(args.save, 'logits_single.pth'))
    else:
        target_predict, val_target = tester.calc(val_loader)
        # test_pred, test_target = tester.calc_logit(test_loader)
        # torch.save((val_pred, val_target, test_pred, test_target),
        #             os.path.join(args.save, 'logits_single.pth'))
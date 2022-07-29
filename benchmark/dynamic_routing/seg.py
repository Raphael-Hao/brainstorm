# encoding: utf-8
# network file -> build basic pipline and decoder for Dynamic Network
# @author: yanwei.li
from typing import Dict

import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
# TODO: differece?
# from brt.frontend.nn import BatchNorm2d
from torch.nn import BatchNorm2d

from image_list import ImageList
from ops import Conv2dNormAct, ShapeSpec, kaiming_init_module
from postprocessing import sem_seg_postprocess

from brt.router import GatherRouter

class DynamicNet4Seg(nn.Module):
    """
    This module implements Dynamic Network for Semantic Segmentation.
    """

    def __init__(self, cfg):
        super().__init__()
        self.constrain_on = cfg.MODEL.BUDGET.CONSTRAIN
        self.unupdate_rate = cfg.MODEL.BUDGET.UNUPDATE_RATE
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.cal_flops = cfg.MODEL.CAL_FLOPS
        self.backbone = cfg.build_backbone(cfg)
        self.sem_seg_head = cfg.build_sem_seg_head(cfg, self.backbone.output_shape())
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.budget_constrint = BudgetConstraint(cfg)
        self.to(self.device)

    def forward(self, batched_inputs, step_rate=0.0, predict_mode=True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
            step_rate: a float, calculated by current_step/total_step,
                This parameter is used for Scheduled Drop Path.
        For now, each item in the list is a dict that contains:
            image: Tensor, image in (C, H, W) format.
            sem_seg: semantic segmentation ground truth
            Other information that's included in the original dicts, such as:
                "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            list[dict]: Each dict is the output for one input image.
                The dict contains one key "sem_seg" whose value is a
                Tensor of the output resolution that represents the
                per-pixel segmentation prediction.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        # features, expt_flops, real_flops = self.backbone(
        features = self.backbone(
            images.tensor, step_rate, predict_mode
        )

        if "sem_seg" in batched_inputs[0]:
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(
                targets, self.backbone.size_divisibility, self.sem_seg_head.ignore_value
            ).tensor
        else:
            targets = None
        if not predict_mode:
            self.sem_seg_head.to(self.device)
        results, losses = self.sem_seg_head(features, targets)
        # calculate flops
        if self.cal_flops:
            real_flops += self.sem_seg_head.flops
            flops = {"real_flops": real_flops, "expt_flops": expt_flops}
        else:
            flops = None
        # use budget constraint for training
        if self.training:
            if self.constrain_on and step_rate >= self.unupdate_rate:
                warm_up_rate = min(1.0, (step_rate - self.unupdate_rate) / 0.02)
                loss_budget = self.budget_constrint(
                    expt_flops, warm_up_rate=warm_up_rate
                )
                losses.update({"loss_budget": loss_budget})
            return losses, flops

        processed_results = []
        for result, input_per_image, image_size in zip(
            results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            r = sem_seg_postprocess(result, image_size, height, width)
            if self.cal_flops:
                processed_results.append({"sem_seg": r, "flops": flops})
            else:
                processed_results.append({"sem_seg": r})
        return processed_results


class SemSegDecoderHead(nn.Module):
    """
    This module implements simple decoder head for Semantic Segmentation.
    It creats decoder on top of the dynamic backbone.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        # fmt: off
        self.in_features = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        feature_strides = {k: v.stride for k, v in input_shape.items()}  # noqa:F841
        feature_channels = {k: v.channels for k, v in input_shape.items()}
        feature_resolution = {
            k: np.array([v.height, v.width])
            for k, v in input_shape.items()
        }
        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        norm = cfg.MODEL.SEM_SEG_HEAD.NORM
        self.loss_weight = cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT
        self.cal_flops = cfg.MODEL.CAL_FLOPS
        self.real_flops = 0.0
        # fmt: on

        self.layer_decoder_list = nn.ModuleList()
        # set affine in BatchNorm
        if "Sync" in norm:
            affine = True
        else:
            affine = False
        # use simple decoder
        for _feat in self.in_features:
            res_size = feature_resolution[_feat]
            in_channel = feature_channels[_feat]
            if _feat == "layer_0":
                out_channel = in_channel
            else:
                out_channel = in_channel // 2
            conv_1x1 = Conv2dNormAct(
                in_channel,
                out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                norm=BatchNorm2d(out_channel),
                activation=nn.ReLU(),
            )
            self.layer_decoder_list.append(conv_1x1)
        # using Kaiming init
        for layer in self.layer_decoder_list:
            kaiming_init_module(layer, mode="fan_in")
        in_channel = feature_channels["layer_0"]
        # the output layer
        self.predictor = nn.Conv2d(
            in_channels=in_channel,
            out_channels=num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.gather_router_2 = GatherRouter(2)

        # using Kaiming init
        kaiming_init_module(self.predictor, mode="fan_in")

    def forward(self, features, targets=None):
        pred, pred_output = None, None
        for _index in range(len(self.in_features)):
            out_index = len(self.in_features) - _index - 1
            out_feat = features[self.in_features[out_index]]

            # if isinstance(out_feat, float): # TODO:
            # if out_feat.numel() == 0:
                # continue

            if out_index <= 2 and pred is not None:
                # out_feat = pred + out_feat # TODO: GatherRouter
                out_feat = self.gather_router_2((pred, out_feat))

            pred = self.layer_decoder_list[out_index](out_feat)
            if out_index > 0:
                pred = F.interpolate(
                    input=pred, scale_factor=2, mode="bilinear", align_corners=False
                )
            else:
                pred_output = pred
        # pred output
        pred_output = self.predictor(pred_output)
        pred_output = F.interpolate(
            input=pred_output, scale_factor=4, mode="bilinear", align_corners=False
        )

        if self.training:
            losses = {}
            losses["loss_sem_seg"] = (
                F.cross_entropy(
                    pred_output,
                    targets,
                    reduction="mean",
                    ignore_index=self.ignore_value,
                )
                * self.loss_weight
            )
            return [], losses
        else:
            return pred_output, {}

    @property
    def flops(self):
        return self.real_flops


class BudgetConstraint(nn.Module):
    """
    Given budget constraint to reduce expected inference FLOPs in the Dynamic Network.
    """

    def __init__(self, cfg):
        super().__init__()
        # fmt: off
        self.loss_weight = cfg.MODEL.BUDGET.LOSS_WEIGHT
        self.loss_mu = cfg.MODEL.BUDGET.LOSS_MU
        self.flops_all = cfg.MODEL.BUDGET.FLOPS_ALL
        self.warm_up = cfg.MODEL.BUDGET.WARM_UP
        # fmt: on

    def forward(self, flops_expt, warm_up_rate=1.0):
        if self.warm_up:
            warm_up_rate = min(1.0, warm_up_rate)
        else:
            warm_up_rate = 1.0
        losses = (
            self.loss_weight
            * warm_up_rate
            * ((flops_expt / self.flops_all - self.loss_mu) ** 2)
        )
        return losses

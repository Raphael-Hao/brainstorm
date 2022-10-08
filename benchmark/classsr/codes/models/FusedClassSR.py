import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss, class_loss_3class, average_loss_3class
from torchsummary import summary
from models.archs import arch_util
import cv2
import numpy as np
from utils import util
from data import util as ut
import os.path as osp
import os

logger = logging.getLogger("base")


class FusedClassSR(BaseModel):
    def __init__(self, opt):
        super(FusedClassSR, self).__init__(opt)

        self.patch_size = int(opt["patch_size"])
        self.step = int(opt["step"])
        self.scale = int(opt["scale"])
        self.name = opt["name"]
        self.which_model = opt["network_G"]["which_model_G"]

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt["train"]

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)

        # if opt["dist"]:
        #     self.netG = DistributedDataParallel(
        #         self.netG, device_ids=[torch.cuda.current_device()]
        #     )
        # else:
        #     self.netG = DataParallel(self.netG)
        self.load()
        self.netG = networks.fuse_G(opt, self.netG).to(self.device)
        # print network
        self.print_network()

    def feed_data(self, data, need_GT=True):
        self.need_GT = need_GT
        self.var_L = data["LQ"].to(self.device)
        self.LQ_path = data["LQ_path"][0]
        if need_GT:
            self.real_H = data["GT"].to(self.device)  # GT
            self.GT_path = data["GT_path"][0]

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        self.fake_H, self.type = self.netG(self.var_L, self.is_train)
        # print(self.type)
        l_pix = self.cri_pix(self.fake_H, self.real_H)
        class_loss = self.class_loss(self.type)
        average_loss = self.average_loss(self.type)
        loss = (
            self.l1w * l_pix
            + self.class_loss_w * class_loss
            + self.average_loss_w * average_loss
        )

        if step % self.pf == 0:
            self.print_res(self.type)

        loss.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict["l_pix"] = l_pix.item()
        self.log_dict["class_loss"] = class_loss.item()
        self.log_dict["average_loss"] = average_loss.item()
        self.log_dict["loss"] = loss.item()

    def test(self):
        self.netG.eval()
        self.var_L = cv2.imread(self.LQ_path, cv2.IMREAD_UNCHANGED)
        if self.need_GT:
            self.real_H = cv2.imread(self.GT_path, cv2.IMREAD_UNCHANGED)

        lr_list, num_h, num_w, h, w = self.crop_cpu(
            self.var_L, self.patch_size, self.step
        )
        if self.need_GT:
            gt_list = self.crop_cpu(self.real_H, self.patch_size * 4, self.step * 4)[0]
        sr_list = []
        index = 0

        psnr_type1 = 0
        psnr_type2 = 0
        psnr_type3 = 0

        brt_lr = (
            torch.Tensor(np.array(lr_list))
            .to(self.device)
            .index_select(
                dim=3,
                index=torch.tensor([2, 1, 0], dtype=torch.int, device=self.device),
            )
            .permute((0, 3, 1, 2))
            .contiguous()
        )
        # WARNING: unknow code: torch.Tensor.__getitem__
        if self.need_GT:
            brt_gt = (
                torch.Tensor(np.array(gt_list))
                .to(self.device)
                .index_select(
                    dim=3,
                    index=torch.tensor([2, 1, 0], dtype=torch.int, device=self.device),
                )
                .permute((0, 3, 1, 2))
                .contiguous()
            )
        if self.which_model != "classSR_3class_rcan":
            brt_lr = brt_lr.divide(255.0)
        assert brt_lr.shape[1:] == (3, 32, 32)

        with torch.no_grad():
            brt_srt, brt_type = self.netG(brt_lr, False)

        sr_list = []
        for srt in brt_srt:
            if self.which_model == "classSR_3class_rcan":
                sr_img = util.tensor2img(
                    srt.detach().float().cpu(), out_type=np.uint8, min_max=(0, 255)
                )
            else:
                sr_img = util.tensor2img(srt.detach().float().cpu())
            sr_list.append(sr_img)
        self.fake_H = self.combine(
            sr_list, num_h, num_w, h, w, self.patch_size, self.step
        )
        if self.opt["add_mask"]:
            assert False, "no implement"
            self.fake_H_mask = self.combine_addmask(
                sr_list, num_h, num_w, h, w, self.patch_size, self.step, type_res
            )
        if self.need_GT:
            self.real_H = self.real_H[0 : h * self.scale, 0 : w * self.scale, :]
        # self.num_res = self.print_res(type_res)
        self.num_res = brt_type
        # self.psnr_res = [psnr_type1, psnr_type2, psnr_type3]
        self.psnr_res = [0, 1, 2]

        # NOTE: type_res
        # self.type_res = torch.argmax(type_res, dim=1).tolist()
        self.type_res = brt_type

        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["LQ"] = self.var_L
        out_dict["rlt"] = self.fake_H
        out_dict["num_res"] = self.num_res
        out_dict["psnr_res"] = self.psnr_res
        out_dict["type_res"] = self.type_res
        if need_GT:
            out_dict["GT"] = self.real_H
        if self.opt["add_mask"]:
            out_dict["rlt_mask"] = self.fake_H_mask
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(
            self.netG, DistributedDataParallel
        ):
            net_struc_str = "{} - {}".format(
                self.netG.__class__.__name__, self.netG.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                "Network G structure: {}, with parameters: {:,d}".format(
                    net_struc_str, n
                )
            )
            logger.info(s)

    def load(self):
        load_path_G = self.opt["path"]["pretrain_model_G"]
        load_path_classifier = self.opt["path"]["pretrain_model_classifier"]
        load_path_G_branch3 = self.opt["path"]["pretrain_model_G_branch3"]
        load_path_G_branch2 = self.opt["path"]["pretrain_model_G_branch2"]
        load_path_G_branch1 = self.opt["path"]["pretrain_model_G_branch1"]
        load_path_Gs = [load_path_G_branch1, load_path_G_branch2, load_path_G_branch3]
        if load_path_G is not None:
            logger.info("Loading model for G [{:s}] ...".format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt["path"]["strict_load"])
        if load_path_classifier is not None:
            logger.info(
                "Loading model for classfier [{:s}] ...".format(load_path_classifier)
            )
            self.load_network_classifier_rcan(
                load_path_classifier, self.netG, self.opt["path"]["strict_load"]
            )
        if (
            load_path_G_branch3 is not None
            and load_path_G_branch1 is not None
            and load_path_G_branch2 is not None
        ):
            logger.info(
                "Loading model for branch1 [{:s}] ...".format(load_path_G_branch1)
            )
            logger.info(
                "Loading model for branch2 [{:s}] ...".format(load_path_G_branch2)
            )
            logger.info(
                "Loading model for branch3 [{:s}] ...".format(load_path_G_branch3)
            )
            self.load_network_classSR_3class(
                load_path_Gs, self.netG, self.opt["path"]["strict_load"]
            )

    def save(self, iter_label):
        self.save_network(self.netG, "G", iter_label)

    def crop_cpu(self, img, crop_sz, step):
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

    def combine(self, sr_list, num_h, num_w, h, w, patch_size, step):
        index = 0
        sr_img = np.zeros((h * self.scale, w * self.scale, 3), "float32")
        for i in range(num_h):
            for j in range(num_w):
                sr_img[
                    i * step * self.scale : i * step * self.scale
                    + patch_size * self.scale,
                    j * step * self.scale : j * step * self.scale
                    + patch_size * self.scale,
                    :,
                ] += sr_list[index]
                index += 1
        sr_img = sr_img.astype("float32")

        for j in range(1, num_w):
            sr_img[
                :,
                j * step * self.scale : j * step * self.scale
                + (patch_size - step) * self.scale,
                :,
            ] /= 2

        for i in range(1, num_h):
            sr_img[
                i * step * self.scale : i * step * self.scale
                + (patch_size - step) * self.scale,
                :,
                :,
            ] /= 2
        return sr_img

    def combine_addmask(self, sr_list, num_h, num_w, h, w, patch_size, step, type):
        index = 0
        sr_img = np.zeros((h * self.scale, w * self.scale, 3), "float32")

        for i in range(num_h):
            for j in range(num_w):
                sr_img[
                    i * step * self.scale : i * step * self.scale
                    + patch_size * self.scale,
                    j * step * self.scale : j * step * self.scale
                    + patch_size * self.scale,
                    :,
                ] += sr_list[index]
                index += 1
        sr_img = sr_img.astype("float32")

        for j in range(1, num_w):
            sr_img[
                :,
                j * step * self.scale : j * step * self.scale
                + (patch_size - step) * self.scale,
                :,
            ] /= 2

        for i in range(1, num_h):
            sr_img[
                i * step * self.scale : i * step * self.scale
                + (patch_size - step) * self.scale,
                :,
                :,
            ] /= 2

        index2 = 0
        for i in range(num_h):
            for j in range(num_w):
                # add_mask
                alpha = 1
                beta = 0.2
                gamma = 0
                bbox1 = [
                    j * step * self.scale + 8,
                    i * step * self.scale + 8,
                    j * step * self.scale + patch_size * self.scale - 9,
                    i * step * self.scale + patch_size * self.scale - 9,
                ]  # xl,yl,xr,yr
                zeros1 = np.zeros((sr_img.shape), "float32")

                if torch.max(type, 1)[1].data.squeeze()[index2] == 0:
                    # mask1 = cv2.rectangle(zeros1, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]),
                    #                      color=(0, 0, 0), thickness=1)
                    mask2 = cv2.rectangle(
                        zeros1,
                        (bbox1[0] + 1, bbox1[1] + 1),
                        (bbox1[2] - 1, bbox1[3] - 1),
                        color=(0, 255, 0),
                        thickness=-1,
                    )  # simple green
                elif torch.max(type, 1)[1].data.squeeze()[index2] == 1:
                    # mask1 = cv2.rectangle(zeros1, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]),
                    #                       color=(0, 0, 0), thickness=1)
                    mask2 = cv2.rectangle(
                        zeros1,
                        (bbox1[0] + 1, bbox1[1] + 1),
                        (bbox1[2] - 1, bbox1[3] - 1),
                        color=(0, 255, 255),
                        thickness=-1,
                    )  # medium yellow
                elif torch.max(type, 1)[1].data.squeeze()[index2] == 2:
                    # mask1 = cv2.rectangle(zeros1, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]),
                    #                       color=(0, 0, 0), thickness=1)
                    mask2 = cv2.rectangle(
                        zeros1,
                        (bbox1[0] + 1, bbox1[1] + 1),
                        (bbox1[2] - 1, bbox1[3] - 1),
                        color=(0, 0, 255),
                        thickness=-1,
                    )  # hard red

                sr_img = cv2.addWeighted(sr_img, alpha, mask2, beta, gamma)
                # sr_img = cv2.addWeighted(sr_img, alpha, mask1, 1, gamma)
                index2 += 1
        return sr_img

    def print_res(self, type_res):
        num0 = 0
        num1 = 0
        num2 = 0

        for i in torch.max(type_res, 1)[1].data.squeeze():
            if i == 0:
                num0 += 1
            if i == 1:
                num1 += 1
            if i == 2:
                num2 += 1

        return [num0, num1, num2]

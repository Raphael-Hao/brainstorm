import torch

import models.archs.FSRCNN_arch as FSRCNN
import models.archs.CARN_arch as CARN
import models.archs.SRResNet_arch as SRResNet
import models.archs.RCAN_arch as RCAN

import models.archs.classSR_fsrcnn_arch as classSR_fsrcnn
import models.archs.classSR_carn_arch as classSR_carn
import models.archs.classSR_srresnet_arch as classSR_srresnet
import models.archs.classSR_rcan_arch as classSR_rcan

import models.archs.fused_classSR_fsrcnn_arch as fused_classSR_3class_fsrcnn
import models.archs.fused_classSR_rcan_arch as fused_classSR_3class_rcan
import models.archs.classSR_fused_fsrcnn_arch as classSR_3class_fused_fsrcnn
import models.archs.classSR_fused_rcan_arch as classSR_3class_fused_rcan

# Generator
def define_G(opt):
    opt_net = opt["network_G"]
    which_model = opt_net["which_model_G"]

    # image restoration
    if which_model == "MSRResNet":
        netG = SRResNet.MSRResNet(
            in_nc=opt_net["in_nc"],
            out_nc=opt_net["out_nc"],
            nf=opt_net["nf"],
            nb=opt_net["nb"],
            upscale=opt_net["scale"],
        )

    elif which_model == "RCAN":
        netG = RCAN.RCAN(
            n_resblocks=opt_net["n_resblocks"],
            n_feats=opt_net["n_feats"],
            res_scale=opt_net["res_scale"],
            n_colors=opt_net["n_colors"],
            rgb_range=opt_net["rgb_range"],
            scale=opt_net["scale"],
            reduction=opt_net["reduction"],
            n_resgroups=opt_net["n_resgroups"],
        )
    elif which_model == "CARN_M":
        netG = CARN.CARN_M(
            in_nc=opt_net["in_nc"],
            out_nc=opt_net["out_nc"],
            nf=opt_net["nf"],
            scale=opt_net["scale"],
            group=opt_net["group"],
        )

    elif which_model == "fsrcnn":
        netG = FSRCNN.FSRCNN_net(
            input_channels=opt_net["in_nc"],
            upscale=opt_net["scale"],
            d=opt_net["d"],
            s=opt_net["s"],
            m=opt_net["m"],
        )

    elif which_model == "classSR_3class_fsrcnn_net":
        netG = classSR_fsrcnn.classSR_3class_fsrcnn_net(
            in_nc=opt_net["in_nc"], out_nc=opt_net["out_nc"]
        )
    elif which_model == "classSR_3class_rcan":
        netG = classSR_rcan.classSR_3class_rcan_net(
            in_nc=opt_net["in_nc"], out_nc=opt_net["out_nc"]
        )
    elif which_model == "classSR_3class_srresnet":
        netG = classSR_srresnet.ClassSR(
            in_nc=opt_net["in_nc"], out_nc=opt_net["out_nc"]
        )
    elif which_model == "classSR_3class_carn":
        netG = classSR_carn.ClassSR(in_nc=opt_net["in_nc"], out_nc=opt_net["out_nc"])

    elif (
        which_model == "classSR_3class_fused_fsrcnn_net"
        or which_model == "fused_classSR_3class_fsrcnn_net"
    ):
        netG = classSR_fsrcnn.classSR_3class_fsrcnn_net(
            in_nc=opt_net["in_nc"], out_nc=opt_net["out_nc"]
        )
    elif (
        which_model == "classSR_3class_fused_rcan_net"
        or which_model == "fused_classSR_3class_rcan_net"
    ):
        netG = classSR_rcan.classSR_3class_rcan_net(
            in_nc=opt_net["in_nc"], out_nc=opt_net["out_nc"]
        )

    else:
        raise NotImplementedError(
            "Generator model [{:s}] not recognized".format(which_model)
        )

    return netG


def fuse_G(opt, raw):
    which_model = opt["network_G"]["which_model_G"]
    input_bs = opt["network_G"]["in_bs"]

    # image restoration
    if which_model == "fused_classSR_3class_fsrcnn_net":
        netG = fused_classSR_3class_fsrcnn.fused_classSR_3class_fsrcnn_net(
            raw, input_bs
        ).cuda()
    elif which_model == "fused_classSR_3class_rcan_net":
        netG = fused_classSR_3class_rcan.fused_classSR_3class_rcan_net(
            raw, input_bs
        ).cuda()
    elif which_model == "classSR_3class_fused_fsrcnn_net":
        netG = classSR_3class_fused_fsrcnn.classSR_3class_fused_fsrcnn_net(
            raw, input_bs
        ).cuda()
    elif which_model == "classSR_3class_fused_rcan_net":
        netG = classSR_3class_fused_rcan.classSR_3class_fused_rcan_net(
            raw, input_bs
        ).cuda()
    else:
        raise NotImplementedError(
            "Generator model [{:s}] not recognized".format(which_model)
        )

    return netG

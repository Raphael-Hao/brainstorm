import random, sys, os
import torch
import torch.nn as nn
import torch.utils.data as data

# import ops
# from dataset import DatasetForDASH
# from option import opt
# import template
import math


class Conv_ReLU_Block(nn.Module):
    def __init__(self, nFeat, bias):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=nFeat,
            out_channels=nFeat,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class ResBlock(nn.Module):
    def __init__(
        self, nFeat, bias=True, bn=False, act=None, res_scale=1, kernel_size=3
    ):
        super(ResBlock, self).__init__()

        modules_body = []
        for i in range(2):
            modules_body.append(
                nn.Conv2d(
                    in_channels=nFeat,
                    out_channels=nFeat,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=(kernel_size // 2),
                    bias=bias,
                )
            )
            if bn:
                modules_body.append(nn.BatchNorm2d(nFeat))
            if i == 0:
                if act is None:
                    act = nn.ReLU(True)
                modules_body.append(act)

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        if self.res_scale != 1:
            res = self.body(x).mul(self.res_scale)
        else:
            res = self.body(x)
        # res = torch.mul(self.body(x), (self.res_scale))
        res += x
        return res


class Upsampler(nn.Sequential):
    def __init__(self, scale, nFeat, bn=False, act=False, bias=True):
        super(Upsampler, self).__init__()

        modules = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                # modules.append(conv(nFeat, 4 * nFeat, 3, bias))
                modules.append(
                    nn.Conv2d(
                        in_channels=nFeat,
                        out_channels=4 * nFeat,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=bias,
                    )
                )
                modules.append(nn.PixelShuffle(2))
                if bn:
                    modules.append(nn.BatchNorm2d(nFeat))
                if act:
                    modules.append(act())
        elif scale == 3:
            # modules.append(conv(nFeat, 9 * nFeat, 3, bias))
            modules.append(
                nn.Conv2d(
                    in_channels=nFeat,
                    out_channels=9 * nFeat,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=bias,
                )
            )
            modules.append(nn.PixelShuffle(3))
            if bn:
                modules.append(nn.BatchNorm2d(nFeat))
            if act:
                modules.append(act())
        else:
            raise NotImplementedError

        self.upsampler = nn.Sequential(*modules)

    def forward(self, x):
        return self.upsampler(x)


"""
random_uniform, random_gradual_**: These functions are used for training a scalable DNN by providing the probability distribution of each layer
"""


def random_uniform(elem_list):
    return elem_list


def random_gradual_01(elem_list, bias, weight):
    random_list = []

    for i in range(len(elem_list)):
        random_list.extend([elem_list[i]] * (bias + weight * i))

    return random_list


def random_gradual_02(elem_list, bias, weight):
    random_list = []

    for i in range(len(elem_list)):
        if i == len(elem_list) - 1:
            random_list.extend([elem_list[i]] * len(random_list))
        else:
            random_list.extend([elem_list[i]] * (bias + weight * i))

    return random_list


def random_gradual_03(elem_list):
    random_list = []

    if len(elem_list) == 1:
        random_list.extend([elem_list[0]])
    else:
        for i in range(len(elem_list)):
            if i == len(elem_list) - 1:
                random_list.extend([elem_list[i]] * len(random_list))
            else:
                random_list.extend([elem_list[i]] * 1)

    return random_list


def random_gradual_04(elem_list):
    random_list = []

    if len(elem_list) == 1:
        random_list.extend([elem_list[0]])
    else:
        for i in range(len(elem_list)):
            if i == len(elem_list) - 1:
                random_list.extend([elem_list[i]] * (len(random_list) - 2))
            else:
                random_list.extend([elem_list[i]] * 1)

    return random_list


# M(ulti-resolution)Nas_Net
class MultiNetwork(nn.Module):
    def __init__(self, config, act=nn.ReLU(True)):
        super(MultiNetwork, self).__init__()

        self.networks = nn.ModuleList()
        self.scale_dict = {}

        # set model parameter (e.g., layer, channel)
        for iteration, scale in enumerate(config):
            self.networks.append(
                SingleNetwork(
                    num_block=config[scale]["block"],
                    num_feature=config[scale]["feature"],
                    num_channel=3,
                    scale=scale,
                    output_filter=config[scale]["output_filter"],
                    bias=True,
                    act=act,
                )
            )
            self.scale_dict[scale] = iteration

        self.target_scale = None

    def getOutputNodes(self):
        assert self.target_scale != None
        return self.networks[self.scale_dict[self.target_scale]].getOutputNodes()

    def setTargetScale(self, scale):
        assert scale in self.scale_dict.keys()
        self.target_scale = scale

    def forward(self, x, idx=None):
        assert self.target_scale != None
        x = self.networks[self.scale_dict[self.target_scale]].forward(x, idx)

        return x

    # save a scalable DNN by dividing into chunks
    def save_chunk(self, save_dir):
        total_dict = {}
        model_out_path = os.path.join(save_dir, "DNN_chunk_1.pth")
        state_dict = self.state_dict()
        for index in range(4):
            dict01 = {
                k: v for (k, v) in state_dict.items() if "{}.head".format(index) in k
            }
            dict02 = {
                k: v for (k, v) in state_dict.items() if "{}.tail".format(index) in k
            }
            dict03 = {
                k: v for (k, v) in state_dict.items() if "{}.upscale".format(index) in k
            }
            dict04 = {
                k: v
                for (k, v) in state_dict.items()
                if "{}.body_end.".format(index) in k
            }
            total_dict = {**total_dict, **dict01, **dict02, **dict03, **dict04}
        torch.save(total_dict, model_out_path)

        total_dict = {}
        model_out_path = os.path.join(save_dir, "DNN_chunk_2.pth")
        total_dict = {
            k: v for (k, v) in state_dict.items() if "{}.body.0.0".format(0) in k
        }

        for index in range(4):
            partial_dict = {
                k: v
                for (k, v) in state_dict.items()
                if "{}.body.0.0".format(index) in k
            }
            total_dict = {**total_dict, **partial_dict}
        torch.save(total_dict, model_out_path)

        total_dict = {}
        model_out_path = os.path.join(save_dir, "DNN_chunk_3.pth")
        total_dict = {
            k: v for (k, v) in state_dict.items() if "{}.body.1.0".format(0) in k
        }

        for index in range(4):
            partial_dict = {
                k: v
                for (k, v) in state_dict.items()
                if "{}.body.1.0".format(index) in k
            }
            total_dict = {**total_dict, **partial_dict}
        torch.save(total_dict, model_out_path)

        total_dict = {}
        model_out_path = os.path.join(save_dir, "DNN_chunk_4.pth")
        total_dict = {
            k: v for (k, v) in state_dict.items() if "{}.body.2.0".format(0) in k
        }

        for index in range(4):
            partial_dict = {
                k: v
                for (k, v) in state_dict.items()
                if "{}.body.2.0".format(index) in k
            }
            total_dict = {**total_dict, **partial_dict}
        torch.save(total_dict, model_out_path)

        total_dict = {}
        model_out_path = os.path.join(save_dir, "DNN_chunk_5.pth")
        total_dict = {
            k: v for (k, v) in state_dict.items() if "{}.body.3.0".format(0) in k
        }

        for index in range(4):
            partial_dict = {
                k: v
                for (k, v) in state_dict.items()
                if "{}.body.3.0".format(index) in k
            }
            total_dict = {**total_dict, **partial_dict}
        torch.save(total_dict, model_out_path)


# S(ingle-resolution)NAS_Net
class SingleNetwork(nn.Module):
    def __init__(
        self,
        num_block,
        num_feature,
        num_channel,
        scale,
        output_filter,
        bias=True,
        act=None,
    ):
        super(SingleNetwork, self).__init__()
        if act is None:
            act = nn.ReLU(True)

        self.num_block = num_block
        self.num_feature = num_feature
        self.num_channel = num_channel
        self.scale = scale

        assert self.scale in [1, 2, 3, 4]

        # output_filter is used for selecting intermediate layers that are used as early-exit
        self.outputNode = []
        for i in range(self.num_block // output_filter + 1):
            self.outputNode.append(self.num_block - output_filter * i)
            assert self.num_block - (output_filter * i) >= 0
        if self.num_block not in self.outputNode:
            self.outputNode.append(self.num_block)
        self.outputNode = sorted(self.outputNode)
        self.outputList = random_gradual_03(self.outputNode)

        self.head = nn.Sequential(
            *[
                nn.Conv2d(
                    in_channels=self.num_channel,
                    out_channels=self.num_feature,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=bias,
                )
            ]
        )

        self.body = nn.ModuleList()
        for _ in range(self.num_block):
            # modules_body = [ResBlock(self.num_feature, bias=bias)]
            modules_body = [ResBlock(self.num_feature, bias=bias, act=act)]
            self.body.append(nn.Sequential(*modules_body))

        body_end = []
        body_end.append(
            nn.Conv2d(
                in_channels=self.num_feature,
                out_channels=self.num_feature,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            )
        )
        self.body_end = nn.Sequential(*body_end)

        if self.scale > 1:
            self.upscale = nn.Sequential(
                *Upsampler(self.scale, self.num_feature, bias=bias)
            )

        self.tail = nn.Sequential(
            *[
                nn.Conv2d(
                    in_channels=self.num_feature,
                    out_channels=self.num_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=bias,
                )
            ]
        )

    def getOutputNodes(self):
        return self.outputNode

    def forward(self, x, idx=None):
        idx = self.num_block

        x = self.head(x)
        res = x

        for i in range(idx):
            res = self.body[i](res)
        res = self.body_end(res)
        res += x

        if self.scale > 1:
            x = self.upscale(res)
        else:
            x = res

        x = self.tail(x)

        return x


# if __name__ == "__main__":
#     """Simple test code for model"""
#     model = MultiNetwork(template.get_nas_config('low')).to('cuda:0')
#     model.setTargetScale(4)
#     dataset = DatasetForDASH(opt)
#     dataset.setTargetLR(240)
#     dataset.setDatasetType('train')
#     #print(len(dataset))
#     #print(model.getOutputNodes(4))
#     dataloader = data.DataLoader(dataset=dataset, num_workers=opt.num_thread, batch_size=opt.num_batch, pin_memory=True, shuffle=True)

#     for iteration, batch in enumerate(dataloader):
#         batch[0] = batch[0].cuda()
#         output = model(batch[0])
#         print(batch[0].size(), output.size())
#         break

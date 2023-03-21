import sys
from typing import List, Dict
import pickle

import tvm
from tvm import auto_scheduler, relay
from tvm.contrib import graph_executor

from sklearn.cluster import MiniBatchKMeans

import torch
from torch import nn, fx
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils import dlpack

from brt.runtime import BRT_CACHE_PATH
from brt.router import ScatterRouter, GatherRouter
from brt.trace.leaf_node import register_leaf_node

from archs.nas_mdsr import SingleNetwork as NAS_MDSR


class LiveSR(nn.Module):
    """LiveSR using brainstorm"""

    def __init__(self, n_subnets: int = 10, subnet_num_block: int = 8, num_feature=36):
        super().__init__()
        self.n_subnets = n_subnets
        self.subnet_num_block = subnet_num_block
        self.num_feature = num_feature
        # self.classifier = Classifier(n_subnets).eval()
        self.classifier = TunedClassifier(n_subnets, 88).eval()
        self.scatter = ScatterRouter(capturing=True, capture_mode="max")
        self.subnets = nn.ModuleList(
            NAS_MDSR(
                num_block=self.subnet_num_block,
                num_feature=num_feature,
                num_channel=3,
                scale=4,
                output_filter=2,
            )
            for _ in range(n_subnets)
        )
        self.gather = GatherRouter()

    def forward(self, inputs: torch.Tensor):
        """@param x: Tensor with shape [N, 3, 32, 32]"""
        scores = self.classifier(inputs)
        # print(scores)
        scattered = self.scatter(inputs, scores)
        subnet_outputs = []
        for i in range(self.n_subnets):
            m = self.subnets[i]
            x = scattered[i]
            subnet_outputs.append(m(x))
        # subnet_outputs = [m(x, m.num_block) for m, x in zip(self.subnets, scattered)]
        gathered = self.gather(subnet_outputs)
        return gathered


@register_leaf_node
class Classifier(nn.Module):
    def __init__(self, n_subnets: int):
        super().__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
        with open(
            BRT_CACHE_PATH.parent / f"benchmark/livesr/kmeans_{n_subnets}.pkl", "rb"
        ) as pkl:
            self.kmeans: MiniBatchKMeans = pickle.load(pkl)["kmeans"]

    def forward(self, x: torch.Tensor):
        """@param x: Tensor with shape [N, 3, 32, 32]"""
        output = torch.empty(x.shape[0], 512, requires_grad=False, device=x.device)
        copy_output = lambda m, i, o: output.copy_(o.detach().squeeze())
        hook = self.resnet._modules.get("avgpool").register_forward_hook(copy_output)
        self.resnet(x)
        hook.remove()
        distacne = self.kmeans.transform(output.cpu())
        t_distacne = 1.0 / torch.from_numpy(distacne).to(device=x.device)
        return t_distacne


@register_leaf_node
class TunedClassifier(nn.Module):
    def __init__(self, n_subnets: int = 10, bs: int = 88):
        super().__init__()
        self.bs = bs
        raw_resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).cuda().eval()
        target = tvm.target.Target("cuda")
        input_shape = (bs, 3, 32, 32)
        traced = fx.symbolic_trace(raw_resnet)
        for node in traced.graph.nodes:
            if node.name == "avgpool":
                avg_pool_node = node
            if node.op == "output":
                node.args = (avg_pool_node,)
        traced.recompile()
        mod, params = relay.frontend.from_pytorch(
            torch.jit.trace(
                traced,
                example_inputs=torch.randn(input_shape).cuda(),
            ),
            input_infos=[("input0", input_shape)],
        )
        with auto_scheduler.ApplyHistoryBest(
            BRT_CACHE_PATH.parent / f"benchmark/livesr/resnet-18-NCHW-B88-cuda.json"
        ):
            with tvm.transform.PassContext(
                opt_level=3, config={"relay.backend.use_auto_scheduler": True}
            ):
                lib = relay.build(mod, target=target, params=params)

        # Create graph executor
        dev = tvm.device(str(target), 0)
        self.resnet = graph_executor.GraphModule(lib["default"](dev))
        self.kmeans = kMeans(n_subnets)

    def forward(self, inputs: torch.Tensor):
        tvm_input = tvm.nd.from_dlpack(dlpack.to_dlpack(inputs))
        self.resnet.set_input("input0", tvm_input)
        self.resnet.run()
        tvm_out = dlpack.from_dlpack(self.resnet.get_output(0).to_dlpack()).squeeze()
        dis = self.kmeans(tvm_out)
        return dis


@register_leaf_node
class kMeans(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        with open(BRT_CACHE_PATH.parent / f"benchmark/livesr/kmeans_{k}.pkl", "rb") as pkl:
            kmeans: MiniBatchKMeans = pickle.load(pkl)["kmeans"]
        self.kmeans_centers = torch.nn.Parameter(
            torch.from_numpy(kmeans.cluster_centers_).to(torch.float32),
            requires_grad=False,
        )

    def forward(self, inputs: torch.Tensor):
        distance = (
            (inputs.unsqueeze(1).repeat(1, self.k, 1) - self.kmeans_centers)
            .square()
            .sum(dim=2)
        )
        weight = 1.0 / distance
        weight = weight.cuda()
        return weight

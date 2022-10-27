import sys
from typing import List, Dict
import pickle

# import

from sklearn.cluster import MiniBatchKMeans
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

from include.nas_mdsr import SingleNetwork as NAS_MDSR

# from include.img2vec_pytorch import Img2Vec

from brt.router import ScatterRouter, GatherRouter


class LiveSR(nn.Module):
    """LiveSR using brainstorm"""

    def __init__(self, n_subnets: int = 10):
        super().__init__()
        self.classifier = Classifier(n_subnets).eval()
        self.scatter = ScatterRouter(
            protocol_type="label", protocol_kwargs={"flow_num": 10}
        )
        self.subnets = nn.ModuleList(
            NAS_MDSR(
                num_block=1,
                num_feature=48,
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
        print(scores)
        scattered = self.scatter(inputs, scores)
        subnet_outputs = [m(x) for m, x in zip(self.subnets, scattered)]
        gathered = self.gather(subnet_outputs)
        return gathered


class Classifier(nn.Module):
    def __init__(self, n_subnets: int):
        super().__init__()
        with open(
            f"/home/v-louyang/brt/benchmark/livesr/kmeans_{n_subnets}.pkl", "rb"
        ) as pkl:
            self.kmeans: MiniBatchKMeans = pickle.load(pkl)["kmeans"]
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()

    def forward(self, x: torch.Tensor):
        """@param x: Tensor with shape [N, 3, 32, 32]"""
        output = torch.empty(x.shape[0], 512, requires_grad=False)
        copy_output = lambda m, i, o: output.copy_(o.detach().squeeze())
        hook = self.resnet._modules.get("avgpool").register_forward_hook(copy_output)
        self.resnet(x)
        hook.remove()
        print(output.requires_grad)
        labels = self.kmeans.predict(output)
        labels = torch.from_numpy(labels).to(dtype=torch.long)
        return labels

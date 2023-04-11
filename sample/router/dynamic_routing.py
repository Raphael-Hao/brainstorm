import torch
import torch.nn as nn
from brt.router import ScatterRouter, GatherRouter
from brt import Annotator


class DynamicRouting(nn.Module):
    def __init__(self, dst_num):
        super().__init__()
        self.annotator = Annotator([0])
        self.route_func = nn.Sequential(
            nn.Conv2d(4, 2, 1), nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(2, 2, 1),
        )
        self.scatter_router_0 = ScatterRouter(
            dispatch_score=True, protocol_type="threshold", fabric_type="dispatch",
        )
        self.scatter_router_1 = ScatterRouter(
            protocol_type="threshold", fabric_type="dispatch",
        )
        self.expert1 = nn.Conv2d(4, 4, 1)
        self.expert2 = nn.Conv2d(4, 8, 1)
        self.expert3 = nn.Conv2d(4, 4, 1)
        self.expert4 = nn.Conv2d(4, 8, 1)
        self.gather_router_0 = GatherRouter(fabric_type="combine")
        self.gather_router_1 = GatherRouter(fabric_type="combine")

    def forward(self, x, y):
        x = self.annotator(x)
        y = self.annotator(y)
        gates_x = self.route_func(x).view(-1, 2)
        gates_y = self.route_func(y).view(-1, 2)

        route_results_x, _ = self.scatter_router_0(x, gates_x)
        route_results_y = self.scatter_router_1(y, gates_y)
        x_0 = self.expert1(route_results_x[0])
        x_1 = self.expert2(route_results_x[1])
        y_0 = self.expert3(route_results_y[0])
        y_1 = self.expert4(route_results_y[1])
        x = self.gather_router_0([x_0, y_0])
        y = self.gather_router_1([x_1, y_1])
        return x, y


dy_model = DynamicRouting(2)
dy_model = dy_model.cuda().eval()
with torch.inference_mode():
    for i in range(100):
        x = torch.randn((18, 4, 2, 2)).cuda()
        y = torch.randn((18, 4, 2, 2)).cuda()
        x, y = dy_model(x, y)
        print(x)
        print(y)
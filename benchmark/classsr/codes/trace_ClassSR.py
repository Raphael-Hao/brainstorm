# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from models.archs.classSR_fsrcnn_arch import classSR_3class_fsrcnn_net
from models.archs.classSR_carn_arch import ClassSR as classSR_3class_carn_net
from models.archs.classSR_srresnet_arch import ClassSR as classSR_3class_srresnet_net
from models.archs.classSR_rcan_arch import classSR_3class_rcan as classSR_3class_rcan_net
from models.archs.classSR_fused_fsrcnn_arch import fused_classSR_3class_fsrcnn_net

import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from torch.fx.graph_module import GraphModule

from brt.trace.graph import GraphTracer
from brt.runtime import BRT_CACHE_PATH

from torch.fx.passes.graph_drawer import FxGraphDrawer

if __name__ == "__main__":
    models = {
        'classsr_fsrcnn': classSR_3class_fsrcnn_net,
        'classsr_carn': classSR_3class_carn_net,
        'classsr_srresnet': classSR_3class_srresnet_net,
        'classsr_rcan': classSR_3class_rcan_net,
        'fused_classsr_fsrcnn': fused_classSR_3class_fsrcnn_net,
    }
    for model_name in models:
        if 'fused' in model_name:
            model = models[model_name](models[model_name[5:]]())
        else:
            model = models[model_name]()
        # model.to('cuda:0')
        model.eval()
        # x = torch.arange(0, 4 * 3 * 32 * 32, dtype=torch.long).view(4, 3, 32, 32)
        x = torch.ones(4, 3, 32, 32)
        y = model(x)
        tracer = GraphTracer()
        graph = tracer.trace(model)
        name = model.__class__.__name__ if isinstance(model, torch.nn.Module) else model.__name__
        graph_module= GraphModule(tracer.root, graph, name)
        # gmodels = graph_module.named_modules()
        graph_drawer = FxGraphDrawer(graph_module, "brt_model")
        with open(BRT_CACHE_PATH/f"transformed_model/classsr/{model_name}_graph.svg", "wb") as f:
            f.write(graph_drawer.get_dot_graph().create_svg())
        print(f'{model_name} done.')
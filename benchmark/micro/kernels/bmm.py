import numpy as np

import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_executor

import torch
from torch import nn
from torch.utils import dlpack

def tune_bmm(B, M, N, P):

    network = "bmm"
    target = tvm.target.Target("cuda")
    dtype = "float32"
    log_file = f"tuner_record_{network}_{B}_{M}_{N}_{P}.json"

    bmm_func = lambda a, b: torch.bmm(a, b)
    a_shape = (B, M, N)
    b_shape = (B, N, P)
    a = torch.randn(a_shape).cuda()
    b = torch.randn(b_shape).cuda()

    # Extract tasks from the network
    print("Extract tasks...")
    mod, params = relay.frontend.from_pytorch(
        torch.jit.trace(
            bmm_func,
            example_inputs=(a, b),
        ),
        input_infos=[("a", a_shape), ("b", b_shape)],
    )
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

    print(type(mod))

    for idx, task in enumerate(tasks):
        print(
            "========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key)
        )
        print(task.compute_dag)

    def run_tuning():
        print("Begin tuning...")
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(
            repeat=1, min_repeat_ms=300, timeout=10
        )

        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=2000,  # change this to 20000 to achieve the best performance
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )

        tuner.tune(tune_option)

    run_tuning()

    # # Compile with the history best
    # print("Compile...")
    # with auto_scheduler.ApplyHistoryBest(log_file):
    #     with tvm.transform.PassContext(
    #         opt_level=3, config={"relay.backend.use_auto_scheduler": True}
    #     ):
    #         lib = relay.build(mod, target=target, params=params)

    # # Create graph executor
    # dev = tvm.device(str(target), 0)
    # module = graph_executor.GraphModule(lib["default"](dev))
    # a_tvm = tvm.nd.from_dlpack(dlpack.to_dlpack(torch.randn(a_shape).cuda()))
    # b_tvm = tvm.nd.from_dlpack(dlpack.to_dlpack(torch.randn(b_shape).cuda()))
    # module.set_input("a", a_tvm)
    # module.set_input("b", b_tvm)

    # # Evaluate
    # print("Evaluate inference time cost...")
    # print(module.benchmark(dev, repeat=3, min_repeat_ms=500))




all_expert_num = [2, 4, 8, 16]
all_bs_dict = {
    2: [160, 140, 140],
    4: [94, 76, 70],
    8: [59, 45, 34],
    16: [36, 26, 17],
}

for expert_num in all_expert_num:
    for bs in set(all_bs_dict[expert_num]):
        tune_bmm(expert_num, bs, 512, 1024)
        tune_bmm(expert_num, bs, 1024, 512)


class TunedBatchMalmul(nn.Module):
    """[b, m, n] @ [b, n, p]"""
    def __init__(self, b, m, n, p):
        super().__init__()
        target = tvm.target.Target("cuda")
        log_file = f"tuner_record_bmm_{b}_{m}_{n}_{p}.json" # TODO
        bmm_func = lambda a, b: torch.bmm(a, b)
        a_shape = (b, m, n)
        b_shape = (b, n, p)
        a = torch.empty(a_shape).cuda()
        b = torch.empty(b_shape).cuda()
        mod, params = relay.frontend.from_pytorch(
            torch.jit.trace(
                bmm_func,
                example_inputs=(a, b),
            ),
            input_infos=[("a", a_shape), ("b", b_shape)],
        )
        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(
                opt_level=3, config={"relay.backend.use_auto_scheduler": True}
            ):
                lib = relay.build(mod, target=target, params=params)
        dev = tvm.device(str(target), 0)
        self.func = graph_executor.GraphModule(lib["default"](dev))

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        """a: [b, m, n]; b: [b, n, p]"""
        a_tvm = tvm.nd.from_dlpack(dlpack.to_dlpack(a))
        b_tvm = tvm.nd.from_dlpack(dlpack.to_dlpack(b))
        self.func.set_input("a", a_tvm)
        self.func.set_input("b", b_tvm)
        self.func.run()
        out_tvm = self.func.get_output(0)
        out = dlpack.from_dlpack(out_tvm.to_dlpack()).squeeze()
        return out
        
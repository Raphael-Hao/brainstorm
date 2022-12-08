import torch
from torch.utils.benchmark import Timer

from brt.jit import make_jit_kernel
from brt.jit.tvm import TVMTuner
from brt.jit.codegen import ModuleKernel

all_bs = [
    # 2,
    # 4,
    # 8,
    # 16,
    # 32,
    # 64,
    # 128,
    416,
    224,
    320,
    # 512,
]

in_out_features = [
    [768, 3072],
    [3072, 768]
]

for bs in all_bs:
    for in_features, out_features in in_out_features:
        input_infos = {"input_0": (bs, in_features)}
        output_infos = {"output_0": (bs, out_features)}
        parameters = {
            "in_features": in_features,
            "out_features": out_features,
        }
        kernel_name = f"Linear_{bs}_{in_features}_{out_features}"

        linear = torch.nn.Linear(in_features, out_features).eval()

        tvm_tuner = TVMTuner()
        tvm_tuner.import_pt_netlet(
            "LinearBias",
            "forward",
            linear,
            input_infos,
            output_infos,
            parameters,
            # log_fname,
        )

        print(f"#### # Linerar {bs} {in_features} {out_features}")
        if tvm_tuner.tune_log_file.exists():
            with open(str(tvm_tuner.tune_log_file)) as f:
                num_trials = len(f.readlines())
            if num_trials < 2000:
                print("#### Find incomplete record, continue")
                tvm_tuner.task_scheduler.load_log_file = str(tvm_tuner.tune_log_file)
            else:
                print("#### Find tuned kernel, pass")
        else:
            print("#### Start tuning kernel")
        # tvm_tuner.task_scheduler.load_log_file = str(tvm_tuner.tune_log_file)
        tvm_tuner.tune_netlet()
        tvm_tuner.insert_netlet_to_storage()

        # linear.cuda()
        # for rank in range(1, 11):
        #     x = torch.randn((bs, in_features)).cuda()
        #     y = torch.randn((bs, out_features)).cuda()
        #     linear_kernel = make_jit_kernel(
        #         modules = linear,
        #         sample_inputs=x,
        #         rank = rank,
        #     )
        #     time = Timer(
        #         stmt="y = torch.empty(oshape).cuda(); model(x, weight, y, bias)",
        #         setup="import torch",
        #         globals={"model":linear_kernel, "x": x, "y": y, "weight": linear.weight, "bias":linear.bias, "oshape": (bs, out_features)}
        #     ).timeit(1000).mean * 1e6
        #     print(f"{rank = }, {time}")

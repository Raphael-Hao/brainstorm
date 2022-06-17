# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.fx as fx
from brt.cpp import jit
from functorch.compile import aot_function, aot_module

__all__ = ["CUDACompiler"]


class CUDACompiler:
    @staticmethod
    def create_raw(source):
        torch.cuda.init()
        kernel_type, __ctx__ = jit.inject_source(source)

        if kernel_type == "global" or kernel_type == "horiz_fuse":

            def func(*inputs, extra=[]):
                jit.static_invoke(inputs, extra, __ctx__)

        elif kernel_type == "hetero_fuse":

            def func(*inputs, active_blocks=[]):
                jit.hetero_invoke(inputs, active_blocks, __ctx__)

        elif kernel_type == "homo_fuse":

            def func(shared_inputs, standalone_inputs, capacities=[]):
                jit.homo_invoke(shared_inputs, standalone_inputs, capacities, __ctx__)

        else:
            raise NotImplementedError
        return func

    @staticmethod
    def generate_kernel(keyword_dict, template: str):
        if keyword_dict is not None:
            for key in keyword_dict:
                template = template.replace("@%s@" % key, str(keyword_dict[key]))
        return CUDACompiler.create_raw(template)


""" example from functorch.compile for tvm
def _tvm_compile(
    fx_module, example_inputs, target=None, tuning_logfile=None, use_ansor_tuning=False
):
    import os

    import tvm
    from tvm import auto_scheduler, relay
    from tvm.contrib import graph_executor

    # Find the target and device for TVM.
    dev = tvm.cpu(0)
    if target is None:
        raise ValueError("Setup the TVM target correctly.")
    elif isinstance(target, str):
        if "cuda" in target:
            dev = tvm.cuda(0)
        target = tvm.target.Target(target)
    elif isinstance(target, tvm.target.target.Target):
        if "cuda" in target.keys:
            dev = tvm.cuda(0)

    # JIT the model and pass it to Torchscript to Relay frontend parser. TVM
    # tutorials suggest tracing instead of scripting. The main reason is to
    # avoid Pythonic computation to show up in JIT module. However, with Python
    # key tracing, AOT Autograd leads to simpler graphs. Therefore, we use
    # scripting here to retrieve the JIT module.
    jit_mod = torch.jit.script(fx_module)
    shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(example_inputs)]
    mod, params = relay.frontend.from_pytorch(jit_mod, shape_list)

    # TVM Autotuning
    if use_ansor_tuning:
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
        if tuning_logfile is None:
            log_file = f"{time.time()}.json"
        else:
            log_file = f"{tuning_logfile}.json"
        if len(tasks) != 0:
            tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
            tune_option = auto_scheduler.TuningOptions(
                num_measure_trials=20000,
                measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
                # early_stopping=1000,
                # verbose=2,
            )
            tuner.tune(tune_option)
    elif tuning_logfile is not None:
        log_file = f"{tuning_logfile}.json"

    if use_ansor_tuning or tuning_logfile is not None:
        assert os.path.exists(log_file)
        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(
                opt_level=3, config={"relay.backend.use_auto_scheduler": True}
            ):
                lib = relay.build(mod, target=target, params=params)
    else:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)

    # Get a graph executor graph module
    m = graph_executor.GraphModule(lib["default"](dev))

    def exec_tvm(*args):
        for idx, arg in enumerate(args, 0):
            if arg.dim() != 0:
                m.set_input(
                    f"inp_{idx}",
                    tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(arg.contiguous())),
                )
        m.run()
        outs = [
            torch.utils.dlpack.from_dlpack(m.get_output(i).to_dlpack())
            for i in range(m.get_num_outputs())
        ]
        return outs

    return exec_tvm


def tvm_compile(target, tuning_logfile=None, use_ansor_tuning=False):
    return partial(
        _tvm_compile,
        target=target,
        tuning_logfile=tuning_logfile,
        use_ansor_tuning=use_ansor_tuning,
    )
"""


def brt_compile(fx_module: fx.GraphModule, example_inputs, fuse_type):
    pass

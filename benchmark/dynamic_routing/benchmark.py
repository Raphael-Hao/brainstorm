# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import os
import sys
from typing import Iterator, List

import time
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from brt.runtime import BRT_LOG_PATH
from collections import OrderedDict

from brt.passes import (
    DeadPathEliminatePass,
    OnDemandMemoryPlanPass,
    PermanentPathFoldPass,
    PredictMemoryPlanPass,
    TracePass,
)
from brt.router import switch_router_mode
from brt.runtime.benchmark import (
    BenchmarkArgumentManager,
    Benchmarker,
    CUDATimer,
    MemoryStats,
)
from brt.runtime.memory_planner import pin_memory
from dynamic_A_config import config as A_config
from dynamic_B_config import config as B_config
from dynamic_C_config import config as C_config
from dynamic_raw_config import config as raw_config

sys.path.insert(0, ".")  # noqa: E402


import dl_lib.utils.comm as comm
import torch
from dl_lib.checkpoint import DetectionCheckpointer
from dl_lib.data import MetadataCatalog
from dl_lib.engine import CustomizedTrainer, default_argument_parser, default_setup
from dl_lib.evaluation import (
    CityscapesEvaluator,
    DatasetEvaluators,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
)
from dl_lib.modeling import SemanticSegmentorWithTTA
from net import build_model


def profile_v2(model: nn.Module, data_collection: List[torch.Tensor], vendor: str):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=1),
        # schedule=torch.profiler.schedule(wait=2, warmup=2, active=5),
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"./results/{vendor}/locality"
        ),
        record_shapes=True,
    ) as p:
        print("os is: ", os.getcwd())
        print("start profiling")
        print("path is: ", f"./results/{vendor}/locality")
        with torch.inference_mode():
            for step, data in enumerate(data_collection):
                torch.cuda.empty_cache()
                model(data)
                p.step()
                if step + 1 >= 10:
                    break


class Trainer(CustomizedTrainer):
    """
    We use the "CustomizedTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type == "cityscapes":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesEvaluator(dataset_name)
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if hasattr(cfg, "EVALUATORS"):
            for evaluator in cfg.EVALUATORS:
                evaluator_list.append(
                    evaluator(dataset_name, True, output_folder, dump=False)
                )
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("dl_lib.trainer")
        # In the end of training, run an evaluation with TTA
        logger.info("Running inference with test-time augmentation ...")
        evaluator_type = MetadataCatalog.get(cfg.DATASETS.TEST[0]).evaluator_type
        if evaluator_type == "sem_seg":
            model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def test_argument_parser():
    parser = default_argument_parser()
    parser.add_argument(
        "--arch",
        choices=["raw", "A", "B", "C"],
        default="raw",
        help="choose model architecture",
    )
    parser.add_argument("--debug", action="store_true", help="use debug mode or not")
    bench_arg_manager = BenchmarkArgumentManager(parser)
    bench_arg_manager.add_item("liveness")
    bench_arg_manager.add_item("memory_plan")
    bench_arg_manager.add_item("dce_memory_plan")

    parser.add_argument(
        "--memory-mode", choices=["predict", "on_demand"], default="on_demand"
    )

    return parser


def main(args):
    if args.arch == "raw":
        config = raw_config
    elif args.arch == "A":
        config = A_config
    elif args.arch == "B":
        config = B_config
    elif args.arch == "C":
        config = C_config
    else:
        raise ValueError("Invalid arch: {}".format(args.arch))

    config.merge_from_list(args.opts)
    cfg, _logger = default_setup(config, args)

    model = build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    model.cuda()
    torch.cuda.synchronize()

    res = Trainer.test(cfg, model)
    torch.cuda.empty_cache()

    benchmarker = Benchmarker()

    def liveness_benchmark():
        timer = CUDATimer(repeat=5)

        backbone_input = model.backbone_input

        naive_backbone = model.backbone

        naive_backbone = switch_router_mode(naive_backbone, False).eval()

        trace_pass = TracePass(naive_backbone)
        trace_pass.run_on_graph()
        naive_backbone = trace_pass.finalize()
        # MemoryStats.reset_cuda_stats()
        timer.execute(lambda: naive_backbone(backbone_input), "naive")
        # MemoryStats.print_cuda_stats()
        # naive_out= naive_backbone(backbone_input)

        eliminate_pass = DeadPathEliminatePass(naive_backbone, runtime_load=1)
        eliminate_pass.run_on_graph()
        new_backbone = eliminate_pass.finalize()

        timer.execute(lambda: new_backbone(backbone_input), "dead_path_eliminated")
        # eliminate_pass_output=new_backbone(backbone_input)

        permanent_pass = PermanentPathFoldPass(new_backbone, upper_perm_load=500)
        permanent_pass.run_on_graph()
        new_backbone = permanent_pass.finalize()

        timer.execute(lambda: new_backbone(backbone_input), "all_liveness_pass")
        # all_liveness_output=new_backbone(backbone_input)

        if args.debug:
            from torch.fx.passes.graph_drawer import FxGraphDrawer

            graph_drawer = FxGraphDrawer(new_backbone, "new_backbone")
            with open("new_backbone.svg", "wb") as f:
                f.write(graph_drawer.get_dot_graph().create_svg())

            model.backbone = new_backbone
            new_res = Trainer.test(cfg, model)

    benchmarker.add_benchmark("liveness", liveness_benchmark)

    def dce_memroy_plan_benchmark():
        # input_list=[]
        timer = CUDATimer(repeat=5)
        backbone_input = model.backbone_input.detach().cuda()
        # for i in range(12):
        #     input_list.append(backbone_input)
        backbone = switch_router_mode(model.backbone, False).eval()
        MemoryStats.reset_cuda_stats()
        timer.execute(lambda: backbone(backbone_input), "naive0")
        MemoryStats.print_cuda_stats()

        trace_pass = TracePass(backbone)
        trace_pass.run_on_graph()
        backbone = trace_pass.finalize()
        MemoryStats.reset_cuda_stats()
        timer.execute(lambda: backbone(backbone_input), "naive")
        MemoryStats.print_cuda_stats()
        # naive_output= backbone(backbone_input)
        eliminate_pass = DeadPathEliminatePass(backbone, runtime_load=1)
        eliminate_pass.run_on_graph()
        new_backbone = eliminate_pass.finalize()
        timer.execute(lambda: new_backbone(backbone_input), "dead_path_eliminated")
        permanent_pass = PermanentPathFoldPass(new_backbone, upper_perm_load=12)
        permanent_pass.run_on_graph()
        backbone = permanent_pass.finalize()
        timer.execute(lambda: backbone(backbone_input), "all_liveness_pass")
        backbone = pin_memory(backbone.cpu())
        pass_name = None
        memory_plan_pass = None
        if args.memory_mode == "predict":
            pass_name = "PredictorMemoryPlanPass"
            memory_plan_pass = PredictMemoryPlanPass(backbone, 500, is_grouping=True)
        elif args.memory_mode == "on_demand":
            pass_name = "OnDemandMemoryPlanPass"
            memory_plan_pass = OnDemandMemoryPlanPass(backbone, is_grouping=True)
        memory_plan_pass.run_on_graph()
        initial_loaders, new_backbone = memory_plan_pass.finalize()
        # print(new_backbone.code)
        torch.cuda.empty_cache()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.reset_peak_memory_stats()
        for loader in initial_loaders:
            loader()
        torch.cuda.synchronize()
        MemoryStats.reset_cuda_stats()
        # profile(lambda: new_backbone(backbone_input))
        # mem_output1 = new_backbone(backbone_input)
        MemoryStats.reset_cuda_stats()
        timer.execute(lambda: new_backbone(backbone_input), pass_name)
        MemoryStats.print_cuda_stats()
        # profile_v2(new_backbone,[backbone_input], "new_dcr_mem_backbone")
        mem_output = new_backbone(backbone_input)

    def memroy_plan_benchmark():
        timer = CUDATimer(repeat=5)
        backbone_input = model.backbone_input.detach().cuda()

        backbone = switch_router_mode(model.backbone, False).eval()

        MemoryStats.reset_cuda_stats()

        timer.execute(lambda: backbone(backbone_input), "naive")

        # MemoryStats.print_cuda_stats()

        backbone = pin_memory(backbone.cpu())
        pass_name = None
        memory_plan_pass = None
        if args.memory_mode == "predict":
            pass_name = "PredictorMemoryPlanPass"
            memory_plan_pass = PredictMemoryPlanPass(backbone, 500, is_grouping=True)
        elif args.memory_mode == "on_demand":
            pass_name = "OnDemandMemoryPlanPass"
            memory_plan_pass = OnDemandMemoryPlanPass(backbone, is_grouping=True)
        memory_plan_pass.run_on_graph()
        initial_loaders, new_backbone = memory_plan_pass.finalize()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.reset_peak_memory_stats()
        for loader in initial_loaders:
            loader()
        torch.cuda.synchronize()
        MemoryStats.reset_cuda_stats()
        # profile_v2(new_backbone,input_list, "only_mem_backbone")
        # profile(lambda: new_backbone(backbone_input))
        timer.execute(lambda: new_backbone(backbone_input), pass_name)
        # MemoryStats.print_cuda_stats()

    benchmarker.add_benchmark("dce_memory_plan", dce_memroy_plan_benchmark)
    benchmarker.add_benchmark("memory_plan", memroy_plan_benchmark)

    benchmarker.benchmarking(args.benchmark)


if __name__ == "__main__":
    args = test_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)

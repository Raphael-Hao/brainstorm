# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from dynamic_raw_config import config as raw_config
from dynamic_A_config import config as A_config
from dynamic_B_config import config as B_config
from dynamic_C_config import config as C_config

from brt.router import switch_router_mode
from brt.passes import (
    DeadPathEliminatePass,
    PermanentPathFoldPass,
    MemoryPlanPass,
    OnDemandMemoryPlanPass,
    PredictMemoryPlanPass,
)
from brt.runtime.memory_planner import pin_memory
from brt.runtime.benchmark import (
    BenchmarkArgumentManager,
    Benchmarker,
    CUDATimer,
    MemoryStats,
    profile,
)

"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in dl_lib.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use dl_lib as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import glob
import logging
import os
import re
import sys

sys.path.insert(0, ".")  # noqa: E402

from collections import OrderedDict
import torch

import dl_lib.utils.comm as comm
from dl_lib.checkpoint import DetectionCheckpointer
from dl_lib.data import MetadataCatalog
from dl_lib.engine import (
    CustomizedTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from dl_lib.evaluation import (
    CityscapesEvaluator,
    DatasetEvaluators,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from dl_lib.modeling import SemanticSegmentorWithTTA
from net import build_model


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
    bench_arg_manager.add_item("preload")
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
    cfg, logger = default_setup(config, args)

    model = build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    model.cuda()
    torch.cuda.synchronize()

    res = Trainer.test(cfg, model)
    torch.cuda.empty_cache()

    if args.debug:
        timer = CUDATimer(repeat=5)
        backbone_input = model.backbone_input.detach().cuda()

        backbone = switch_router_mode(model.backbone, False).eval()

        MemoryStats.reset_cuda_stats()

        timer.execute(lambda: backbone(backbone_input), "naive")

        MemoryStats.print_cuda_stats()

        backbone = pin_memory(backbone.cpu())

        memory_plan_pass = OnDemandMemoryPlanPass(backbone)
        # memory_plan_pass = PredictMemoryPlanPass(backbone, 1)
        memory_plan_pass.run_on_graph()
        new_backbone = memory_plan_pass.finalize()
        print(new_backbone.code)
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        MemoryStats.reset_cuda_stats()
        timer.execute(lambda: new_backbone(backbone_input), "on_demand_load")
        MemoryStats.print_cuda_stats()

    # model.eval()
    # input = torch.randn(1, 3, 1024, 2048).cuda()
    # outputs = model.backbone(input)
    # print(outputs)
    benchmarker = Benchmarker()

    def liveness_benchmark():
        timer = CUDATimer()
        timer.set_iterations(100)

        backbone_input = model.backbone_input

        naive_backbone = model.backbone

        naive_backbone = switch_router_mode(naive_backbone, False).eval()

        timer.execute(lambda: naive_backbone(backbone_input), "naive")

        eliminate_pass = DeadPathEliminatePass(naive_backbone, runtime_load=1)
        eliminate_pass.run_on_graph()
        new_backbone = eliminate_pass.finalize()

        timer.execute(lambda: new_backbone(backbone_input), "dead_path_eliminated")

        permanent_pass = PermanentPathFoldPass(new_backbone, upper_perm_load=500)
        permanent_pass.run_on_graph()
        new_backbone = permanent_pass.finalize()

        timer.execute(lambda: new_backbone(backbone_input), "all_liveness_pass")

        if args.debug:
            from torch.fx.passes.graph_drawer import FxGraphDrawer

            graph_drawer = FxGraphDrawer(new_backbone, "new_backbone")
            with open("new_backbone.svg", "wb") as f:
                f.write(graph_drawer.get_dot_graph().create_svg())

            model.backbone = new_backbone
            new_res = Trainer.test(cfg, model)

    benchmarker.add_benchmark("liveness", liveness_benchmark)

    def memroy_plan_benchmark():
        timer = CUDATimer()
        timer.set_iterations(100)

        backbone_input = model.backbone_input

        naive_backbone = model.backbone

        naive_backbone = switch_router_mode(naive_backbone, False).eval()

        preload_pass = MemoryPlanPass(model.backbone)
        preload_pass.run_on_graph()

    benchmarker.add_benchmark("memory_plan", memroy_plan_benchmark)

    benchmarker.benchmarking(args.benchmark)


if __name__ == "__main__":
    args = test_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)

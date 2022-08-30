# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# from dynamic_raw_config import config
from dynamic_A_config import config

# from dynamic_B_config import config
# from dynamic_C_config import config

from brt.trace.graph import GraphTracer
from torch.fx.graph_module import GraphModule
from torch.fx.passes.graph_drawer import FxGraphDrawer
from brt.router import ScatterRouter, GatherRouter
from brt.passes import get_pass

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
from net_brt import build_model


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
        "--start-iter", type=int, default=0, help="start iter used to test"
    )
    parser.add_argument(
        "--end-iter", type=int, default=None, help="end iter used to test"
    )
    parser.add_argument("--debug", action="store_true", help="use debug mode or not")
    parser.add_argument("--trace", action="store_true", help="trace the model or not")

    return parser


def main(args):
    config.merge_from_list(args.opts)
    cfg, logger = default_setup(config, args)
    if args.debug:
        batches = int(cfg.SOLVER.IMS_PER_BATCH / 8 * args.num_gpus)
        if cfg.SOLVER.IMS_PER_BATCH != batches:
            cfg.SOLVER.IMS_PER_BATCH = batches
            logger.warning("SOLVER.IMS_PER_BATCH is changed to {}".format(batches))

    model = build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )

    res = Trainer.test(cfg, model)
    # model.eval()
    # input = torch.randn(1, 3, 1024, 2048).cuda()
    # outputs = model.backbone(input)
    # print(outputs)

    if args.trace:
        pass_cls = get_pass("dead_path_eliminate")
        eliminate_pass = pass_cls(model.backbone)
        eliminate_pass.run_on_graph()
        new_backbone = eliminate_pass.finalize()

if __name__ == "__main__":
    args = test_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)

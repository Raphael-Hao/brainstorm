# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import os
import sys
from collections import OrderedDict

from brt.router import switch_router_mode
from brt.runtime.benchmark import CUDATimer
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

    _res = Trainer.test(cfg, model)
    torch.cuda.empty_cache()

    timer = CUDATimer(repeat=5, loop=100)

    backbone_input = model.backbone_input

    naive_backbone = model.backbone

    naive_backbone.eval()

    timer.execute(lambda: naive_backbone(backbone_input), "capture mode")

    naive_backbone = switch_router_mode(naive_backbone, False).eval()

    timer.execute(lambda: naive_backbone(backbone_input), "normal mode")


if __name__ == "__main__":
    args = test_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)

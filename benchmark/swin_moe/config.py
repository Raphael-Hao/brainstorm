# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'no'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8
# TSV FILES
_C.DATA.TRAIN_TSV_PREFIX = 'train'
_C.DATA.TEST_TSV_PREFIX = 'val'
_C.DATA.TRAIN_TSV_LIST = []
_C.DATA.TEST_TSV_LIST = []
_C.DATA.CHUNK_MODE = False
_C.DATA.FINETUNE_MAPFILE = "map22kto1k.txt"
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'swin'
# Model name
_C.MODEL.NAME = 'swin_tiny_patch4_window7_224'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''

_C.MODEL.PLACEMENT = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1
# RPE INTERPOLATION
_C.MODEL.RPE_INTERPOLATION = 'bicubic'

# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True

# Swin Transformer MoE parameters
_C.MODEL.SWIN_MOE = CN()
_C.MODEL.SWIN_MOE.PATCH_SIZE = 4
_C.MODEL.SWIN_MOE.IN_CHANS = 3
_C.MODEL.SWIN_MOE.EMBED_DIM = 96
_C.MODEL.SWIN_MOE.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN_MOE.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN_MOE.WINDOW_SIZE = 7
_C.MODEL.SWIN_MOE.MLP_RATIO = 4.
_C.MODEL.SWIN_MOE.QKV_BIAS = True
_C.MODEL.SWIN_MOE.QK_SCALE = None
_C.MODEL.SWIN_MOE.APE = False
_C.MODEL.SWIN_MOE.PATCH_NORM = True
_C.MODEL.SWIN_MOE.MOE_BLOCKS = [[-1], [-1], [-1], [-1]]
_C.MODEL.SWIN_MOE.NUM_LOCAL_EXPERTS = 1
_C.MODEL.SWIN_MOE.TOP_VALUE = 2
_C.MODEL.SWIN_MOE.CAPACITY_FACTOR = 1.25
_C.MODEL.SWIN_MOE.NORMALIZE_GATE = True
_C.MODEL.SWIN_MOE.BATCH_PRIORITIZED_ROUTING = False
_C.MODEL.SWIN_MOE.VITMOE_LOSS = False
_C.MODEL.SWIN_MOE.USE_GLOBAL_LOSS = False
_C.MODEL.SWIN_MOE.MLP_FC2_BIAS = True
_C.MODEL.SWIN_MOE.MOE_DROP = 0.0
_C.MODEL.SWIN_MOE.INIT_STD = 0.02
_C.MODEL.SWIN_MOE.AUX_LOSS_SCALES = [[1.0], [1.0], [1.0], [1.0]]

# Swin MLP parameters
_C.MODEL.SWIN_MLP = CN()
_C.MODEL.SWIN_MLP.PATCH_SIZE = 4
_C.MODEL.SWIN_MLP.IN_CHANS = 3
_C.MODEL.SWIN_MLP.EMBED_DIM = 96
_C.MODEL.SWIN_MLP.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN_MLP.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN_MLP.WINDOW_SIZE = 7
_C.MODEL.SWIN_MLP.MLP_RATIO = 4.
_C.MODEL.SWIN_MLP.APE = False
_C.MODEL.SWIN_MLP.PATCH_NORM = True

# SWIN V2 RPEFC2
_C.MODEL.SWIN_V2_MOE = CN()
_C.MODEL.SWIN_V2_MOE.PATCH_SIZE = 4
_C.MODEL.SWIN_V2_MOE.IN_CHANS = 3
_C.MODEL.SWIN_V2_MOE.EMBED_DIM = 96
_C.MODEL.SWIN_V2_MOE.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN_V2_MOE.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN_V2_MOE.WINDOW_SIZE = 7
_C.MODEL.SWIN_V2_MOE.MLP_RATIO = 4.
_C.MODEL.SWIN_V2_MOE.LN_EPS = 1e-6
_C.MODEL.SWIN_V2_MOE.APE = False
_C.MODEL.SWIN_V2_MOE.PATCH_NORM = True
_C.MODEL.SWIN_V2_MOE.INIT_VALUES = [1e-5]
_C.MODEL.SWIN_V2_MOE.INIT_STD = 0.02
_C.MODEL.SWIN_V2_MOE.ENDNORM_INTERVAL = -1
_C.MODEL.SWIN_V2_MOE.CHECKPOINT_BLOCKS = [255, 255, 255, 255]
_C.MODEL.SWIN_V2_MOE.RELATIVE_COORDS_TABLE_TYPE = 'norm8_log'
_C.MODEL.SWIN_V2_MOE.RPE_HIDDEN_DIM = 512
_C.MODEL.SWIN_V2_MOE.MLPFP32_LAYER_BLOCKS = [[-1], [-1], [-1], [-1]]
_C.MODEL.SWIN_V2_MOE.LN_TYPE = 'fp32'  # normal, fp32, relax
_C.MODEL.SWIN_V2_MOE.RPE_OUTPUT_TYPE = 'sigmoid'
_C.MODEL.SWIN_V2_MOE.ATTN_TYPE = 'cosine_mh'
_C.MODEL.SWIN_V2_MOE.POSTNORM = True
_C.MODEL.SWIN_V2_MOE.MLP_TYPE = 'normal'
_C.MODEL.SWIN_V2_MOE.PATCH_EMBED_TYPE = 'normal'
_C.MODEL.SWIN_V2_MOE.PATCH_MERGE_TYPE = 'normal'
_C.MODEL.SWIN_V2_MOE.DROPPATH_RULE = 'linear'
_C.MODEL.SWIN_V2_MOE.STRID16 = False
_C.MODEL.SWIN_V2_MOE.STRID16_GLOBAL = False
_C.MODEL.SWIN_V2_MOE.MOE_BLOCKS = [[-1], [-1], [-1], [-1]]
_C.MODEL.SWIN_V2_MOE.NUM_LOCAL_EXPERTS = 1
_C.MODEL.SWIN_V2_MOE.NUM_LOCAL_EXPERTS_IN_CKPT = None
_C.MODEL.SWIN_V2_MOE.TOP_VALUE = 2
_C.MODEL.SWIN_V2_MOE.CAPACITY_FACTOR = 1.25
_C.MODEL.SWIN_V2_MOE.NORMALIZE_GATE = True
_C.MODEL.SWIN_V2_MOE.BATCH_PRIORITIZED_ROUTING = False
_C.MODEL.SWIN_V2_MOE.VITMOE_LOSS = False
_C.MODEL.SWIN_V2_MOE.USE_GLOBAL_LOSS = False
_C.MODEL.SWIN_V2_MOE.USE_NOISE = True
_C.MODEL.SWIN_V2_MOE.MLP_FC2_BIAS = True
_C.MODEL.SWIN_V2_MOE.MOE_DROP = 0.0
_C.MODEL.SWIN_V2_MOE.MOE_POST_SCORE = True
_C.MODEL.SWIN_V2_MOE.AUX_LOSS_SCALES = [[1.0], [1.0], [1.0], [1.0]]
_C.MODEL.SWIN_V2_MOE.RM_K_BIAS = True
_C.MODEL.SWIN_V2_MOE.SKIP_MOE_DROPPATH = False
_C.MODEL.SWIN_V2_MOE.HEAD_2FC = False
_C.MODEL.SWIN_V2_MOE.NORM_IN_MOE = False

# SWIN V2 RPEFC2
_C.MODEL.SWIN_V2_MOE_BASE_LAYER = CN()
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.PATCH_SIZE = 4
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.IN_CHANS = 3
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.EMBED_DIM = 96
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.WINDOW_SIZE = 7
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.MLP_RATIO = 4.
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.LN_EPS = 1e-6
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.APE = False
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.PATCH_NORM = True
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.INIT_VALUES = 1e-5
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.INIT_STD = 0.02
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.ENDNORM_INTERVAL = -1
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.CHECKPOINT_BLOCKS = [255, 255, 255, 255]
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.RELATIVE_COORDS_TABLE_TYPE = 'norm8_log'
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.RPE_HIDDEN_DIM = 512
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.MLPFP32_LAYER_BLOCKS = [[-1], [-1], [-1], [-1]]
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.LN_TYPE = 'fp32'  # normal, fp32, relax
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.RPE_OUTPUT_TYPE = 'sigmoid'
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.ATTN_TYPE = 'cosine_mh'
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.POSTNORM = True
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.MLP_TYPE = 'normal'
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.PATCH_EMBED_TYPE = 'normal'
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.PATCH_MERGE_TYPE = 'normal'
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.DROPPATH_RULE = 'linear'
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.STRID16 = False
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.STRID16_GLOBAL = False
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.MOE_BLOCKS = [[-1], [-1], [-1], [-1]]
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.NUM_LOCAL_EXPERTS = 1
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.TOP_VALUE = 2
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.CAPACITY_FACTOR = 1.25
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.NORMALIZE_GATE = True
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.BATCH_PRIORITIZED_ROUTING = False
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.VITMOE_LOSS = False
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.USE_GLOBAL_LOSS = False
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.USE_NOISE = True
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.MLP_FC2_BIAS = True
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.MOE_DROP = 0.0
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.MOE_POST_SCORE = True
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.AUX_LOSS_SCALES = [[1.0], [1.0], [1.0], [1.0]]
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.RM_K_BIAS = True
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.SKIP_MOE_DROPPATH = False
_C.MODEL.SWIN_V2_MOE_BASE_LAYER.HEAD_2FC = False

# SWIN V2 RPEFC2
_C.MODEL.SWIN_V2_MOE_AM = CN()
_C.MODEL.SWIN_V2_MOE_AM.PATCH_SIZE = 4
_C.MODEL.SWIN_V2_MOE_AM.IN_CHANS = 3
_C.MODEL.SWIN_V2_MOE_AM.EMBED_DIM = 96
_C.MODEL.SWIN_V2_MOE_AM.BLOCK_NAMES = [['a', 'm', 'as', 'm'],
                                       ['a', 'm', 'as', 'm'],
                                       ['a', 'm', 'as', 'm', 'a', 'm', 'as', 'm', 'a', 'm', 'as', 'm'],
                                       ['a', 'm', 'as', 'm']]
_C.MODEL.SWIN_V2_MOE_AM.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN_V2_MOE_AM.WINDOW_SIZE = 7
_C.MODEL.SWIN_V2_MOE_AM.MLP_RATIO = 4.
_C.MODEL.SWIN_V2_MOE_AM.LN_EPS = 1e-6
_C.MODEL.SWIN_V2_MOE_AM.APE = False
_C.MODEL.SWIN_V2_MOE_AM.PATCH_NORM = True
_C.MODEL.SWIN_V2_MOE_AM.INIT_VALUES = 1e-5
_C.MODEL.SWIN_V2_MOE_AM.INIT_STD = 0.02
_C.MODEL.SWIN_V2_MOE_AM.ENDNORM_INTERVAL = -1
_C.MODEL.SWIN_V2_MOE_AM.CHECKPOINT_BLOCKS = [255, 255, 255, 255]
_C.MODEL.SWIN_V2_MOE_AM.RELATIVE_COORDS_TABLE_TYPE = 'norm8_log'
_C.MODEL.SWIN_V2_MOE_AM.RPE_HIDDEN_DIM = 512
_C.MODEL.SWIN_V2_MOE_AM.MLPFP32_LAYER_BLOCKS = [[-1], [-1], [-1], [-1]]
_C.MODEL.SWIN_V2_MOE_AM.LN_TYPE = 'fp32'  # normal, fp32, relax
_C.MODEL.SWIN_V2_MOE_AM.RPE_OUTPUT_TYPE = 'sigmoid'
_C.MODEL.SWIN_V2_MOE_AM.ATTN_TYPE = 'cosine_mh'
_C.MODEL.SWIN_V2_MOE_AM.POSTNORM = True
_C.MODEL.SWIN_V2_MOE_AM.MLP_TYPE = 'normal'
_C.MODEL.SWIN_V2_MOE_AM.PATCH_EMBED_TYPE = 'normal'
_C.MODEL.SWIN_V2_MOE_AM.PATCH_MERGE_TYPE = 'normal'
_C.MODEL.SWIN_V2_MOE_AM.DROPPATH_RULE = 'linear'
_C.MODEL.SWIN_V2_MOE_AM.STRID16 = False
_C.MODEL.SWIN_V2_MOE_AM.STRID16_GLOBAL = False
_C.MODEL.SWIN_V2_MOE_AM.MOE_BLOCKS = [[-1], [-1], [-1], [-1]]
_C.MODEL.SWIN_V2_MOE_AM.NUM_LOCAL_EXPERTS = 1
_C.MODEL.SWIN_V2_MOE_AM.TOP_VALUE = 2
_C.MODEL.SWIN_V2_MOE_AM.CAPACITY_FACTOR = 1.25
_C.MODEL.SWIN_V2_MOE_AM.NORMALIZE_GATE = True
_C.MODEL.SWIN_V2_MOE_AM.BATCH_PRIORITIZED_ROUTING = False
_C.MODEL.SWIN_V2_MOE_AM.VITMOE_LOSS = False
_C.MODEL.SWIN_V2_MOE_AM.USE_GLOBAL_LOSS = False
_C.MODEL.SWIN_V2_MOE_AM.USE_NOISE = True
_C.MODEL.SWIN_V2_MOE_AM.MLP_FC2_BIAS = True
_C.MODEL.SWIN_V2_MOE_AM.MOE_DROP = 0.0
_C.MODEL.SWIN_V2_MOE_AM.MOE_POST_SCORE = True
_C.MODEL.SWIN_V2_MOE_AM.AUX_LOSS_SCALES = [[1.0], [1.0], [1.0], [1.0]]
_C.MODEL.SWIN_V2_MOE_AM.RM_K_BIAS = True

# Vision Transformer parameters
_C.MODEL.VIT = CN()
_C.MODEL.VIT.PATCH_SIZE = 16
_C.MODEL.VIT.IN_CHANS = 3
_C.MODEL.VIT.EMBED_DIM = 768
_C.MODEL.VIT.DEPTH = 12
_C.MODEL.VIT.NUM_HEADS = 12
_C.MODEL.VIT.MLP_RATIO = 4.

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Total lr
_C.TRAIN.TOTAL_LR = -1.
_C.TRAIN.TOTAL_WARMUP_LR = -1.
_C.TRAIN.TOTAL_MIN_LR = -1.
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False
# MoE Aux Loss Weight
_C.TRAIN.MOE_LOSS_WEIGHT = 0.1
# MoE Grad Scale
_C.TRAIN.MOE_GRAD_SCALE = 1.0
# Save Step
_C.TRAIN.SAVE_STEP = -1
_C.TRAIN.START_STEP = 0

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# FT_MOE_DEV # todo: changed in 2022/03/28
_C.TRAIN.FT_MOE_DEV = CN()
_C.TRAIN.FT_MOE_DEV.KEEP_LIST = []
_C.TRAIN.FT_MOE_DEV.SKIP_TRAIN_LIST = ['_moe_layer']
# MoE cosine gates
_C.TRAIN.COSINE_GATE_T_START = 0.5
_C.TRAIN.COSINE_GATE_T_END = 0.01

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True
# Whether to shuffle in test (maybe useful for MoE)
_C.TEST.SHUFFLE = True  # todo: change to True in 2022/03/02

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
_C.SINGLE_GPU_EVAL = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0

# DEEPSPEED
_C.ENABLE_DEEPSPEED = False
_C.DPFP16 = False
_C.ZERO_OPT = 0


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.zip:
        config.DATA.ZIP_MODE = True
    if args.cache_mode:
        config.DATA.CACHE_MODE = args.cache_mode
    if args.pretrained:
        config.MODEL.PRETRAINED = args.pretrained
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.placement:
        config.MODEL.PLACEMENT = args.placement
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True
    if args.throughput:
        config.THROUGHPUT_MODE = True

    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    # deepspeed
    config.ZERO_OPT = args.zero_opt

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config

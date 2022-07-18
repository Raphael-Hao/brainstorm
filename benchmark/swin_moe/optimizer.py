# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import json
from torch import optim as optim


def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    has_decay_names = []
    no_decay_names = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
            no_decay_names.append(name)
        else:
            has_decay.append(param)
            has_decay_names.append(name)
    print(f"no_decay_names: {no_decay_names}")
    print(f"has_decay_names: {has_decay_names}")
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def build_optimizer_ft_moe(config, model, keep_list=(), skip_train_list=()):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_wd_lr_ft_moe(config, model, skip, skip_keywords, keep_list, skip_train_list)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    return optimizer


def set_wd_lr_ft_moe(config, model, skip_list=(), skip_keywords=(), keep_list=(), skip_train_list=()):
    has_decay = []
    no_decay = []
    skip_train = []

    has_decay_names = []
    no_decay_names = []
    skip_train_names = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        # check weight decay
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            this_weight_decay = False
        else:
            this_weight_decay = True
        # check skip
        this_need_train = True
        if check_keywords_in_name(name, skip_train_list) and not check_keywords_in_name(name, keep_list):
            if name.startswith("layers"):
                i_layer = int(name.split('.')[1])
                if name.split('.')[2].startswith("blocks"):
                    i_block = int(name.split('.')[3])
                    if config.MODEL.TYPE == 'swinv2_moe':
                        if i_block in config.MODEL.SWIN_V2_MOE.MOE_BLOCKS[i_layer]:
                            this_need_train = False

        if this_need_train:
            if this_weight_decay:
                has_decay.append(param)
                has_decay_names.append(name)
            else:
                no_decay.append(param)
                no_decay_names.append(name)
        else:
            skip_train.append(param)
            skip_train_names.append(name)

    parameter_group_names = {
        'has_decay': {'params': has_decay_names},
        'no_decay': {'params': no_decay_names, 'weight_decay': 0.},
        'skip_train': {'params': skip_train_names, 'weight_decay': 0., "lr": 0.0}

    }
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.},
            {'params': skip_train, 'weight_decay': 0., "lr": 0.0}]

# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import json
from scipy import interpolate
from torch._six import inf
from timm.models.layers import trunc_normal_
from custom_amp import GradScaler


def load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger, loss_scaler=None):
    global_rank = dist.get_rank()
    logger.info(f"==============> Rank[{global_rank}] Resuming form {config.MODEL.RESUME}....................")
    print(f"==============> Rank[{global_rank}] Resuming form {config.MODEL.RESUME}....................")
    if loss_scaler is not None:
        if config.SINGLE_GPU_EVAL:
            if config.MODEL.RESUME.endswith(f'.rank{global_rank}'):
                pass
            elif config.MODEL.RESUME.endswith(f'.pth'):
                ckpt_dir = '/'.join(config.MODEL.RESUME.split('/')[:-1])
                dir_list = os.listdir(ckpt_dir)
                rank_files = [f for f in dir_list if '.pth' in f]
                
                ranks = sorted([int(f.split('.pth.rank')[1]) for f in rank_files])
                assert len(ranks) -1 == ranks[-1]
                num_rank = len(ranks)
                local_expert_ckpt = config.MODEL.SWIN_V2_MOE.NUM_LOCAL_EXPERTS_IN_CKPT
                checkpoint = []
                for i in range(num_rank):
                    checkpoint.append(torch.load(config.MODEL.RESUME+f'.rank{i}', map_location='cpu'))
                if local_expert_ckpt < 0:
                    expert_span = -local_expert_ckpt
                    new_entry_weights = {}
                    for expert_main_rank in range(0, num_rank, expert_span):
                        expert_entries = [k for k in checkpoint[expert_main_rank]['model'].keys() \
                            if '._moe_layer.experts.' in k]
                        for entry in expert_entries:
                            tensors_to_concat = []
                            
                            entry_splits = entry.split('._moe_layer.experts.') 
                            varname = entry_splits[1].split('.')[1]
                            new_entry_name = entry_splits[0] + '._moe_layer.experts.0.' + varname
                            for rank_id in range(expert_main_rank, expert_main_rank+expert_span):
                                tensor_name = entry_splits[0] + \
                                        '._moe_layer.experts.0.' + \
                                        varname
                                        # f'{rank_id%expert_span}.'+ \
                                tensors_to_concat.append(checkpoint[rank_id]['model'][tensor_name])
                            if new_entry_name not in new_entry_weights:
                                new_entry_weights[new_entry_name] = []
                            new_entry_weights[new_entry_name].append(torch.cat(tensors_to_concat,0))
                    state_dict = {}
                    for entry in checkpoint[0]['model'].keys():
                        if '._moe_layer.experts' in entry:
                            continue
                        else:
                            state_dict[entry] = checkpoint[0]['model'][entry]
                    for entry in new_entry_weights.keys():
                        if '_bias' in entry:
                            state_dict[entry] = torch.stack(
                                [torch.unsqueeze(x, dim=0) for x in new_entry_weights[entry]])
                        else:
                            state_dict[entry] = torch.stack(new_entry_weights[entry])
                else:
                    raise NotImplementedError(f'local_expert_ckpt > 0 has not implemented yet')
                    
        else:
            if config.MODEL.RESUME.endswith(f'.rank{global_rank}'):
                pass
            elif config.MODEL.RESUME.endswith(f'.pth'):
                config.defrost()
                config.MODEL.RESUME = config.MODEL.RESUME + f'.rank{global_rank}'
                config.freeze()
                logger.info(f"===> Rank[{global_rank}] Re-formatting checkpoint to {config.MODEL.RESUME}......")
                print(f"===> Rank[{global_rank}] Re-formatting checkpoint to {config.MODEL.RESUME}......")
            else:
                raise NotImplementedError(f"{config.MODEL.RESUME} file error...")

            checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')

            # todo: ze check, modified in 2021/09/06
            state_dict = checkpoint['model']
        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]
        # delete relative_coords_table since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
        for k in relative_position_index_keys:
            del state_dict[k]
        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        msg = model_without_ddp.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        if type(checkpoint) == dict:
            del checkpoint['model']
        elif type(checkpoint) == list:
            for x in checkpoint:
                del x['model']

        max_accuracy = 0.0
        if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            del checkpoint['optimizer']
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            config.defrost()
            if 'step' in checkpoint:
                config.TRAIN.START_EPOCH = checkpoint['epoch']
                config.TRAIN.START_STEP = checkpoint['step'] + 1
                logger.info("resuming from a step-based checkpoint")
            else:
                config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
                logger.info("resuming from a epoch-based checkpoint")
                assert config.TRAIN.START_STEP == 0, "resuming from a epoch-based checkpoint"
            config.freeze()
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            logger.info(f"=> Rank[{global_rank}] loaded successfully "
                        f"'{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
            print(f"=> Rank[{global_rank}] loaded successfully "
                  f"'{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
            if 'max_accuracy' in checkpoint:
                max_accuracy = checkpoint['max_accuracy']

        del checkpoint
        torch.cuda.empty_cache()

        return max_accuracy


def load_pretrained(config, model, logger):
    global_rank = dist.get_rank()
    logger.info(f"==============> Rank[{global_rank}] Loading weight {config.MODEL.PRETRAINED} for finetuning......")
    print(f"==============> Rank[{global_rank}] Loading weight {config.MODEL.PRETRAINED} for finetuning......")

    if config.MODEL.PRETRAINED.endswith(f'.rank{global_rank}'):
        pass
    elif config.MODEL.PRETRAINED.endswith(f'.pth'):
        config.defrost()
        config.MODEL.PRETRAINED = config.MODEL.PRETRAINED + f'.rank{global_rank}'
        config.freeze()
        logger.info(f"===> Rank[{global_rank}] Re-formatting checkpoint to {config.MODEL.PRETRAINED}......")
        print(f"===> Rank[{global_rank}] Re-formatting checkpoint to {config.MODEL.PRETRAINED}......")
    else:
        raise NotImplementedError(f"{config.MODEL.PRETRAINED} file error...")

    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # moe keys due to different version
    moe_fc1_weight_keys = [k for k in state_dict.keys() if "_moe_layer.experts.0.fc1_weight" in k]
    for k in moe_fc1_weight_keys:
        moe_fc1_weight_pretrained = state_dict[k]
        moe_fc1_weight_current = model.state_dict()[k]
        if len(moe_fc1_weight_pretrained.shape) == 2 and len(moe_fc1_weight_current.shape) == 2:
            C0, C1 = moe_fc1_weight_pretrained.shape
            D0, D1 = moe_fc1_weight_current.shape
            if C0 == D0 and C1 == D1:
                pass
            elif C0 == D1 and C1 == D0:
                state_dict[k] = moe_fc1_weight_pretrained.transpose(0, 1)
                logger.info(f"reformatting {k}")
    moe_fc1_bias_keys = [k for k in state_dict.keys() if
                         ("_moe_layer.experts.0.fc1_bias" in k or "_moe_layer.experts.0.fc2_bias" in k)]
    for k in moe_fc1_bias_keys:
        moe_fc1_bias_pretrained = state_dict[k]
        moe_fc1_bias_current = model.state_dict()[k]
        if len(moe_fc1_bias_pretrained.shape) == 2 and len(moe_fc1_bias_current.shape) == 1:
            assert moe_fc1_bias_pretrained.shape[0] == 1
            state_dict[k] = moe_fc1_bias_pretrained.view(-1)
            logger.info(f"reformatting {k}")
        elif len(moe_fc1_bias_pretrained.shape) == 1 and len(moe_fc1_bias_current.shape) == 2:
            assert moe_fc1_bias_current.shape[0] == 1
            state_dict[k] = moe_fc1_bias_pretrained.view(1, -1)
            logger.info(f"reformatting {k}")
    moe_gate_keys = [k for k in state_dict.keys() if "wg.weight" in k]
    for k in moe_gate_keys:
        if k not in model.state_dict():
            if k.endswith("gate.wg.weight"):
                new_k = k[:-len('gate.wg.weight')] + 'gates.0.wg.weight'
                state_dict[new_k] = state_dict[k]
                logger.info(f"reformatting {k}")
            elif k.endswith("gates.0.wg.weight"):
                new_k = k[:-len('gates.0.wg.weight')] + 'gate.wg.weight'
                state_dict[new_k] = state_dict[k]
                logger.info(f"reformatting {k}")

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                if config.MODEL.RPE_INTERPOLATION in ['bicubic', 'bilinear', 'nearest']:
                    logger.info("Interpolate relative_position_bias_table using bicubic.")
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                        mode=config.MODEL.RPE_INTERPOLATION)
                    state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)
                elif config.MODEL.RPE_INTERPOLATION == 'outer_mask':
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    pad_size = (S2 - S1) // 2
                    padding = (pad_size, pad_size, pad_size, pad_size)

                    all_rel_pos_bias = []
                    for i in range(nH1):
                        z = relative_position_bias_table_pretrained[:, i].view(S1, S1)
                        all_rel_pos_bias.append(
                            torch.nn.functional.pad(z, padding, "constant", z.min().item() - 3).view(L2, 1))
                    new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                    state_dict[k] = new_rel_pos_bias
                elif config.MODEL.RPE_INTERPOLATION == 'geo':
                    logger.info("Interpolate relative_position_bias_table using geo.")
                    src_size = int(L1 ** 0.5)
                    dst_size = int(L2 ** 0.5)

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r ** n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    logger.info("Original positions = %s" % str(x))
                    print("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(nH1):
                        z = relative_position_bias_table_pretrained[:, i].view(src_size, src_size).float().numpy()
                        f_cubic = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(torch.Tensor(f_cubic(dx, dy)).contiguous().view(-1, 1).to(
                            relative_position_bias_table_pretrained.device))

                    new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                    state_dict[k] = new_rel_pos_bias
                else:
                    raise NotImplementedError

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:

            logger.info("loading ImageNet22k weight to 1K ......")
            map22kto1k_path = f'data/{config.DATA.FINETUNE_MAPFILE}'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]

            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            nn.init.constant_(model.head.bias, 0.)
            nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            logger.warning(f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger,
                    step=None, loss_scaler=None):
    if loss_scaler is not None:
        global_rank = dist.get_rank()
        save_state = {'model': model_without_ddp.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'lr_scheduler': lr_scheduler.state_dict(),
                      'max_accuracy': max_accuracy,
                      'scaler': loss_scaler.state_dict(),
                      'epoch': epoch,
                      'config': config}
        if step is not None:
            save_state['step'] = step
        if step is not None:
            save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}_{step}.pth.rank{global_rank}')
        else:
            save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth.rank{global_rank}')
        logger.info(f"{save_path} saving......")
        print(f"{save_path} saving......")
        time.sleep(global_rank * 2)  # todo: add in 2022/03/26 to ease blob error
        try:
            torch.save(save_state, save_path)
        except:
            logger.info(f"{save_path} saving error retrying ......")
            print(f"{save_path} saving error retrying ......")
            torch.save(save_state, save_path)
        logger.info(f"{save_path} saved !!!")
        print(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir, logger, deepspeed=False):
    if not deepspeed:
        global_rank = dist.get_rank()
        checkpoints = os.listdir(output_dir)
        checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith(f'.pth.rank{global_rank}')]
        logger.info(f"All checkpoints for rank {global_rank} founded in {output_dir}: {checkpoints}")
        print(f"All checkpoints for rank {global_rank} founded in {output_dir}: {checkpoints}")
        if len(checkpoints) > 0:
            latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
            logger.info(f"The latest checkpoint founded for rank {global_rank} : {latest_checkpoint}")
            print(f"The latest checkpoint founded for rank {global_rank} : {latest_checkpoint}")
            resume_file = latest_checkpoint
        else:
            resume_file = None
        return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                                norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, custom_scaler=False):
        if custom_scaler:
            self._scaler = GradScaler(growth_interval=128, init_scale=2. ** 8)
        else:
            self._scaler = torch.cuda.amp.GradScaler(growth_interval=128, init_scale=2. ** 8)

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def create_ds_config(args, config):
    args.deepspeed_config = os.path.join(config.OUTPUT, f"deepspeed_config_{dist.get_rank()}.json")
    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    assert opt_lower == 'adamw', "deepspeed only support adamw"
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": config.DATA.BATCH_SIZE * config.TRAIN.ACCUMULATION_STEPS * dist.get_world_size(),
            "train_micro_batch_size_per_gpu": config.DATA.BATCH_SIZE,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": config.TRAIN.BASE_LR,
                    "weight_decay": config.TRAIN.WEIGHT_DECAY,
                    "bias_correction": True,
                    "betas": [
                        config.TRAIN.OPTIMIZER.BETAS[0],
                        config.TRAIN.OPTIMIZER.BETAS[1]
                    ],
                    "eps": config.TRAIN.OPTIMIZER.EPS
                }
            },
        }
        if args.dpfp16:
            ds_config["fp16"] = {"enabled": True,
                                 "loss_scale": 0,
                                 "initial_scale_power": 7,
                                 "loss_scale_window": 128}
        if config.TRAIN.CLIP_GRAD:
            ds_config["gradient_clipping"] = config.TRAIN.CLIP_GRAD
        if args.zero_opt > 0:
            ds_config["zero_optimization"] = {"stage": args.zero_opt}
        writer.write(json.dumps(ds_config, indent=2))


def hook_scale_grad(scale, tensor):
    return tensor / scale

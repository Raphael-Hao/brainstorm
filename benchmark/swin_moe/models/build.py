# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
from functools import partial

import torch.nn as nn

from .swin_mlp import SwinMLP
from .swin_transformer import SwinTransformer
from .swin_transformer_moe import SwinTransformerMoE
from .swin_transformer_v2_moe import LayerNormFP32, SwinV2TransformerMoE
from .micro_swin_v2_moe import MicroSwinV2TransformerMoE
from .vision_transformer import VisionTransformer


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == "swin":
        model = SwinTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )
    elif model_type == "swin_moe":
        model = SwinTransformerMoE(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN_MOE.PATCH_SIZE,
            in_chans=config.MODEL.SWIN_MOE.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN_MOE.EMBED_DIM,
            depths=config.MODEL.SWIN_MOE.DEPTHS,
            num_heads=config.MODEL.SWIN_MOE.NUM_HEADS,
            window_size=config.MODEL.SWIN_MOE.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN_MOE.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN_MOE.QKV_BIAS,
            qk_scale=config.MODEL.SWIN_MOE.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN_MOE.APE,
            patch_norm=config.MODEL.SWIN_MOE.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            moe_blocks=config.MODEL.SWIN_MOE.MOE_BLOCKS,
            num_local_experts=config.MODEL.SWIN_MOE.NUM_LOCAL_EXPERTS,
            top_value=config.MODEL.SWIN_MOE.TOP_VALUE,
            capacity_factor=config.MODEL.SWIN_MOE.CAPACITY_FACTOR,
            normalize_gate=config.MODEL.SWIN_MOE.NORMALIZE_GATE,
            batch_prioritized_routing=config.MODEL.SWIN_MOE.BATCH_PRIORITIZED_ROUTING,
            vitmoe_loss=config.MODEL.SWIN_MOE.VITMOE_LOSS,
            use_global_loss=config.MODEL.SWIN_MOE.USE_GLOBAL_LOSS,
            mlp_fc2_bias=config.MODEL.SWIN_MOE.MLP_FC2_BIAS,
            moe_drop=config.MODEL.SWIN_MOE.MOE_DROP,
            init_std=config.MODEL.SWIN_MOE.INIT_STD,
            aux_loss_scales=config.MODEL.SWIN_MOE.AUX_LOSS_SCALES,
        )
    elif model_type == "swinv2_moe":
        if config.MODEL.SWIN_V2_MOE.LN_TYPE == "normal":
            _layer_norm = nn.LayerNorm
        elif config.MODEL.SWIN_V2_MOE.LN_TYPE == "fp32":
            _layer_norm = LayerNormFP32
        else:
            raise NotImplementedError()
        model = SwinV2TransformerMoE(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN_V2_MOE.PATCH_SIZE,
            in_chans=config.MODEL.SWIN_V2_MOE.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN_V2_MOE.EMBED_DIM,
            depths=config.MODEL.SWIN_V2_MOE.DEPTHS,
            num_heads=config.MODEL.SWIN_V2_MOE.NUM_HEADS,
            window_size=config.MODEL.SWIN_V2_MOE.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN_V2_MOE.MLP_RATIO,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            norm_layer=partial(_layer_norm, eps=config.MODEL.SWIN_V2_MOE.LN_EPS),
            ape=config.MODEL.SWIN_V2_MOE.APE,
            patch_norm=config.MODEL.SWIN_V2_MOE.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            init_values=config.MODEL.SWIN_V2_MOE.INIT_VALUES,
            init_std=config.MODEL.SWIN_V2_MOE.INIT_STD,
            endnorm_interval=config.MODEL.SWIN_V2_MOE.ENDNORM_INTERVAL,
            relative_coords_table_type=config.MODEL.SWIN_V2_MOE.RELATIVE_COORDS_TABLE_TYPE,
            rpe_hidden_dim=config.MODEL.SWIN_V2_MOE.RPE_HIDDEN_DIM,
            rpe_output_type=config.MODEL.SWIN_V2_MOE.RPE_OUTPUT_TYPE,
            attn_type=config.MODEL.SWIN_V2_MOE.ATTN_TYPE,
            checkpoint_blocks=config.MODEL.SWIN_V2_MOE.CHECKPOINT_BLOCKS,
            mlpfp32_layer_blocks=config.MODEL.SWIN_V2_MOE.MLPFP32_LAYER_BLOCKS,
            postnorm=config.MODEL.SWIN_V2_MOE.POSTNORM,
            mlp_type=config.MODEL.SWIN_V2_MOE.MLP_TYPE,
            patch_embed_type=config.MODEL.SWIN_V2_MOE.PATCH_EMBED_TYPE,
            patch_merge_type=config.MODEL.SWIN_V2_MOE.PATCH_MERGE_TYPE,
            droppath_rule=config.MODEL.SWIN_V2_MOE.DROPPATH_RULE,
            strid16=config.MODEL.SWIN_V2_MOE.STRID16,
            strid16_global=config.MODEL.SWIN_V2_MOE.STRID16_GLOBAL,
            moe_blocks=config.MODEL.SWIN_V2_MOE.MOE_BLOCKS,
            num_local_experts=config.MODEL.SWIN_V2_MOE.NUM_LOCAL_EXPERTS,
            top_value=config.MODEL.SWIN_V2_MOE.TOP_VALUE,
            capacity_factor=config.MODEL.SWIN_V2_MOE.CAPACITY_FACTOR,
            normalize_gate=config.MODEL.SWIN_V2_MOE.NORMALIZE_GATE,
            batch_prioritized_routing=config.MODEL.SWIN_V2_MOE.BATCH_PRIORITIZED_ROUTING,
            vitmoe_loss=config.MODEL.SWIN_V2_MOE.VITMOE_LOSS,
            use_global_loss=config.MODEL.SWIN_V2_MOE.USE_GLOBAL_LOSS,
            use_noise=config.MODEL.SWIN_V2_MOE.USE_NOISE,
            mlp_fc2_bias=config.MODEL.SWIN_V2_MOE.MLP_FC2_BIAS,
            moe_drop=config.MODEL.SWIN_V2_MOE.MOE_DROP,
            moe_post_score=config.MODEL.SWIN_V2_MOE.MOE_POST_SCORE,
            moe_temperature_init=config.TRAIN.COSINE_GATE_T_START,
            aux_loss_scales=config.MODEL.SWIN_V2_MOE.AUX_LOSS_SCALES,
            rm_k_bias=config.MODEL.SWIN_V2_MOE.RM_K_BIAS,
            skip_moe_droppath=config.MODEL.SWIN_V2_MOE.SKIP_MOE_DROPPATH,
            head_2fc=config.MODEL.SWIN_V2_MOE.HEAD_2FC,
            norm_in_moe=config.MODEL.SWIN_V2_MOE.NORM_IN_MOE,
        )
    elif model_type == "micro_swinv2_moe":
        if config.MODEL.SWIN_V2_MOE.LN_TYPE == "normal":
            _layer_norm = nn.LayerNorm
        elif config.MODEL.SWIN_V2_MOE.LN_TYPE == "fp32":
            _layer_norm = LayerNormFP32
        else:
            raise NotImplementedError()
        model = MicroSwinV2TransformerMoE(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN_V2_MOE.PATCH_SIZE,
            in_chans=config.MODEL.SWIN_V2_MOE.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN_V2_MOE.EMBED_DIM,
            depths=config.MODEL.SWIN_V2_MOE.DEPTHS,
            num_heads=config.MODEL.SWIN_V2_MOE.NUM_HEADS,
            window_size=config.MODEL.SWIN_V2_MOE.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN_V2_MOE.MLP_RATIO,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            norm_layer=partial(_layer_norm, eps=config.MODEL.SWIN_V2_MOE.LN_EPS),
            ape=config.MODEL.SWIN_V2_MOE.APE,
            patch_norm=config.MODEL.SWIN_V2_MOE.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            init_values=config.MODEL.SWIN_V2_MOE.INIT_VALUES,
            init_std=config.MODEL.SWIN_V2_MOE.INIT_STD,
            endnorm_interval=config.MODEL.SWIN_V2_MOE.ENDNORM_INTERVAL,
            relative_coords_table_type=config.MODEL.SWIN_V2_MOE.RELATIVE_COORDS_TABLE_TYPE,
            rpe_hidden_dim=config.MODEL.SWIN_V2_MOE.RPE_HIDDEN_DIM,
            rpe_output_type=config.MODEL.SWIN_V2_MOE.RPE_OUTPUT_TYPE,
            attn_type=config.MODEL.SWIN_V2_MOE.ATTN_TYPE,
            checkpoint_blocks=config.MODEL.SWIN_V2_MOE.CHECKPOINT_BLOCKS,
            mlpfp32_layer_blocks=config.MODEL.SWIN_V2_MOE.MLPFP32_LAYER_BLOCKS,
            postnorm=config.MODEL.SWIN_V2_MOE.POSTNORM,
            mlp_type=config.MODEL.SWIN_V2_MOE.MLP_TYPE,
            patch_embed_type=config.MODEL.SWIN_V2_MOE.PATCH_EMBED_TYPE,
            patch_merge_type=config.MODEL.SWIN_V2_MOE.PATCH_MERGE_TYPE,
            droppath_rule=config.MODEL.SWIN_V2_MOE.DROPPATH_RULE,
            strid16=config.MODEL.SWIN_V2_MOE.STRID16,
            strid16_global=config.MODEL.SWIN_V2_MOE.STRID16_GLOBAL,
            moe_blocks=config.MODEL.SWIN_V2_MOE.MOE_BLOCKS,
            num_local_experts=config.MODEL.SWIN_V2_MOE.NUM_LOCAL_EXPERTS,
            top_value=config.MODEL.SWIN_V2_MOE.TOP_VALUE,
            capacity_factor=config.MODEL.SWIN_V2_MOE.CAPACITY_FACTOR,
            normalize_gate=config.MODEL.SWIN_V2_MOE.NORMALIZE_GATE,
            batch_prioritized_routing=config.MODEL.SWIN_V2_MOE.BATCH_PRIORITIZED_ROUTING,
            vitmoe_loss=config.MODEL.SWIN_V2_MOE.VITMOE_LOSS,
            use_global_loss=config.MODEL.SWIN_V2_MOE.USE_GLOBAL_LOSS,
            use_noise=config.MODEL.SWIN_V2_MOE.USE_NOISE,
            mlp_fc2_bias=config.MODEL.SWIN_V2_MOE.MLP_FC2_BIAS,
            moe_drop=config.MODEL.SWIN_V2_MOE.MOE_DROP,
            moe_post_score=config.MODEL.SWIN_V2_MOE.MOE_POST_SCORE,
            moe_temperature_init=config.TRAIN.COSINE_GATE_T_START,
            aux_loss_scales=config.MODEL.SWIN_V2_MOE.AUX_LOSS_SCALES,
            rm_k_bias=config.MODEL.SWIN_V2_MOE.RM_K_BIAS,
            skip_moe_droppath=config.MODEL.SWIN_V2_MOE.SKIP_MOE_DROPPATH,
            head_2fc=config.MODEL.SWIN_V2_MOE.HEAD_2FC,
            norm_in_moe=config.MODEL.SWIN_V2_MOE.NORM_IN_MOE,
        )
    elif model_type == "swin_mlp":
        model = SwinMLP(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN_MLP.PATCH_SIZE,
            in_chans=config.MODEL.SWIN_MLP.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN_MLP.EMBED_DIM,
            depths=config.MODEL.SWIN_MLP.DEPTHS,
            num_heads=config.MODEL.SWIN_MLP.NUM_HEADS,
            window_size=config.MODEL.SWIN_MLP.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN_MLP.MLP_RATIO,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN_MLP.APE,
            patch_norm=config.MODEL.SWIN_MLP.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )
    elif model_type == "vit":
        model = VisionTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.VIT.PATCH_SIZE,
            in_chans=config.MODEL.VIT.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.VIT.EMBED_DIM,
            depth=config.MODEL.VIT.DEPTH,
            num_heads=config.MODEL.VIT.NUM_HEADS,
            mlp_ratio=config.MODEL.VIT.MLP_RATIO,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model

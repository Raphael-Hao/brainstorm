#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /run.sh
# \brief:
# Author: raphael hao
python tvm_tune.py --model sparse_fusion_4_thor_model > sparse_fusion_4_thor_model.log
python tvm_tune.py --model sparse_fusion_8_thor_model > sparse_fusion_8_thor_model.log
python tvm_tune.py --model sparse_fusion_16_thor_model > sparse_fusion_16_thor_model.log
python tvm_tune.py --model sparse_fusion_32_thor_model > sparse_fusion_32_thor_model.log

#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /run.sh
# \brief:
# Author: raphael hao
python tvm_tune.py --model mask_serial_2_thor_model 2>&1 | tee mask_serial_2_thor_model.log
python tvm_tune.py --model mask_fusion_2_thor_model 2>&1 | tee mask_fusion_2_thor_model.log
python tvm_tune.py --model mask_serial_4_thor_model 2>&1 | tee mask_serial_4_thor_model.log
python tvm_tune.py --model mask_fusion_4_thor_model 2>&1 | tee mask_fusion_4_thor_model.log
python tvm_tune.py --model mask_serial_8_thor_model 2>&1 | tee mask_serial_8_thor_model.log
python tvm_tune.py --model mask_fusion_8_thor_model 2>&1 | tee mask_fusion_8_thor_model.log
python tvm_tune.py --model mask_serial_16_thor_model 2>&1 | tee mask_serial_16_thor_model.log
python tvm_tune.py --model mask_fusion_16_thor_model 2>&1 | tee mask_fusion_16_thor_model.log
python tvm_tune.py --model mask_serial_32_thor_model 2>&1 | tee mask_serial_32_thor_model.log
python tvm_tune.py --model mask_fusion_32_thor_model 2>&1 | tee mask_fusion_32_thor_model.log
python tvm_tune.py --model mask_serial_64_thor_model 2>&1 | tee mask_serial_64_thor_model.log
python tvm_tune.py --model mask_fusion_64_thor_model 2>&1 | tee mask_fusion_64_thor_model.log

# python tvm_tune.py --model sparse_serial_2_thor_model 2>&1 | teesparse_serial_2_thor_model.log
# python tvm_tune.py --model sparse_serial_4_thor_model 2>&1 | teesparse_serial_4_thor_model.log
# python tvm_tune.py --model sparse_serial_8_thor_model 2>&1 | teesparse_serial_8_thor_model.log
# python tvm_tune.py --model sparse_serial_64_thor_model 2>&1 | teesparse_serial_64_thor_model.log
# python tvm_tune.py --model sparse_fusion_64_thor_model 2>&1 | teesparse_fusion_64_thor_model.log

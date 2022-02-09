#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /run.sh
# \brief:
# Author: raphael hao
python tvm_tune.py --model sparse_fusion_2_thor_model > sparse_fusion_2_thor_model.log
python tvm_tune.py --model serial_2_thor_model > serial_2_thor_model.log
python tvm_tune.py --model fusion_4_thor_model > fusion_4_thor_model.log
python tvm_tune.py --model serial_4_thor_model > serial_4_thor_model.log

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /batched_matmul.py
# \brief:
# Author: raphael hao
from microbench.horizontal_expert.expert import run_batched_expert, run_serial_expert, onnx_check_results
from argparse import ArgumentParser

arg_parser = ArgumentParser()
arg_parser.add_argument("--batched", action="store_true")
arg_parser.add_argument("--bs", type=int, default=1)
arg_parser.add_argument("--T", type=int, default=128)
arg_parser.add_argument("--N", type=int, default=3072)
args = arg_parser.parse_args()
args.providers = ["TensorrtExecutionProvider" ,"CUDAExecutionProvider"]
# if args.batched:
#     run_batched_expert(args.bs, args.T, args.N, args.providers)
# else:
#     run_serial_expert(args.bs, args.T, args.N, args.providers)

batched_output = run_batched_expert(args.bs, args.T, args.N, args.providers)
serial_output = run_serial_expert(args.bs, args.T, args.N, args.providers)
# print(batched_output)
# print(serial_output)
# onnx_check_results(batched_output, serial_output)

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /batched_matmul.py
# \brief:
# Author: raphael hao
from microbench.horizontal_expert.expert import export_batched_expert, export_serial_expert, torch_check_results
from argparse import ArgumentParser

arg_parser = ArgumentParser()
arg_parser.add_argument("--bs", type=int, default=1)
arg_parser.add_argument("--T", type=int, default=128)
arg_parser.add_argument("--E", type=int, default=64)
arg_parser.add_argument("--N", type=int, default=3072)
arg_parser.add_argument("--K", type=int, default=768)
args = arg_parser.parse_args()

batched_out = export_batched_expert(args.bs, args.T, args.E, args.N, args.K)
serial_out = export_serial_expert(args.bs, args.T, args.E, args.N, args.K)
torch_check_results(batched_out, serial_out)

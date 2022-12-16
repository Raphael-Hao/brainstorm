#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /transform.py
# \brief:
# Author: raphael hao

import tvm
import tvm.te as te
import tvm.auto_scheduler as auto_scheduler
import tvm.topi as topi
import os
import argparse
import itertools


def gather_kernel(T, H, out_dtype="float32"):
    data = te.placeholder((T, H), dtype=out_dtype, name="input")
    indices = te.placeholder((T, H), dtype="int32", name="indices")
    out = topi.gather(data, 0, indices)
    return [data, indices, out]

def build_gather_kernel(T, H, out_dtype="float32"):
    data, indices, out = gather_kernel(T, H, out_dtype)
    with tvm.target.Target("cuda"):
        sch = topi.cuda.schedule_transpose(out)
        print(tvm.lower(sch, [data, indices, out], simple_mode=True))
    dev = tvm.device("cuda", 0)
    gather_kernel_func = tvm.build(sch, [data, indices, out], "cuda", name="gather_kernel")
    print(gather_kernel_func.imported_modules[0].get_source())

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--T", type=int, nargs="+", default=1)
    argparser.add_argument("--H", type=int, nargs="+", default=1)

    args = argparser.parse_args()

    for T, H in itertools.product(args.T, args.H):
        build_gather_kernel(T, H)


if __name__ == "__main__":
    main()

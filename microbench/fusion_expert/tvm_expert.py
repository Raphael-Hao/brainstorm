#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /tvm_tune.py
# \brief:
# Author: raphael hao
#%%
import argparse
import itertools
import os

import numpy as np
import tvm
import tvm.auto_scheduler as auto_scheduler
import tvm.te as te
import tvm.topi as topi
from tvm.auto_scheduler.measure_record import load_best_record




@auto_scheduler.register_workload
def serial_expert(batch, E, M, K, N, out_dtype="float32"):
    data = te.placeholder((M, K), dtype=out_dtype, name="input")
    weight = te.placeholder((N, K), dtype=out_dtype, name="weight")
    out = topi.nn.matmul(
        data,
        weight,
        transpose_a=False,
        transpose_b=True,
        out_dtype=out_dtype,
    )
    return [data, weight, out]


@auto_scheduler.register_workload
def fusion_expert(batch, E, M, K, N, out_dtype="float32"):
    data = te.placeholder((batch, M, K), dtype=out_dtype, name="input")
    weight = te.placeholder((E, N, K), dtype=out_dtype, name="weight")
    out = topi.nn.batch_matmul(
        data,
        weight,
        out_dtype=out_dtype,
    )
    return [data, weight, out]


def search_expert_kernel(
    batch, E, M, K, N, out_dtype="float32", resume=False, log_dir="."
):
    expert_kernel = fusion_expert
    if batch == 1 and E == 1:
        expert_kernel = serial_expert
    log_filename = f"tvm_{expert_kernel.__name__}_{batch}_{E}_{M}_{K}_{N}.json"
    log_file = os.path.join(log_dir, log_filename)
    print(f"Writing log to file: {log_file}")
    target = tvm.target.Target("cuda")
    task = auto_scheduler.SearchTask(
        func=expert_kernel, args=(batch, E, M, K, N, out_dtype), target=target
    )
    print(task.compute_dag)
    search_policy = None
    if resume:
        # check if the log file exists
        if os.path.exists(log_file):
            cost_model = auto_scheduler.XGBModel()
            cost_model.update_from_file(log_file)
            search_policy = auto_scheduler.SketchPolicy(
                task,
                cost_model,
                init_search_callbacks=[auto_scheduler.PreloadMeasuredStates(log_file)],
            )
    else:
        # if log_file exists, delete it
        if os.path.exists(log_file):
            os.remove(log_file)
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=5, min_repeat_ms=300)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=1000,
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    task.tune(tune_option, search_policy=search_policy)


def report_best_expert_kernel(batch, E, M, K, N, header=False, log_dir="."):
    expert_kernel = fusion_expert
    if batch == 1 and E == 1:
        expert_kernel = serial_expert
    log_filename = f"tvm_{expert_kernel.__name__}_{batch}_{E}_{M}_{K}_{N}.json"
    log_file = os.path.join(log_dir, log_filename)
    _, best_result = load_best_record(log_file)
    costs = [v.value for v in best_result.costs]
    best_time_cost_in_sec = np.mean(costs)
    best_time_cost_in_ms = best_time_cost_in_sec * 1000
    if header is True:
        print(
            f"batch, E, M, K, N, best_results \n{batch},{E},{M},{K},{N},{best_time_cost_in_ms}"
        )
    else:
        print(f"{batch},{E},{M},{K},{N},{best_time_cost_in_ms}")


def generate_standalone():
    # sch, args = task.apply_best(log_file=log_file)
    raise NotImplementedError


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--task",
        type=str,
        default="search",
        required=True,
        choices=["search", "export_best", "report_best"],
    )
    argparser.add_argument("--B", type=int, nargs="+", default=1)
    argparser.add_argument("--E", type=int, nargs="+", default=2)
    argparser.add_argument("--M", type=int, nargs="+", default=128)
    argparser.add_argument("--K", type=int, nargs="+", default=512)
    argparser.add_argument("--N", type=int, nargs="+", default=1024)

    argparser.add_argument(
        "--resume", action="store_true", help="resume from previous search"
    )
    argparser.add_argument(
        "--log_dir", type=str, default="log", help="directory to save log files"
    )
    argparser.add_argument("--header", action="store_false", help="print header")

    args = argparser.parse_args()
    args.log_dir = os.path.abspath(args.log_dir)

    if len(args.B) == len(args.E):
        args.BE = zip(args.B, args.E)
    elif (len(args.B) == 1 and args.B[0] == 1) or (len(args.E) == 1 and args.E[0] == 1):
        args.BE = itertools.product(args.B, args.E)
    else:
        raise NotImplementedError

    if args.task == "search":
        for (B, E), M, K, N in itertools.product(args.BE, args.M, args.K, args.N):
            search_expert_kernel(
                B, E, M, K, N, resume=args.resume, log_dir=args.log_dir
            )
    elif args.task == "report_best":
        print_header = True if args.header is True else False
        for (B, E), M, K, N in itertools.product(args.BE, args.M, args.K, args.N):
            report_best_expert_kernel(
                B,
                E,
                M,
                K,
                N,
                header=print_header,
                log_dir=args.log_dir,
            )
            print_header = False
    elif args.task == "export_best":
        generate_standalone()
    else:
        raise ValueError("unknown task")


if __name__ == "__main__":
    main()

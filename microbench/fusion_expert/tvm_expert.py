#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /tvm_tune.py
# \brief:
# Author: raphael hao
#%%
import tvm
import tvm.topi as topi
import tvm.te as te
import tvm.auto_scheduler as auto_scheduler
import argparse


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
    batch, E, M, K, N, out_dtype="float32", expert_kernel=fusion_expert
):
    target = tvm.target.Target("cuda")
    task = auto_scheduler.SearchTask(
        func=expert_kernel, args=(batch, E, M, K, N, out_dtype), target=target
    )
    print(task.compute_dag)
    log_file = f"tvm_{expert_kernel.__name__}_{batch}_{E}_{M}_{K}_{N}.json"
    print(log_file)
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=1000,
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    task.tune(tune_option)

    sch, args = task.apply_best(log_file=log_file)


def benchmark_expert_kernel(
    batch, E, M, K, N, out_dtype="float32", expert_kernel=fusion_expert
):
    # target = tvm.target.Target("cuda")
    # task = auto_scheduler.BenchmarkTask(
    #     func=fusion_expert, args=(batch, E, M, K, N, out_dtype), target=target
    # )
    # print(task.compute_dag)
    # log_file = f"tvm_{expert_kernel.__name__}_{batch}_{E}_{M}_{K}_{N}.json"
    # measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
    # tune_option = auto_scheduler.TuningOptions(
    #     num_measure_trials=1000,
    #     runner=measure_ctx.runner,
    #     measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    #     verbose=2,
    # )
    # task.tune(tune_option)

    # sch, args = task.apply_best(log_file=log_file)
    raise NotImplementedError


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--task",
        type=str,
        default="tune",
        required=True,
        choices=["search", "benchmark"],
    )
    argparser.add_argument(
        "--type",
        type=str,
        default="all",
        required=True,
        choices=["all", "fusion", "serial"],
    )
    argparser.add_argument("--batch", type=int, default=1)
    argparser.add_argument("--E", type=int, default=2)
    argparser.add_argument("--M", type=int, default=128)
    argparser.add_argument("--K", type=int, default=512)
    argparser.add_argument("--N", type=int, default=1024)
    args = argparser.parse_args()
    if args.task == "search":
        if args.type == "all":
            search_expert_kernel(
                args.batch, args.E, args.M, args.K, args.N, expert_kernel=fusion_expert
            )
            search_expert_kernel(
                args.batch, args.E, args.M, args.K, args.N, expert_kernel=serial_expert
            )
        elif args.type == "fusion":
            search_expert_kernel(
                args.batch, args.E, args.M, args.K, args.N, expert_kernel=fusion_expert
            )
        elif args.type == "serial":
            search_expert_kernel(
                args.batch, args.E, args.M, args.K, args.N, expert_kernel=serial_expert
            )
        else:
            raise ValueError("unknown type for search task")
    elif args.task == "benchmark":
        if args.type == "all":
            benchmark_expert_kernel(
                args.batch, args.E, args.M, args.K, args.N, expert_kernel=fusion_expert
            )
            benchmark_expert_kernel(
                args.batch, args.E, args.M, args.K, args.N, expert_kernel=serial_expert
            )
        elif args.type == "fusion":
            benchmark_expert_kernel(
                args.batch, args.E, args.M, args.K, args.N, expert_kernel=fusion_expert
            )
        elif args.type == "serial":
            benchmark_expert_kernel(
                args.batch, args.E, args.M, args.K, args.N, expert_kernel=serial_expert
            )
        else:
            raise ValueError("unknown type for benchmark task")
    else:
        raise ValueError("unknown task")


if __name__ == "__main__":
    main()

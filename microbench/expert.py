#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /expert.py
# \brief:
# Author: raphael hao

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import time
import numpy as np

class BatchedExpert(nn.Module):
    def __init__(self, E, N, K):
        super().__init__()
        self.M = E
        self.N = N
        self.K = K
        self.weight = torch.nn.Parameter(torch.ones(E, N, K))

    def forward(self, A):
        return torch.matmul(A, self.weight)


class SerialExpert(nn.Module):
    def __init__(self, E, N, K):
        super().__init__()
        self.E = E
        self.N = N
        self.K = K
        self.weights = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.ones(self.N, self.K)) for _ in range(self.E)]
        )

    def forward(self, A):
        tmp = [torch.matmul(A, w) for w in self.weights]
        return torch.stack(tmp, dim=1)


def torch_check_results(lft_output, rht_output):
    assert lft_output.shape == rht_output.shape
    assert torch.allclose(lft_output, rht_output)
    print("Results are the same!")

def onnx_check_results(lft_output, rht_output):
    assert np.allclose(lft_output, rht_output)
    print("Results are the same!")

def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def export_batched_expert(bs, T, E, N, K):
    model = BatchedExpert(E, N, K).cuda()
    model.eval()
    dummy_input = torch.ones(T, 1, bs, N, requires_grad=False).cuda()
    output = model(dummy_input)
    print(output.shape)
    torch.onnx.export(
        model,
        dummy_input,
        "batched_expert.onnx",
        opset_version=10,
        input_names=["input"],
        output_names=['output'],
        dynamic_axes={"input": {2: "batch_size"}, "output": {2: "batch_size"}},
    )
    onnx_model = onnx.load("batched_expert.onnx")
    onnx.checker.check_model(onnx_model)
    print("export batched_expert.onnx successfully!")
    return output


def run_batched_expert(bs, T, N, providers):
    ort_session = ort.InferenceSession("batched_expert.onnx", providers=providers)
    dummy_input = torch.ones(T, 1, bs, N, requires_grad=False)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    for i in range(10):
        ort_outputs = ort_session.run(None, ort_inputs)
    beg = time.time()
    for i in range(10):
        ort_outputs = ort_session.run(None, ort_inputs)
    end = time.time()
    print("time:", (end - beg) / 10)
    return ort_outputs


def export_serial_expert(bs, T, E, N, K):
    model = SerialExpert(E, N, K).cuda()
    model.eval()
    dummy_input = torch.ones(T, bs, N, requires_grad=False).cuda()
    output = model(dummy_input)
    print(output.shape)
    torch.onnx.export(
        model,
        dummy_input,
        "serial_expert.onnx",
        opset_version=10,
        input_names=["input"],
        output_names=['output'],
        dynamic_axes={"input": {1: "batch_size"}, "output": {2: "batch_size"}},
    )
    onnx_model = onnx.load("serial_expert.onnx")
    onnx.checker.check_model(onnx_model)
    print("export serial_expert.onnx successfully!")
    return output


def run_serial_expert(bs, T, N, providers):
    ort_session = ort.InferenceSession("serial_expert.onnx", providers=providers)
    dummy_input = torch.ones(T, bs, N, requires_grad=False)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    for i in range(10):
        ort_outputs = ort_session.run(None, ort_inputs)
    beg = time.time()
    for i in range(10):
        ort_outputs = ort_session.run(None, ort_inputs)
    end = time.time()
    print("time:", (end - beg) / 10)
    return ort_outputs


def export(bs, T, E, N, K):
    export_batched_expert(bs, T, E, N, K)
    export_serial_expert(bs, T, E, N, K)


def run(bs, T, N, providers):
    run_batched_expert(bs, T, N, providers)
    run_serial_expert(bs, T, N, providers)

#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time

import torch
import torch.nn.functional as F

from tutel_ea import moe, net, system

if torch.cuda.is_available():
  dist = system.init_data_model_parallel(backend='nccl')
else:
  dist = system.init_data_model_parallel(backend='gloo')

num_samples = 1024
model_dim, hidden_size = 64, 64
num_local_experts = 8
num_global_experts = num_local_experts * dist.global_size


class CustomGate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.register_parameter(name='wg', param=torch.nn.Parameter(torch.randn([model_dim, num_global_experts]) * 1e-3))

    def forward(self, x):
        return torch.matmul(x, self.wg)

class CustomExpert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(dist.global_rank + 1)
        self.register_parameter(name='batched_fc1_w', param=torch.nn.Parameter(torch.randn([num_local_experts, model_dim, hidden_size]) * 1e-3))
        self.register_parameter(name='batched_fc2_w', param=torch.nn.Parameter(torch.randn([num_local_experts, hidden_size, model_dim]) * 1e-3))
        self.register_parameter(name='batched_fc1_bias', param=torch.nn.Parameter(torch.zeros([num_local_experts, 1, hidden_size])))
        self.register_parameter(name='batched_fc2_bias', param=torch.nn.Parameter(torch.zeros([num_local_experts, 1, model_dim])))
        for x in self.parameters(): setattr(x, 'skip_allreduce', True)

    def forward(self, x):
        y = torch.add(torch.matmul(x, self.batched_fc1_w), self.batched_fc1_bias)
        y = F.relu(y)
        y = torch.add(torch.matmul(y, self.batched_fc2_w), self.batched_fc2_bias)
        return y

class CustomMoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = CustomGate()
        self.expert = CustomExpert()
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
    def start_timer(self):
        self.start_event.record(torch.cuda.current_stream())
    
    def end_timer(self, timer_name):
        self.end_event.record(torch.cuda.current_stream())
        torch.cuda.current_stream().synchronize()
        print("{}: {:.3f}".format(timer_name, self.start_event.elapsed_time(self.end_event)))
        
    def forward(self, x, k=1):
        logits = self.gate(x)
        scores = F.softmax(logits, dim=-1)
        self.start_timer()
        crit, l_aux = moe.top_k_routing(scores, top_k=k,capacity_factor=0)
        self.end_timer("top_k_routing")
        self.start_timer()
        y = moe.fast_encode(x, crit)
        y = net.all_to_all(y, 1, 0)
        self.end_timer("fast_encode")
        self.start_timer()
        y = self.expert(y)
        self.end_timer("expert")
        self.start_timer()
        y = net.all_to_all(y, 0, 1)
        output = moe.fast_decode(y, crit)
        self.end_timer("fast_decode")
        return output, l_aux

model = CustomMoE().to(dist.local_device)

torch.manual_seed(dist.global_rank + 1)
data = torch.randn([num_samples, model_dim], device=dist.local_device)
label = torch.LongTensor(num_samples).random_(1).to(dist.local_device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for i in range(10):
    t_start = system.record_time()

    optimizer.zero_grad()
    result, l_aux = model(data)
    result = F.log_softmax(result, dim=1)
    loss = F.nll_loss(result, label) + 0.0001 * l_aux
    loss.backward()

    for p in model.parameters():
        if not hasattr(p, 'skip_allreduce'):
            p.grad = net.simple_all_reduce(p.grad)
    optimizer.step()

    t_stop = system.record_time()

    dist.dist_print('STEP-%d: loss = %.5f, step_time = %.3f s' % (i, loss, t_stop - t_start))


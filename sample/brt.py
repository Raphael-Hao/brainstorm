# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Optional, Union

from abc import ABC
import torch.nn as nn
import torch


x: Union[torch.Tensor, List[torch.Tensor]]
Linear: nn.Module
ScatterRouter: nn.Module
GatherRouter: nn.Module
Netlet: nn.Module
RandomRouter: nn.Module
BeamSearchRouter: nn.Module
CandidateModules: nn.ModuleList
DistScatterRouter: nn.Module
DistGatherRouter: nn.Module
RecursiveRouter: nn.Module
SwitchRouter: nn.Module
FusionNetlet: nn.Module
NetletGroup: nn.ModuleDict
FusionNetletGroup: nn.ModuleDict
Split: nn.Module
DecoderBlock: nn.Module


class NetletGroup(nn.Module):
    def __init__(self, netlet_group: nn.ModuleDict):
        super(NetletGroup, self).__init__()
        self.netlet_group = netlet_group

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x_out = []
        for x_input, netlet in zip(x, self.netlet_group.values()):
            x_tmp = netlet(x_input)
            x_out.append(x_tmp)
        return x_out


class SwitchRouter(ABC):
    def __init__(self, scatter_method: nn.Module, gather_method: nn.Module):
        self.scatter_method = scatter_method
        self.gather_method = gather_method

    def Scatter(self, x: torch.Tensor) -> torch.Tensor:
        return self.scatter_method(x)

    def Gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.gather_method(x)


class RecursiveRouter(nn.Module):
    def __init__(self, next_method: nn.Module):
        self.next_method = next_method
        self.condtion = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.condtion = x
        return x

    def Proceed(self, x: torch.Tensor) -> torch.Tensor:
        if x == self.condtion:
            return True
        else:
            return False

    def Next(self, x):
        x = self.next_method(x)


x = Linear(x)
x = Split(x)
x = RandomRouter(x, CandidateModules)
x = Linear(x)

x = Linear(x)
x = Split(x)
x = BeamSearchRouter(x, DecoderBlock)
x = Linear(x)

# transform to our DSL
x = Linear(x)
x = Split(x)
with SwitchRouter as r:
    x, y = r.Scatter(x)
    x = NetletGroup(x)
    x = r.Gather(x, y)
x = Linear(x)

# determine single or distribute acording to hint
x = Linear(x)
x = Split(x)
with SwitchRouter as r:
    x, y = r.DistScatter(x)
    x = NetletGroup(x)
    x = r.DistGather(x, y)
x = Linear(x)

# horizontal fusion
x = Linear(x)
x = Split(x)
with SwitchRouter as r:
    x, y = r.DistScatter(x)
    x = FusionNetletGroup(x)
    x = r.DistGather(x, y)
x = Linear(x)


# recursive router for lstm
x = Linear(x)
x = Split(x)
# recursive router is defined with a generator operator, infinite or finite
# y is a condition, eos/ maxlenght
y: Union[torch.Tensor, List[torch.Tensor]]

# with RecursiveRouter(y) as r:
#     while x := r.next(x):
#         x = Netlet(x)

# with RecursiveRouter(y) as r:
with RecursiveRouter(y) as r:
    while True:
        x = Netlet(x)
        if r.Proceed(x):
            x = r.Next(x)
        else:
            break

# nesting router for  decoder

RecursiveRouter(x)

# %% greedy search example
from numpy import array
from numpy import argmax

# greedy decoder
def greedy_decoder(data):
    # index for largest probability each row
    return [argmax(s) for s in data]


# define a sequence of 10 words over a vocab of 5 words
data = [
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.5, 0.4, 0.3, 0.2, 0.1],
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.5, 0.4, 0.3, 0.2, 0.1],
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.5, 0.4, 0.3, 0.2, 0.1],
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.5, 0.4, 0.3, 0.2, 0.1],
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.5, 0.4, 0.3, 0.2, 0.1],
]
data = array(data)
# decode sequence
result = greedy_decoder(data)
print(result)

# beam search example
from math import log
from numpy import array
from numpy import argmax

# %% beam search
def beam_search_decoder(data, k):
    sequences = [[list(), 0.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score - log(row[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        # select k best
        sequences = ordered[:k]
    return sequences


""""
# greedy decoder
def greedy_decoder(input):
	token = input
	sequence = list()
	while True:
		token = decoder_block(token)
		sequence.append(token)
		if meet_condition:
			break
    # index for largest probability each row
    return [argmax(s) for s in sequences]

# decode sequence
result = greedy_decoder(data)

def beam_search(input, k):
	token = input
	sequences = [[list(), 0.0]]
	# walk over each step in sequence
	while true:
		token = decoder_block(data)
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(token)):
                candidate = [seq + [j], score - log(row[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        # select k best
        sequences = ordered[:k]
    return sequences
"""

result = beam_search_decoder(data, 3)

# define a sequence of 10 words over a vocab of 5 words
data = [
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.5, 0.4, 0.3, 0.2, 0.1],
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.5, 0.4, 0.3, 0.2, 0.1],
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.5, 0.4, 0.3, 0.2, 0.1],
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.5, 0.4, 0.3, 0.2, 0.1],
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.5, 0.4, 0.3, 0.2, 0.1],
]
data = array(data)
# decode sequence
result = beam_search_decoder(data, 3)
# print result
for seq in result:
    print(seq)

# %%
def foo():
    print("starting")
    while True:
        r = yield 2
        print(r)


f = foo()
print(f.send(None))
print(f.send(1))
print(f.send(1))

# %%

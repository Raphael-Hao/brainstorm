# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
from brt.routers.protocols.protocol import ProtocolBase, register_protocol


class DecoderProtocol(ProtocolBase):
    def __init__(
        self,
        index_format: str,
        pad_token_id: int,
        eos_token_id: int,
        skip_eos_check: bool = False,
    ):
        super().__init__(path_num=2, index_format=index_format)
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.skip_eos_check = skip_eos_check
        self.started_generative = False
        self.sorted_sequence_lengths = None
        self.iteration = 0

    def forward(self, score: torch.Tensor, seqence_length: torch.Tensor):
        raise NotImplementedError("DecoderProtocol needs to be subclassed")


@register_protocol("decoder_greedy_search")
class DecoderGreedySearchProtocol(DecoderProtocol):
    def __init__(
        self,
        index_format: str,
        pad_token_id: int,
        eos_token_id: int,
        skip_eos_check: bool = False,
    ):
        super().__init__(index_format, pad_token_id, eos_token_id, skip_eos_check)

    def forward(self, score: torch.Tensor, seqence_length: torch.Tensor = None):
        if self.skip_eos_check:
            if not self.started_generative:
                self.started_generative = True
                self.sorted_sequence_length = seqence_length.unique(sorted=True)
                self.checking_sequence_length = self.sorted_sequence_length[0]
                self.next_sequence_length_id = 1
                self.iteration = 0
            else:
                self.iteration += 1
                if self.iteration == self.checking_sequence_length:
                    self.checking_sequence_length = self.sorted_sequence_length[
                        self.next_sequence_length_id
                    ]
                    self.next_sequence_length_id += 1
        else:
            unfinished_mask = (score != self.eos_token_id).long()
            finished_mask = (score == self.eos_token_id).long()
            hot_mask = torch.stack([unfinished_mask, finished_mask]).t()
            

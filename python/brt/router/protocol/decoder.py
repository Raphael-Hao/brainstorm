# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
from brt.router.utils import generate_src_indices
from brt.router.protocol.base import ProtocolBase, register_protocol


class DecoderProtocol(ProtocolBase):
    def __init__(
        self,
        index_format: str,
        pad_token_id: int,
        eos_token_id: int,
        skip_eos_check: bool = False,
    ):
        super().__init__(index_format=index_format)
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.skip_eos_check = skip_eos_check
        if self.skip_eos_check:
            self.default_path = 0
        self.sorted_sequence_lengths = None
        self.iteration = 0

    def make_route_decision(self, score: torch.Tensor, seqence_length: torch.Tensor):
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

    def make_route_decision(self, score: torch.Tensor):
        if self.skip_eos_check:
            self.iteration += 1
            if self.iteration == self.checking_sequence_length:
                self.checking_sequence_length = self.sorted_sequence_length[
                    self.next_sequence_length_id
                ]
                self.next_sequence_length_id += 1
            if self.iteration + 1 < self.checking_sequence_length:
                route_indices = torch.zeros(
                    score.size(0), 2, dtype=torch.int64, device=score.device
                )
                route_indices[:, self.default_path] = torch.arange(
                    0, score.size(0), dtype=torch.int64, device=score.device
                )
                loads = torch.tensor(
                    [score.size(0), 0], dtype=torch.int64, device=score.device
                )

                return route_indices, loads, loads

        unfinished_mask = (score != self.eos_token_id).long()
        finished_mask = (score == self.eos_token_id).long()
        hot_mask = torch.cat([unfinished_mask, finished_mask], dim=1)
        route_indices, loads = generate_src_indices(hot_mask)

        return hot_mask, loads, loads

    def update(self, seqence_length: torch.Tensor):
        self.sorted_sequence_length = seqence_length.unique(sorted=True).cpu().numpy()
        self.checking_sequence_length = self.sorted_sequence_length[0]
        self.next_sequence_length_id = 1
        self.iteration = 0


@register_protocol("decoder_beam_search")
class DecoderBeamSearchProtocol(DecoderProtocol):
    def __init__(
        self,
        index_format: str,
        pad_token_id: int,
        eos_token_id: int,
        skip_eos_check: bool = False,
    ):
        super().__init__(index_format, pad_token_id, eos_token_id, skip_eos_check)

    def make_route_decision(self, score: torch.Tensor):
        if self.skip_eos_check:
            self.iteration += 1
            if self.iteration == self.checking_sequence_length:
                self.checking_sequence_length = self.sorted_sequence_length[
                    self.next_sequence_length_id
                ]
                self.next_sequence_length_id += 1
            if self.iteration + 1 < self.checking_sequence_length:
                pass

        raise NotImplementedError("DecoderBeamSearchProtocol not completed yet")

    def update(self, seqence_length: torch.Tensor):
        self.sorted_sequence_length = seqence_length.unique(sorted=True).cpu().numpy()
        self.checking_sequence_length = self.sorted_sequence_length[0]
        self.next_sequence_length_id = 1
        self.iteration = 0

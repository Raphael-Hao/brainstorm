import warnings
from abc import ABC, abstractmethod
from collections import UserDict
from typing import List, Optional, Tuple

import numpy as np
import torch

from transformers.generation_beam_search import BeamScorer, BeamHypotheses

class NoWaitBeamSearchScorer(BeamScorer):
    r"""
    [`BeamScorer`] implementing standard beam search decoding.

    Adapted in part from [Facebook's XLM beam search
    code](https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529).

    Reference for the diverse beam search algorithm and implementation [Ashwin Kalyan's DBS
    implementation](https://github.com/ashwinkalyan/dbs/blob/master/dbs/beam_utils.lua)

    Args:
        batch_size (`int`):
            Batch Size of `input_ids` for which standard beam search decoding is run in parallel.
        max_length (`int`):
            The maximum length of the sequence to be generated.
        num_beams (`int`):
            Number of beams for beam search.
        device (`torch.device`):
            Defines the device type (*e.g.*, `"cpu"` or `"cuda"`) on which this instance of `BeamSearchScorer` will be
            allocated.
        length_penalty (`float`, *optional*, defaults to 1.0):
            Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the
            model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer
            sequences.
        do_early_stopping (`bool`, *optional*, defaults to `False`):
            Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.
        num_beam_hyps_to_keep (`int`, *optional*, defaults to 1):
            The number of beam hypotheses that shall be returned upon calling
            [`~transformer.BeamSearchScorer.finalize`].
        num_beam_groups (`int`):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            See [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
    """

    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        device: torch.device,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[bool] = False,
        num_beam_hyps_to_keep: Optional[int] = 1,
        num_beam_groups: Optional[int] = 1,
        **kwargs,
    ):
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups

        self._is_init = False
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.num_beams,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
            )
            for _ in range(batch_size)
        ]
        self._done = torch.tensor(
            [False for _ in range(batch_size)], dtype=torch.bool, device=self.device
        )

        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                f"`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For `num_beams` == 1, one should make use of `greedy_search` instead."
            )

        if (
            not isinstance(num_beam_groups, int)
            or (num_beam_groups > num_beams)
            or (num_beams % num_beam_groups != 0)
        ):
            raise ValueError(
                f"`num_beam_groups` has to be an integer smaller or equal than `num_beams` and `num_beams` "
                f"has to be divisible by `num_beam_groups`, but is {num_beam_groups} with `num_beams` being {num_beams}."
            )

        if "max_length" in kwargs:
            warnings.warn(
                "Passing `max_length` to BeamSearchScorer is deprecated and has no effect. "
                "`max_length` should be passed directly to `beam_search(...)`, `beam_sample(...)`"
                ", or `group_beam_search(...)`."
            )

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor]:
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        if not (batch_size == (input_ids.shape[0] // self.group_size)):
            if self.num_beam_groups > 1:
                raise ValueError(
                    f"A group beam size of {input_ids.shape[0]} is used as the input, but a group beam "
                    f"size of {self.group_size} is expected by the beam scorer."
                )
            else:
                raise ValueError(
                    f"A beam size of {input_ids.shape[0]} is used as the input, but a beam size of "
                    f"{self.group_size} is expected by the beam scorer."
                )

        device = input_ids.device
        next_beam_scores = torch.zeros(
            (batch_size, self.group_size), dtype=next_scores.dtype, device=device
        )
        next_beam_tokens = torch.zeros(
            (batch_size, self.group_size), dtype=next_tokens.dtype, device=device
        )
        next_beam_indices = torch.zeros(
            (batch_size, self.group_size), dtype=next_indices.dtype, device=device
        )

        next_beam_scores[:, :] = next_scores[:, : self.group_size]
        next_beam_tokens[:, :] = next_tokens[:, : self.group_size]
        next_beam_indices[:, :] = (
            next_indices[:, : self.group_size]
            + torch.arange(0, batch_size, dtype=next_indices.dtype, device=device)
            * self.group_size
        )

        # for batch_idx, beam_hyp in enumerate(self._beam_hyps):
        #     # if self._done[batch_idx]:
        #     #     if self.num_beams < len(beam_hyp):
        #     #         raise ValueError(
        #     #             f"Batch can only be done if at least {self.num_beams} beams have been generated"
        #     #         )
        #     #     if eos_token_id is None or pad_token_id is None:
        #     #         raise ValueError(
        #     #             "Generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
        #     #         )
        #     #     # pad the batch
        #     #     next_beam_scores[batch_idx, :] = 0
        #     #     next_beam_tokens[batch_idx, :] = pad_token_id
        #     #     next_beam_indices[batch_idx, :] = 0
        #     #     continue
        #     # next tokens for this sentence
        #     beam_idx = 0
        #     for beam_token_rank, (next_token, next_score, next_index) in enumerate(
        #         zip(
        #             next_tokens[batch_idx],
        #             next_scores[batch_idx],
        #             next_indices[batch_idx],
        #         )
        #     ):
        #         batch_beam_idx = batch_idx * self.group_size + next_index
        #         # add to generated hypotheses if end of sentence
        #         # if (eos_token_id is not None) and (next_token.item() == eos_token_id):
        #         #     # if beam_token does not belong to top num_beams tokens, it should not be added
        #         #     is_beam_token_worse_than_top_num_beams = (
        #         #         beam_token_rank >= self.group_size
        #         #     )
        #         #     if is_beam_token_worse_than_top_num_beams:
        #         #         continue
        #         #     beam_hyp.add(
        #         #         input_ids[batch_beam_idx].clone(),
        #         #         next_score.item(),
        #         #     )
        #         # else:
        #         # add next predicted token since it is not eos_token
        #         next_beam_scores[batch_idx, beam_idx] = next_score
        #         next_beam_tokens[batch_idx, beam_idx] = next_token
        #         next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
        #         beam_idx += 1

        #         # once the beam for next step is full, don't add more tokens to it.
        #         if beam_idx == self.group_size:
        #             break

        #     if beam_idx < self.group_size:
        #         raise ValueError(
        #             f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id: {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
        #         )

        #     # Check if we are done so that we can save a pad step if all(done)
        #     # self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
        #     #     next_scores[batch_idx].max().item(), cur_len
        #     # )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
            }
        )

    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        max_length: int,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.LongTensor]:
        batch_size = len(self._beam_hyps)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_hyp.add(final_tokens, final_score)

        # select the best hypotheses
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []
        best_scores = torch.zeros(
            batch_size * self.num_beam_hyps_to_keep,
            device=self.device,
            dtype=torch.float32,
        )

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

                # append to lists
                best.append(best_hyp)
                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

        # prepare for adding eos
        sent_lengths_max = sent_lengths.max().item() + 1
        sent_max_len = (
            min(sent_lengths_max, max_length)
            if max_length is not None
            else sent_lengths_max
        )
        decoded: torch.LongTensor = input_ids.new(
            batch_size * self.num_beam_hyps_to_keep, sent_max_len
        )
        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded.fill_(pad_token_id)
        # fill with hypotheses and eos_token_id if the latter fits in
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < sent_max_len:
                decoded[i, sent_lengths[i]] = eos_token_id

        return UserDict(
            {
                "sequences": decoded,
                "sequence_scores": best_scores,
            }
        )

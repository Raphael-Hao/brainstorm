# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
from brt.common import log

logger = log.get_logger(__file__)
from . import protocol


def make_protocol(protocol_type, **kwargs):
    for key, value in kwargs.items():
        logger.debug(f"{key}: {value}")

    if protocol_type == "topk":
        protocol_cls = protocol.TopKProtocol
    elif protocol_type == "threshold":
        protocol_cls = protocol.ThresholdProtocol
    else:
        raise ValueError(f"Unknown protocol type: {protocol_type}")

    protocol_cls.forward = torch.jit.ignore(protocol_cls.forward)
    proto = protocol_cls(**kwargs)

    return proto

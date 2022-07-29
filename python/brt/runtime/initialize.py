# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

def initialize(shallow_transport=False):
    from brt.runtime.proto_tensor import ProtoTensor
    ProtoTensor.SHALLOW_TRANSPORT = shallow_transport
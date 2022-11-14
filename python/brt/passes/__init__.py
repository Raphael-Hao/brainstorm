# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from brt.passes.liveness import DeadPathEliminatePass, PermanentPathFoldPass
from brt.passes.memory_plan import (
    MemoryPlanPass,
    OnDemandMemoryPlanPass,
    PredictMemoryPlanPass,
)
from brt.passes.reorder import OperatorReorderPass
from brt.passes.no_batch import NoBatchPass
from brt.passes.constant_propagation import ConstantPropagationPass
from brt.passes.trace import TracePass
from brt.passes.vertical_fuse import VerticalFusePass
from brt.passes.horiz_fuse import HorizFusePass
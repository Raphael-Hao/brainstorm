# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import re

from brt.common import log

from .base import CUDATypeSizeInByte, GlobalFunction
from .utils import remove_empty_lines

logger = log.get_logger(__file__)


class RawFunction(GlobalFunction):
    def __init__(self, raw_source) -> None:
        self.raw_source = raw_source
        self.extract_raw_source()
        self.extract_culaunch_dims()
        self.extract_func_args()
        self.extract_shared_memory()
        self.clean_raw_body()

    def extract_raw_source(self):
        """
        Parse raw source code to extract function name and arguments.
        """
        # Example:
        # extern "C" __global__ void __launch_bounds__(32) fuse_add_blocks(float* %0, float* %1, float* %2 ) {}
        launch_bound_regex = r"\s+__launch_bounds__\((\w+)\)\s+"
        launch_bounds = re.findall(launch_bound_regex, self.raw_source)
        self.min_blocks_per_sm = 1
        if len(launch_bounds) == 0:
            self.max_threads_per_block = 0
        else:
            launch_bound_params = launch_bounds[0].split(",")
            self.max_threads_per_block = int(launch_bound_params[0])
            if len(launch_bound_params) == 2:
                self.min_blocks_per_sm = int(launch_bound_params[1])
            source_without_launch_bound = re.sub(
                launch_bound_regex, " ", self.raw_source
            )
        parsed_source = re.search(
            r"extern\s+\"C\"\s+__global__\s+void\s+(\w+)\s*\((.*)\)\s*(\{[\s\S]*)",
            source_without_launch_bound,
        )
        self.name = parsed_source.group(1)
        self.args = parsed_source.group(2)
        self.body = parsed_source.group(3)
        self.body = self.body[self.body.find("{") + 1 : self.body.rfind("}")]

    def extract_culaunch_dims(self):
        self.blockidx_x = int(
            re.search(
                r"\/\/\s*\[thread_extent\]\s*blockIdx.x\s*=\s*(\d+)", self.raw_source
            ).group(1)
        )
        self.blockidx_y = int(
            re.search(
                r"\/\/\s*\[thread_extent\]\s*blockIdx.y\s*=\s*(\d+)", self.raw_source
            ).group(1)
        )
        self.blockidx_z = int(
            re.search(
                r"\/\/\s*\[thread_extent\]\s*blockIdx.z\s*=\s*(\d+)", self.raw_source
            ).group(1)
        )
        self.threadidx_x = int(
            re.search(
                r"\/\/\s*\[thread_extent\]\s*threadIdx.x\s*=\s*(\d+)",
                self.raw_source,
            ).group(1)
        )
        self.threadidx_y = int(
            re.search(
                r"\/\/\s*\[thread_extent\]\s*threadIdx.y\s*=\s*(\d+)",
                self.raw_source,
            ).group(1)
        )
        self.threadidx_z = int(
            re.search(
                r"\/\/\s*\[thread_extent\]\s*threadIdx.z\s*=\s*(\d+)",
                self.raw_source,
            ).group(1)
        )
        self.blockidx_xydim = self.blockidx_x * self.blockidx_y
        self.threadidx_xydim = self.threadidx_x * self.threadidx_y
        self.grid_size = self.blockidx_xydim * self.blockidx_z
        self.block_size = self.threadidx_xydim * self.threadidx_z

    def extract_func_args(self):
        # find all function arguments
        self.arg_types = []
        self.arg_decorators = []
        self.arg_names = []
        for func_arg in self.args.split(","):
            func_arg = func_arg.strip()
            func_arg_info = func_arg.split(" ")
            if len(func_arg_info) == 2:
                self.arg_types.append(func_arg_info[0])
                self.arg_decorators.append([""])
                self.arg_names.append(func_arg_info[1])
            elif len(func_arg_info) == 3:
                self.arg_types.append(func_arg_info[0])
                self.arg_decorators.append(func_arg_info[1])
                self.arg_names.append(func_arg_info[2])
            else:
                raise ValueError("Invalid function argument: %s" % func_arg)

    def extract_shared_memory(self):
        # find all shared memory declarations
        shm_regex = r"__shared__\s+(\w+)\s+(\w+)\s*\[\s*(\d+)\s*\]\s*;"
        shm_declares = re.findall(shm_regex, self.body)
        if len(shm_declares) == 0:
            self.shm_size_in_bytes = 0
            return
        self.shm_types = []
        self.shm_symbols = []
        self.shm_sizes = []
        self.shm_size_in_bytes = 0
        for shm_declare in shm_declares:
            shm_type = shm_declare[0]
            shm_symbol = shm_declare[1]
            shm_size = int(shm_declare[2])
            self.shm_types.append(shm_type)
            self.shm_symbols.append(shm_symbol)
            self.shm_sizes.append(shm_size)
            self.shm_size_in_bytes += shm_size * CUDATypeSizeInByte[shm_type]
        self.body = re.sub(shm_regex, "", self.body)

    def clean_raw_body(self):
        self.body = remove_empty_lines(self.body)

    def extract_syncthreads_times(self):
        # only count the number of __syncthreads existing in the code
        self.syncthreads_times = self.body.count("__syncthreads()")

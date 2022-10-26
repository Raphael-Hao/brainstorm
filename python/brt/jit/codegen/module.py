# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import json
import re
from typing import Dict, List, Union

import torch

from brt.runtime import log
from brt.jit.utils import get_device_name
from brt.jit.codegen.cuda import CUDATypeSizeInByte, GlobalKernel
from brt.jit.codegen.storage import kernel_storager
from brt.jit.codegen.utils import (
    make_func_name,
    make_identifier,
    remove_comments,
    remove_empty_lines,
)

logger = log.get_logger(__file__)

__all__ = ["ModuleKernel", "ModuleDTypeSizeInByte"]

ModuleDTypeSizeInByte = {
    torch.float32: 4,
    torch.float64: 8,
    torch.float16: 2,
    torch.bfloat16: 2,
}


class ModuleKernel(GlobalKernel):
    def __init__(
        self,
        module_name,
        method: str,
        kernel_source=None,
        platform="CUDA_GPU",
        input_infos: Dict[str, List[int]] = None,
        output_infos: Dict[str, List[int]] = None,
        parameters: Dict[str, List[Union[int, float]]] = None,
    ):
        if not hasattr(self, "kernel_type"):
            setattr(self, "kernel_type", "global")
        super().__init__()
        self.module_name = module_name
        self.method = method
        self.kernel_source = kernel_source
        self.platform = platform
        self.input_infos = input_infos
        self.output_infos = output_infos
        self.parameters = parameters
        if self.kernel_source is not None:
            self.initialize()

    def initialize(self):
        self.func_name = make_func_name(
            self.module_name, self.input_infos, self.output_infos, self.parameters
        )
        self.extract_raw_source()
        self.extract_culaunch_dims()
        self.extract_func_args()
        self.extract_shared_memory()
        self.clean_raw_body()
        self.initialized = True

    def extract_raw_source(self):
        """
        Parse raw source code to extract function name and arguments.
        """
        # Example:
        # extern "C" __global__ void __launch_bounds__(32) fuse_add_blocks(float* %0, float* %1, float* %2 ) {}
        launch_bound_regex = r"\s+__launch_bounds__\((\w+)\)\s+"
        launch_bounds = re.findall(launch_bound_regex, self.kernel_source)
        self.min_blocks_per_sm = 1
        if len(launch_bounds) == 0:
            self.max_threads_per_block = 0
        else:
            launch_bound_params = launch_bounds[0].split(",")
            self.max_threads_per_block = int(launch_bound_params[0])
            if len(launch_bound_params) == 2:
                self.min_blocks_per_sm = int(launch_bound_params[1])
            source_without_launch_bound = re.sub(
                launch_bound_regex, " ", self.kernel_source
            )
        parsed_source = re.search(
            r"extern\s+\"C\"\s+__global__\s+void\s+(\w+)\s*\((.*)\)\s*(\{[\s\S]*)",
            source_without_launch_bound,
        )
        self.args = parsed_source.group(2)
        self.raw_body = parsed_source.group(3)
        self.raw_body = self.raw_body[
            self.raw_body.find("{") + 1 : self.raw_body.rfind("}")
        ]

    def extract_culaunch_dims(self):
        self.blockidx_x = int(
            re.search(
                r"//\s*\[thread_extent\]\s*blockIdx.x\s*=\s*(\d+)", self.kernel_source
            ).group(1)
        )
        self.blockidx_y = int(
            re.search(
                r"//\s*\[thread_extent\]\s*blockIdx.y\s*=\s*(\d+)", self.kernel_source
            ).group(1)
        )
        self.blockidx_z = int(
            re.search(
                r"//\s*\[thread_extent\]\s*blockIdx.z\s*=\s*(\d+)", self.kernel_source
            ).group(1)
        )
        self.threadidx_x = int(
            re.search(
                r"//\s*\[thread_extent\]\s*threadIdx.x\s*=\s*(\d+)",
                self.kernel_source,
            ).group(1)
        )
        self.threadidx_y = int(
            re.search(
                r"//\s*\[thread_extent\]\s*threadIdx.y\s*=\s*(\d+)",
                self.kernel_source,
            ).group(1)
        )
        self.threadidx_z = int(
            re.search(
                r"//\s*\[thread_extent\]\s*threadIdx.z\s*=\s*(\d+)",
                self.kernel_source,
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
        shm_declares = re.findall(shm_regex, self.raw_body)
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
        self.raw_body = re.sub(shm_regex, "", self.raw_body)

    def clean_raw_body(self):
        self.raw_body = remove_comments(self.raw_body)
        self.raw_body = remove_empty_lines(self.raw_body)

    def make_identifier(self):
        if self.platform == "CUDA_GPU":
            assert torch.cuda.is_available()
        self.device_name = get_device_name(self.platform)
        return make_identifier(
            self.module_name,
            self.method,
            self.device_name,
            self.input_infos,
            self.output_infos,
            self.parameters,
        )

    def dump_to_db(self, objective_func: str = "fastest", rank: int = 1):
        assert self.input_infos is not None and self.output_infos is not None
        code, func_deps, func_signature, func_body = self.get_code()
        key = code
        identifier = self.make_identifier()
        source = "BRT"
        device_type = self.platform
        attribute_dict = {}
        attribute_dict.update({"input_shape:": self.input_infos})
        attribute_dict.update({"output_shape:": self.output_infos})
        if self.parameters is not None:
            attribute_dict.update({"parameters:": self.parameters})
        attributes = json.dumps(attribute_dict)
        function_dict = {}
        function_dict.update({"function_deps": func_deps})
        function_dict.update({"function_signature": func_signature})
        function_dict.update({"function_body": func_body})
        function_dict.update(
            {"grid_dim": [self.blockidx_x, self.blockidx_y, self.blockidx_z]}
        )
        function_dict.update(
            {"block_dim": [self.threadidx_x, self.threadidx_y, self.threadidx_z]}
        ),
        function = json.dumps(function_dict)
        tag_dict = {}
        tag_dict.update({"kernel_type": self.kernel_type})
        tag = json.dumps(tag_dict)
        miscs_dict = {}
        miscs = json.dumps(miscs_dict)
        module_dict = {
            "Key": key,
            "Identifier": identifier,
            "OpType": self.module_name,
            "Attributes": attributes,
            "Source": source,
            "DeviceType": device_type,
            "Function": function,
            "Tags": tag,
            "Miscs": miscs,
            "ObjectiveFunc": objective_func,
            "Rank": rank,
        }
        kernel_storager.add_kernel(module_dict, overwrite=True)
        return self

    def load_from_db(self, objective_func: str = "fastest", rank: int = 1):
        identifier = self.make_identifier()
        fetched_kernel = kernel_storager.query_kernel(
            identifier, self.platform, objective_func, rank
        )
        if fetched_kernel is None:
            raise ValueError(
                f"No kernel found in database with {identifier = }, {objective_func = }"
            )
        attribute_dict = json.loads(fetched_kernel[3])
        function_dict = json.loads(fetched_kernel[6])
        tag_dict = json.loads(fetched_kernel[7])
        assert self.kernel_type == tag_dict["kernel_type"]
        if self.kernel_type == "global":
            self.kernel_source = fetched_kernel[0]
            self.initialize()
        else:
            self.func_deps = function_dict["function_deps"]
            self.func_sig = function_dict["function_signature"]
            self.func_body = function_dict["function_body"]
            self.initialized = True
        return self

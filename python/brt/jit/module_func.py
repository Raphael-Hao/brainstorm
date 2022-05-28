# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import json
import re
from typing import Dict, List, Union

from brt.common import log

from .base import CUDATypeSizeInByte, GlobalFunction
from .utils import make_attributes, make_identifier, remove_empty_lines

logger = log.get_logger(__file__)


class ModuleFunction(GlobalFunction):
    def __init__(
        self,
        module_name,
        kernel_source=None,
        platform="CUDA_GPU",
        input_infos: Dict[str, List[int]] = None,
        output_infos: Dict[str, List[int]] = None,
        parameters: Dict[str, List[Union[int, float]]] = None,
    ):
        super().__init__("global")
        self.module_name = module_name
        self.kernel_source = kernel_source
        self.platform = platform
        self.input_infos = input_infos
        self.output_infos = output_infos
        self.parameters = parameters
        if self.kernel_source is not None:
            self.initialize(source="raw")

    def initialize(self, source="raw"):
        self.extract_raw_source()
        self.extract_culaunch_dims()
        self.extract_func_args()
        self.extract_shared_memory()
        self.clean_raw_body()
        self.kernel_type = "global"
        self.initialized = True

    def extract_raw_source(self, source="raw"):
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
        if source == "raw":
            self.func_name = parsed_source.group(1)
        else:
            self.func_name = self.module_name
        self.args = parsed_source.group(2)
        self.raw_body = parsed_source.group(3)
        self.raw_body = self.raw_body[
            self.raw_body.find("{") + 1 : self.raw_body.rfind("}")
        ]

    def extract_culaunch_dims(self):
        self.blockidx_x = int(
            re.search(
                r"\/\/\s*\[thread_extent\]\s*blockIdx.x\s*=\s*(\d+)", self.kernel_source
            ).group(1)
        )
        self.blockidx_y = int(
            re.search(
                r"\/\/\s*\[thread_extent\]\s*blockIdx.y\s*=\s*(\d+)", self.kernel_source
            ).group(1)
        )
        self.blockidx_z = int(
            re.search(
                r"\/\/\s*\[thread_extent\]\s*blockIdx.z\s*=\s*(\d+)", self.kernel_source
            ).group(1)
        )
        self.threadidx_x = int(
            re.search(
                r"\/\/\s*\[thread_extent\]\s*threadIdx.x\s*=\s*(\d+)",
                self.kernel_source,
            ).group(1)
        )
        self.threadidx_y = int(
            re.search(
                r"\/\/\s*\[thread_extent\]\s*threadIdx.y\s*=\s*(\d+)",
                self.kernel_source,
            ).group(1)
        )
        self.threadidx_z = int(
            re.search(
                r"\/\/\s*\[thread_extent\]\s*threadIdx.z\s*=\s*(\d+)",
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
        self.raw_body = remove_empty_lines(self.raw_body)

    def dump_json(self):
        assert self.input_infos is not None and self.output_infos is not None
        code, func_signature, func_body = self.get_code()
        key = code
        identifier = make_identifier(
            self.module_name, self.input_infos, self.output_infos, self.parameters,
        )
        source = "BRT"
        device_type = self.platform
        attribute_dict = {}
        attribute_dict.update({"input_shape:": list(self.input_infos.values())})
        attribute_dict.update({"output_shape:": list(self.output_infos.values())})
        if self.parameters is not None:
            attribute_dict.update({"parameters:": list(self.parameters.values())})
        attributes = json.dumps(attribute_dict)
        function_dict = {}
        function_dict.update({"function_signature": func_signature})
        function_dict.update({"function_body": func_body})
        function_dict.update(
            {"grid_dim": [self.blockidx_x, self.blockidx_y, self.blockidx_z]}
        )
        function_dict.update(
            {"block_dim": [self.threadidx_x, self.threadidx_y, self.threadidx_z]}
        ),
        function = json.dumps(function_dict)
        miscs_dict = {}
        miscs = json.dumps(miscs_dict)
        module_json_dict = {
            "Key": key,
            "Identifier": identifier,
            "OpType": self.module_name,
            "Attributes": attributes,
            "Source": source,
            "DeviceType": device_type,
            "Function": function,
            "Tags": self.kernel_type,
            "Miscs": miscs,
        }
        return module_json_dict

    def load_json(self, module_in_json):
        self.kernel_source = module_in_json["Key"]
        self.initialize()


# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import re

from .utils import remove_empty_lines

CUDATypeSizeInByte = {
    # signed type
    "char": 1,
    "short": 2,
    "int": 4,
    "float": 4,
    "double": 8,
    "int8_t": 1,
    "int16_t": 2,
    "int32_t": 4,
    "int64_t": 8,
    # unsigned type
    "uchar": 1,
    "ushort": 2,
    "uint": 4,
    "uint8_t": 1,
    "uint16_t": 2,
    "uint32_t": 4,
    "uint64_t": 8,
}


class GenericFunction:
    common_defines = """
#define uint unsigned int
#define uchar unsigned char
#define ushort unsigned short
#define int64_t long long
#define uint64_t unsigned long long
"""
    global_decorator = """extern "C" __global__ void """
    device_decorator = """__device__ __forceinline__ void """

    def __init__(self, raw_source) -> None:
        self.raw_source = raw_source
        self.extract_raw_source()
        self.extract_culaunch_dims()
        self.extract_shared_memory()
        self.extract_func_args()

    def extract_raw_source(self):
        """
        Parse raw source code to extract function name and arguments.
        """
        # Example:
        # extern "C" __global__ void fuse_add_blocks(float* %0, float* %1, float* %2 ) {}
        parsed_source = re.search(
            r"extern\s+\"C\"\s+__global__\s+void\s+__launch_bounds__\((\d+)\)\s+(\w+)\s*\((.*)\)\s*(\{[\s\S]*)",
            self.raw_source,
        )
        self.launch_bounds = int(parsed_source.group(1))
        self.name = parsed_source.group(2)
        self.args = parsed_source.group(3)
        self.body = parsed_source.group(4)
        self.body = self.body[self.body.find("{") + 1 : self.body.rfind("}")]

    def extract_culaunch_dims(self):
        self.blockidx_xdim = int(
            re.search(
                r"\/\/\s*\[thread_extent\]\s*blockIdx.xdim\s*=\s*(\d+)", self.raw_source
            ).group(1)
        )
        self.blockidx_ydim = int(
            re.search(
                r"\/\/\s*\[thread_extent\]\s*blockIdx.ydim\s*=\s*(\d+)", self.raw_source
            ).group(1)
        )
        self.blockidx_zdim = int(
            re.search(
                r"\/\/\s*\[thread_extent\]\s*blockIdx.zdim\s*=\s*(\d+)", self.raw_source
            ).group(1)
        )
        self.threadidx_xdim = int(
            re.search(
                r"\/\/\s*\[thread_extent\]\s*threadIdx.xdim\s*=\s*(\d+)",
                self.raw_source,
            ).group(1)
        )
        self.threadidx_ydim = int(
            re.search(
                r"\/\/\s*\[thread_extent\]\s*threadIdx.ydim\s*=\s*(\d+)",
                self.raw_source,
            ).group(1)
        )
        self.threadidx_zdim = int(
            re.search(
                r"\/\/\s*\[thread_extent\]\s*threadIdx.zdim\s*=\s*(\d+)",
                self.raw_source,
            ).group(1)
        )
        self.blockidx_xydim = self.blockidx_xdim * self.blockidx_ydim
        self.threadidx_xydim = self.threadidx_xdim * self.threadidx_ydim
        self.grid_size = self.blockidx_xydim * self.blockidx_zdim
        self.block_size = self.threadidx_xydim * self.threadidx_zdim

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
        self.body = remove_empty_lines(self.body)

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

    def get_code(self, mode: str = "global") -> str:
        clean_code = ""
        if mode == "global":
            clean_code += self.common_defines
            clean_code += self.global_decorator
            clean_code += f"__launch_bounds__({self.launch_bounds})"
            clean_code += f" {self.name}("
            clean_code += f"{self.args})"
            clean_code += "{\n"
            clean_code += f"  // [thread_extent] blockIdx.xdim = {self.blockidx_xdim}\n"
            clean_code += f"  // [thread_extent] blockIdx.ydim = {self.blockidx_ydim}\n"
            clean_code += f"  // [thread_extent] blockIdx.zdim = {self.blockidx_zdim}\n"
            clean_code += (
                f"  // [thread_extent] threadIdx.xdim = {self.threadidx_xdim}\n"
            )
            clean_code += (
                f"  // [thread_extent] threadIdx.ydim = {self.threadidx_ydim}\n"
            )
            clean_code += (
                f"  // [thread_extent] threadIdx.zdim = {self.threadidx_zdim}\n"
            )
            clean_code += "\n"
            if self.shm_size_in_bytes > 0:
                for i in range(len(self.shm_sizes)):
                    clean_code += f"  __shared__ {self.shm_types[i]} {self.shm_symbols[i]}[{self.shm_sizes[i]}];\n"
            clean_code += f"{self.body}"
            clean_code += "}\n"
        elif mode == "device":
            clean_code += self.device_decorator
            clean_code += f" {self.name}("
            clean_code += f"{self.args}"
            clean_code += (
                f", char* shared_buffer, const uint block_idx, const uint thread_idx)"
            )
            clean_code += "{\n"
            clean_code += f"  // [thread_extent] gridSize = {self.grid_size}\n"
            clean_code += f"  // [thread_extent] blockSize = {self.block_size}\n"
            clean_code += "\n"
            clean_code += f"  if (thread_idx >= {self.block_size}) return;\n"
            clean_code += f"  uint block_idx_x = block_idx % {self.blockidx_xdim};\n"
            clean_code += f"  uint block_idx_y = (block_idx / {self.blockidx_xdim}) % {self.blockidx_ydim};\n"
            clean_code += f"  uint block_idx_z = block_idx / {self.blockidx_xydim};\n"
            clean_code += f"  uint thread_idx_x = thread_idx % {self.threadidx_xdim};\n"
            clean_code += f"  uint thread_idx_y = (thread_idx / {self.threadidx_xdim}) % {self.threadidx_ydim};\n"
            clean_code += (
                f"  uint thread_idx_z = thread_idx / {self.threadidx_xydim};\n"
            )
            clean_code += (
                f"  dim3 brt_blockIdx = {{block_idx_x, block_idx_y, block_idx_z}};\n"
            )
            clean_code += f"  dim3 brt_threadIdx = {{thread_idx_x, thread_idx_y, thread_idx_z}};\n"
            allocated_shm_size = 0
            if self.shm_size_in_bytes > 0:
                for i in range(len(self.shm_sizes)):
                    clean_code += f"  {self.shm_types[i]}* {self.shm_symbols[i]} = ({self.shm_types[i]}*)(shared_buffer + {allocated_shm_size});\n"
                    allocated_shm_size += (
                        self.shm_sizes[i] * CUDATypeSizeInByte[self.shm_types[i]]
                    )
            assert allocated_shm_size == self.shm_size_in_bytes

            device_func_body = self.body.replace("blockIdx", "brt_blockIdx")
            device_func_body = device_func_body.replace("threadIdx", "brt_threadIdx")
            clean_code += f"{device_func_body}"
            clean_code += "}\n"
        else:
            raise ValueError("Invalid mode: %s" % mode)
        return clean_code


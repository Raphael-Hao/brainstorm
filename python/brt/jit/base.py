# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from brt.common import log

logger = log.get_logger(__file__)

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

class BaseFunction:
    def __init__(self) -> None:
        pass

    def set_launch_bounds(self):
        if self.max_threads_per_block is 0:
            return
        if self.min_blocks_per_sm == 1:
            self.clean_code += f"__launch_bounds__({self.max_threads_per_block})"
        else:
            self.clean_code += f"__launch_bounds__({self.max_threads_per_block}, {self.min_blocks_per_sm})"

    def declare_name_args(self):
        self.clean_code += f" {self.name}("
        self.clean_code += f"{self.args}"
        if self.mode == "device":
            self.clean_code += (
                f", char* shared_buffer, const uint& block_idx, const uint& thread_idx"
            )
        self.clean_code += ")"

    def set_culaunch_dims(self):
        if self.mode == "device":
            self.clean_code += f"  // [thread_extent] gridSize = {self.grid_size}\n"
            self.clean_code += f"  // [thread_extent] blockSize = {self.block_size}\n"
        else:
            self.clean_code += (
                f"  // [thread_extent] blockIdx.xdim = {self.blockidx_xdim}\n"
            )
            self.clean_code += (
                f"  // [thread_extent] blockIdx.ydim = {self.blockidx_ydim}\n"
            )
            self.clean_code += (
                f"  // [thread_extent] blockIdx.zdim = {self.blockidx_zdim}\n"
            )
            self.clean_code += (
                f"  // [thread_extent] threadIdx.xdim = {self.threadidx_xdim}\n"
            )
            self.clean_code += (
                f"  // [thread_extent] threadIdx.ydim = {self.threadidx_ydim}\n"
            )
            self.clean_code += (
                f"  // [thread_extent] threadIdx.zdim = {self.threadidx_zdim}\n"
            )
        self.clean_code += "\n"
        
    def alloc_shared_memory(self):
        if self.shm_size_in_bytes > 0:
            if self.mode == "device":
                allocated_shm_size = 0
                for i in range(len(self.shm_sizes)):
                    self.clean_code += f"  {self.shm_types[i]}* {self.shm_symbols[i]} = ({self.shm_types[i]}*)(shared_buffer + {allocated_shm_size});\n"
                    allocated_shm_size += (
                        self.shm_sizes[i] * CUDATypeSizeInByte[self.shm_types[i]]
                    )
                assert allocated_shm_size == self.shm_size_in_bytes
            else:
                for i in range(len(self.shm_sizes)):
                    self.clean_code += f"  __shared__ {self.shm_types[i]} {self.shm_symbols[i]}[{self.shm_sizes[i]}];\n"

    def reset_mode(self, mode="global"):
        self.mode = mode
        self.clean_code = ""
        self.indent = 0

    def new_emtpy_line(self):
        self.clean_code += "\n"

    def new_codeblock(self):
        self.clean_code += "{\n"
        self.indent += 1

    def close_codeblock(self):
        self.clean_code += "}\n"
        self.indent -= 1

    def verify_code(self):
        try:
            assert self.indent == 0
        except AssertionError:
            logger.exception("Code verify failed")


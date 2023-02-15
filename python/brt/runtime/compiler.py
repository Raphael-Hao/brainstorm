# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

#%%
import re
import hashlib
from collections import defaultdict
from typing import Any, Callable, Dict, List, Union, Tuple, NamedTuple

# from dataclasses import dataclass

import torch
from brt.runtime.singleton import Singleton


def make_binary(source: str) -> Callable[..., None]:
    pass


class LaunchBounds(NamedTuple):
    max_threads: int = None
    min_blocks: int = None


class Dim(NamedTuple):
    x: int = 1
    y: int = 1
    z: int = 1


class HomoFuseInfo(NamedTuple):
    branch_num: int
    capacities: List[int]
    arg_num: int
    shared_arg_num: int
    shared_arg_granularities: List[int]


class KernelConfig(NamedTuple):
    source: str
    kernel_type: str
    launch_mode: str
    launch_bounds: LaunchBounds
    grid_dims: List[Dim]
    block_dims: List[Dim]
    shared_memory_bytes: int
    arg_types: List[str]
    arg_names: List[str]
    fuse_info: Union[HomoFuseInfo, None]


class KernelBinary:
    def __init__(self, launcher, source) -> None:
        self.source = source
        self.launcher = launcher

    def _call_impl(self, *args, **kwargs):
        self.launcher(*args, **kwargs)

    __call__: Callable[..., Any] = _call_impl


class KernelCompiler(metaclass=Singleton):
    """A compiler that can compile CUDA source strings into callables.

    Attributes:
        source (str): The CUDA source string.
    """

    def __init__(self) -> None:
        self.supported_launch_mode = ["static", "mask", "dispatch"]
        self.supported_kernel_type = [
            "global",
            "horiz_fuse",
            "hetero_fuse",
            "homo_fuse",
        ]
        self.bin_cache: Dict[int, Dict[str, KernelBinary]] = defaultdict(dict)
        self.reset()

    def reset(self):
        self.source: str = None
        self.config: KernelConfig = None

    def compile(self, source: str) -> KernelBinary:
        """Compile the CUDA source string into a callable.

        Returns:
            Callable: A callable that can be invoked with the same arguments as
                the kernel.
        """
        self.reset()
        self.source = source
        kernel_device = torch.cuda.current_device()
        kernel_key = hashlib.md5(source.encode("utf-8")).hexdigest()
        try:
            return self.bin_cache[kernel_device][kernel_key]
        except KeyError:
            self.parse_kernel_config()
            kernel_bin = self.make_kernel_binary()
            self.bin_cache[kernel_device][kernel_key] = kernel_bin
            return kernel_bin

    def parse_kernel_config(self, source: str = None) -> KernelConfig:
        """Parse the kernel configuration from the source string.

        Returns:
            KernelConfig: The kernel configuration.
        """
        if source is not None:
            self.reset()
            self.source = source
        kernel_type, launch_mode = self._get_type_and_launch_mode()
        grid_dims, block_dims = self._get_launch_dim(launch_mode)
        shared_memory_bytes = self._get_launch_shared_memory()
        launch_bounds, arg_types, arg_names = self._extract_signature()
        fusion_info = None
        if kernel_type == "homo_fuse":
            fusion_info = self._get_homo_fuse_info()

        self.config = KernelConfig(
            self.source,
            kernel_type,
            launch_mode,
            launch_bounds,
            grid_dims,
            block_dims,
            shared_memory_bytes,
            arg_types,
            arg_names,
            fusion_info,
        )
        return self.config

    def _get_type_and_launch_mode(self) -> Tuple[str, str]:
        """Get the launch mode of the kernel from the source string.

        Raises:
            NotImplementedError: If the launch mode is not supported.

        Returns:
            str: The launch mode.
        """

        kernel_type = re.search(
            r"\/\/\s+\[kernel_type\]\s+(\w+)\s*", self.source
        ).groups()[0]
        if kernel_type not in self.supported_kernel_type:
            raise NotImplementedError(f"Kernel type: {kernel_type} is not supported.")
        if kernel_type in ["global", "horiz_fuse"]:
            launch_mode = "static"
        elif kernel_type in ["hetero_fuse"]:
            launch_mode = "mask"
        elif kernel_type in ["homo_fuse"]:
            launch_mode = "dispatch"
        return kernel_type, launch_mode

    def _get_launch_dim(self, launch_mode) -> Tuple[List[Dim], List[Dim]]:
        """Get the launch dimension of the kernel from the source string.

        Args:
            which_dim (str): which dimension to get. should be something like [block|thread]Idx.[x|y|z].
            is_array (bool, optional):  Defaults to False. if the dimension is an array, e.g. blockIdx.x [].

        Returns:
            : int
        """

        dims = ["blockIdx", "threadIdx"]
        axiss = ["x", "y", "z"]
        raw_dims = {"blockIdx": [], "threadIdx": []}
        for dim in dims:
            for axis in axiss:
                raw_dims[dim].append(self._get_dim_axis(dim, axis))
        if launch_mode == "static":
            assert all(isinstance(x, int) for x in raw_dims["blockIdx"]) and all(
                isinstance(x, int) for x in raw_dims["threadIdx"]
            )
            return [Dim(*raw_dims["blockIdx"])], [Dim(*raw_dims["threadIdx"])]
        elif launch_mode == "mask" or launch_mode == "dispatch":
            fused_kernel_num = len(raw_dims["blockIdx"][0])
            assert all(len(raw_dims["threadIdx"]) == fused_kernel_num)
            assert all(isinstance(x, int) for x in raw_dims["threadIdx"][1:])
            assert all(isinstance(x, int) for x in raw_dims["threadIdx"][1:])
            return (
                [Dim(x) for x in raw_dims["blockIdx"][0]],
                [Dim(x) for x in raw_dims["threadIdx"][0]],
            )
        else:
            raise NotImplementedError(f"Launch mode: {launch_mode} is not supported.")

    def _get_dim_axis(self, which_dim: str, which_axis: str) -> Union[int, List[int]]:
        axis_match = re.search(
            rf"\/\/\s+\[thread_extent\]\s+{which_dim}\.{which_axis}\s*=\s*(\d+|\[[0-9,\s]+\])\s*",
            self.source,
        )
        if axis_match:
            raw_axis = axis_match.groups()[0]
            if raw_axis.startswith("["):
                return [int(x) for x in raw_axis[1:-1].split(",")]
            else:
                return int(raw_axis)
        else:
            raise ValueError(
                f"Cannot find launch configuration {which_dim}.{which_axis} in the source string."
            )

    def _get_launch_shared_memory(self) -> int:
        """Get the shared memory size of the kernel from the source string.

        Returns:
            int: The shared memory size in bytes.

        """
        shared_memory_match = re.search(
            r"\/\/\s+\[thread_extent\]\s+shared_memory\s*=\s*(\d+)\s*", self.source
        )
        if shared_memory_match:
            return int(shared_memory_match.groups()[0])
        else:
            return 0

    def _get_homo_fuse_info(self) -> HomoFuseInfo:
        """Get the information of the fused kernels.

        Returns:
            List[Dict[str, Any]]: The information of the fused kernels.
        """
        raw_branch_num = re.search(
            r"\/\/\s+\[homo_fuse_info\]\s+branch_num\s*=\s*(\d+)\s*", self.source
        ).groups()[0]
        branch_num = int(raw_branch_num)
        raw_capacities = re.search(
            r"\/\/\s+\[homo_fuse_info\]\s+supported_capacity\s*=\s*\[([0-9,\s]+)\]",
            self.source,
        ).groups()[0]
        capacities = [int(x) for x in raw_capacities.split(",")]
        raw_arg_num = re.search(
            r"\/\/\s+\[homo_fuse_info\]\s+arg_num\s*=\s*(\d+)\s*", self.source
        ).groups()[0]
        arg_num = int(raw_arg_num)
        raw_shared_arg_num = re.search(
            r"\/\/\s+\[homo_fuse_info\]\s+shared_arg_num\s*=\s*(\d+)\s*", self.source
        ).groups()[0]
        shared_arg_num = int(raw_shared_arg_num)
        raw_shared_arg_granularities = re.search(
            r"\/\/\s+\[homo_fuse_info\]\s+shared_arg_grans\s*=\s*\[([0-9,\s]+)\]",
            self.source,
        ).groups()[0]
        shared_arg_granularities = [
            int(x) for x in raw_shared_arg_granularities.split(",")
        ]
        return HomoFuseInfo(
            branch_num, capacities, arg_num, shared_arg_num, shared_arg_granularities
        )

    def _extract_signature(self) -> Tuple[LaunchBounds, List[str], List[str]]:
        """Extract the arguments of the kernel from the source string.

        Returns:
            Tuple[List[str],List[str]]: [list of argument types, list of argument names]
        """
        signature_pattern = r"__global__\s+void\s+(__launch_bounds__\(\s*(\d+)\s*,?\s*(\d+)?\s*\))?\s*\w+\((((\s*\w+\s*\*)\s+(__restrict__)?\s+(\w+)\s*(,)?)+)\)"

        signature_match = re.search(signature_pattern, self.source)
        if signature_match:
            arg_types = []
            arg_names = []
            launch_bounds = None
            if signature_match.groups()[1]:
                max_threads = int(signature_match.groups()[1])
                min_blocks = None
                if signature_match.groups()[2]:
                    min_blocks = int(signature_match.groups()[2])
                launch_bounds = LaunchBounds(max_threads, min_blocks)
            raw_args = signature_match.groups()[3]
            arg_pattern = re.compile(r"(\s*\w+\s*\*)\s+(__restrict__)?\s+(\w+)\s*")
            args = raw_args.split(",")
            for arg in args:
                arg_match = arg_pattern.match(arg)
                if arg_match:
                    arg_types.append(arg_match.groups()[0])
                    arg_names.append(arg_match.groups()[2])
                else:
                    raise ValueError(
                        f"Cannot parse the argument: {arg} in the source string."
                    )
            return launch_bounds, arg_types, arg_names
        else:
            raise ValueError("Cannot find the kernel signature in the source string.")

    def make_kernel_binary(self, config: KernelConfig = None) -> KernelBinary:
        """Generate the binary of the kernel.

        Args:
            config (KernelConfig): The configuration of the kernel.

        Returns:
            bytes: The binary of the kernel.
        """
        return KernelBinary(None, None)

    def make_paramer_struct(self):
        """Generate the struct of the kernel parameters.

        Returns:
            source code for generating the struct: The struct of the kernel parameters.
        """

        if self.config.launch_mode == "dispatch":
            return f"""
Struct KernelParam {{
    {";".join(f"{self.config.arg_types[i]} {self.config.arg_names[i]}" for i in range(self.config.branch_num))  }
}}
"""
        else:
            return ""

    def make_cuModule_generator(self):
        """Generate the cumodule from the source string.

        Returns:
            source code for generating cuModule: The cumodule of the kernel.
        """
        cuModule_generator_header = """
#include <cuda.h>

"""
        cuModule_generator_src = """
  std::string arch_option = "--gpu-architecture=compute_" + arch;
  std::vector<const char*> param_cstrings = {
      "--restrict",        "--include-path=/usr/local/cuda/include",
      arch_option.c_str(), "--use_fast_math",
      "--std=c++14",       "--extra-device-vectorization"};
  nvrtcProgram prog;
  NVRTC_CHECK(nvrtcCreateProgram(&prog, code, nullptr, 0, nullptr, nullptr));
  nvrtcResult nvrtc_compile_result =
      nvrtcCompileProgram(prog, param_cstrings.size(), param_cstrings.data());

  if (nvrtc_compile_result != NVRTC_SUCCESS) {
    size_t log_size;
    NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &log_size));
    std::string log;
    log.resize(log_size);
    NVRTC_CHECK(nvrtcGetProgramLog(prog, &log[0]));
    LOG(FATAL) << "nvrtcCompileProgram failed: \\n" << log;
  }

  size_t ptx_size;
  NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_size));

  std::string ptx;
  ptx.resize(ptx_size);
  NVRTC_CHECK(nvrtcGetPTX(prog, &ptx[0]));
  NVRTC_CHECK(nvrtcDestroyProgram(&prog));
"""

    def make_kernel_launcher(self):
        launcher_torch_header = """
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
// #include <c10/cuda/CUDACachingAllocator.h>
#include <torch/extension.h>

#undef CHECK_EQ
#undef CHECK_NE
#undef CHECK_ON_CPU
#undef CHECK_ON_CUDA
#undef CHECK_CONTIGUOUS

#define CHECK_EQ(x, y) TORCH_INTERNAL_ASSERT((x) == (y), "CHECK_EQ fails.")
#define CHECK_NE(x, y) TORCH_INTERNAL_ASSERT((x) != (y), "CHECK_NE fails.")
#define CHECK_ON_CPU(x) TORCH_INTERNAL_ASSERT(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_ON_CUDA(x) TORCH_INTERNAL_ASSERT(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_INTERNAL_ASSERT(x.is_contiguous(), #x " must be contiguous")
"""
        static_launcher_src = """
static void launch(const std::vector<::torch::Tensor>& ts, const std::vector<long>& args) {
  std::vector<const void*> pargs(ts.size() + args.size()), ppargs(ts.size() + args.size());
  for (int i = 0; i < (int)ts.size(); ++i) {
    CHECK_ON_CUDA(ts[i]);
    pargs[i] = ts[i].data_ptr();
    ppargs[i] = &pargs[i];
  }
  for (int i = (int)ts.size(); i < (int)pargs.size(); ++i) {
    pargs[i] = (void*)args[i - ts.size()];
    ppargs[i] = &pargs[i];
  }

  int dev = ts[0].device().index();
  CHECK_EQ(0, cudaSetDevice(dev));
  jit::CUDACompiler::GetCompiler().StaticExecute(ppargs, fd, dev,
                                                 at::cuda::getDefaultCUDAStream().stream());
}

"""


# %%
import pathlib


def test_cuda_compiler():
    kernel_code_dir = pathlib.Path(
        "/home/whcui/brainstorm_project/brainstorm/.cache/kernel_template"
    )
    global_code = kernel_code_dir / "global.cu"
    horiz_fuse_code = kernel_code_dir / "horiz_fuse.cu"
    hetero_fuse_code = kernel_code_dir / "hetero_fuse.cu"
    homo_fuse_code = kernel_code_dir / "homo_fuse.cu"

    compiler = KernelCompiler()
    compiler.parse_kernel_config(global_code.read_text())
    print(compiler.parse_kernel_config(horiz_fuse_code.read_text()))
    print(compiler.parse_kernel_config(hetero_fuse_code.read_text()))
    # print(compiler.parse_kernel_config(homo_fuse_code.read_text()))


test_cuda_compiler()
# %%

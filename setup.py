import pathlib
import sys
import copy
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if len(sys.argv) <= 1:
    sys.argv += ["install", "--user"]

root_path = pathlib.Path(__file__).parent.absolute()


def install(use_cuda=False):
    torch_extensions = []
    if use_cuda:
        ext_libs, ext_args = (
            [],
            {
                "cxx": ["-Wno-sign-compare", "-Wno-unused-but-set-variable"],
                "nvcc": [
                    "-O3",
                    "-Xcompiler",
                    "-fopenmp",
                    "-Xcompiler",
                    "-fPIC",
                    "-std=c++14",
                ],
            },
        )
        ext_libs += ["dl", "cuda", "nvrtc"]
        ext_args["cxx"] += ["-DUSE_CUDA"]
        ext_args["nvcc"] += ["-DUSE_CUDA"]
        torch_extensions = [
            CUDAExtension(
                name="brt._C.jit",
                sources=[
                    "./src/backend/torch/jit.cc",
                    "./src/jit/compiler.cu",
                ],
                library_dirs=["/usr/local/cuda/lib64/stubs"],
                libraries=ext_libs,
                include_dirs=[
                    root_path / "include",
                    root_path / "3rdparty/dmlc-core/include",
                ],
                extra_compile_args=ext_args,
            ),
            CUDAExtension(
                name="brt._C.router",
                sources=[
                    "./src/backend/torch/router.cc",
                    "./src/router/location.cu",
                    "./src/router/route.cu",
                ],
                library_dirs=["/usr/local/cuda/lib64/stubs"],
                libraries=ext_libs,
                include_dirs=[
                    root_path / "include",
                    root_path / "3rdparty/dmlc-core/include",
                ],
                extra_compile_args=ext_args,
            ),
        ]
        dist_libs = ext_libs + ["nccl"]
        dist_args = copy.deepcopy(ext_args)
        dist_args["cxx"] += ["-DUSE_NCCL"]
        dist_args["nvcc"] += ["-DUSE_NCCL"]
        torch_extensions.append(
            CUDAExtension(
                name="brt._C.distributed",
                sources=[
                    "./src/backend/torch/distributed.cc",
                    "./src/backend/torch/nccl_manager.cc",
                    "./src/distributed/asymmetry.cc",
                ],
                library_dirs=["/usr/local/cuda/lib64/stubs"],
                libraries=dist_libs,
                include_dirs=[
                    root_path / "include",
                    root_path / "3rdparty/dmlc-core/include",
                ],
                extra_compile_args=dist_args,
            )
        )
    setup(
        name="brt",
        version="0.1",
        author="Weihao Cui",
        package_dir={"": "python"},
        packages=find_packages("python"),
        ext_modules=torch_extensions,
        cmdclass={"build_ext": BuildExtension},
    )


try:
    print("Installing brt with CUDA runtime...")
    install(use_cuda=True)
except:
    print("CUDA build failed, skipping CUDA build")
    try:
        install(use_cuda=False)
    except:
        raise RuntimeError("Failed to install brt")

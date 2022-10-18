import pathlib
import sys

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
        ext_args["cxx"] += ["-DUSE_CUDA", "-DUSE_NCCL"]
        ext_args["nvcc"] += ["-DUSE_CUDA", "-DUSE_NCCL"]
        torch_extensions = [
            CUDAExtension(
                name="brt._C.jit",
                sources=["./src/backend/torch/jit.cc", "./src/jit/compiler.cu",],
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
    print("CUDA not found, skipping CUDA build")
    try:
        install(use_cuda=False)
    except:
        raise RuntimeError("Failed to install brt")

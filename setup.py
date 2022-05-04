import pathlib
import sys

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if len(sys.argv) <= 1:
    sys.argv += ["install", "--user"]

root_path = pathlib.Path(__file__).parent.absolute()


def install():
    ext_libs, ext_args = (
        [],
        {"cxx": ["-Wno-sign-compare", "-Wno-unused-but-set-variable"]},
    )
    ext_libs += ["dl", "cuda", "nvrtc"]
    ext_args["cxx"] += ["-DUSE_CUDA"]

    setup(
        name="brt",
        version="0.1",
        author="Weihao Cui",
        package_dir={"": "python"},
        packages=find_packages("python"),
        ext_modules=[
            CUDAExtension(
                name="brt.jit.cppjit",
                sources=["./src/jit/extension/torch.cc",],
                library_dirs=["/usr/local/cuda/lib64/stubs"],
                libraries=ext_libs,
                include_dirs=[
                    root_path / "include",
                    root_path / "3rdparty/dmlc-core/include",
                ],
                extra_compile_args=ext_args,
            )
        ],
        cmdclass={"build_ext": BuildExtension},
    )


try:
    install()
except:
    raise RuntimeError("Failed to install brt")

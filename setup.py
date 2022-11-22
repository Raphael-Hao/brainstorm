import copy
import pathlib
import subprocess
import sys
import os
import json

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if len(sys.argv) <= 1:
    sys.argv += ["install", "--user"]

root_path = pathlib.Path(__file__).parent.absolute()


def is_ninja_available():
    r"""
    Returns ``True`` if the `ninja <https://ninja-build.org/>`_ build system is
    available on the system, ``False`` otherwise.
    """
    try:
        subprocess.check_output("ninja --version".split())
    except Exception:
        return False
    else:
        return True


class BrtBuildExtension(BuildExtension):
    def build_extension(self, ext) -> None:
        super().build_extension(ext)
        if is_ninja_available():
            is_first_extension = getattr(self, "is_first_extension", True)
            if is_first_extension:
                setattr(self, "is_first_extension", False)
                build_temp_dir = pathlib.Path(self.build_temp)
                build_dir = build_temp_dir.parent
                compdb_filepath = build_dir / "compile_commands.json"
                setattr(self, "compdb_filepath", compdb_filepath)
                compdb_cmd = ["ninja", "-C"]
                compdb_cmd.extend([build_temp_dir.as_posix()])
                compdb_cmd.extend(["-t", "compdb"])
                setattr(self, "compdb_cmd", compdb_cmd)
                compdb_list = []
                setattr(self, "compdb_list", compdb_list)
            env = os.environ.copy()
            try:
                sys.stdout.flush()
                sys.stderr.flush()
                compdb_output = subprocess.check_output(
                    self.compdb_cmd,
                    stderr=subprocess.PIPE,
                    cwd=os.getcwd(),
                    env=env,
                )
                new_compdb_list = json.loads(compdb_output)
                self.compdb_list.extend(new_compdb_list)
                self.compdb_filepath.write_text(json.dumps(self.compdb_list))

            except subprocess.CalledProcessError as e:
                # Python 2 and 3 compatible way of getting the error object.
                _, error, _ = sys.exc_info()
                # error.output contains the stdout and stderr of the build attempt.
                message = "Generating compile_commands.json failed with error: "
                # `error` is a CalledProcessError (which has an `ouput`) attribute, but
                # mypy thinks it's Optional[BaseException] and doesn't narrow
                if hasattr(error, "output") and error.output:  # type: ignore[union-attr]
                    message += f": {error.output.decode(*())}"  # type: ignore[union-attr]
                raise RuntimeError(message) from e


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
                    "./src/distributed/collective.cc",
                    "./src/distributed/local_reorder.cu",
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
        cmdclass={"build_ext": BrtBuildExtension},
    )


print("Installing brt with CUDA runtime...")
install(use_cuda=True)
# try:
#     print("Installing brt with CUDA runtime...")
#     install(use_cuda=True)
# except:
#     print("CUDA build failed, skipping CUDA build")
#     try:
#         install(use_cuda=False)
#     except:
#         raise RuntimeError("Failed to install brt")

# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import os
import pathlib

__all__ = [
    "HOME_PATH",
    "BRT_PKG_PATH",
    "BRT_CACHE_PATH",
    "BRT_LOG_PATH",
    "BRT_LOG_FNAME",
    "BRT_CKPT_PATH",
    "BRT_KERNEL_TUNE_LOG_PATH",
    "BRT_KERNEL_TEMPLATE_PATH",
    "BRT_KERNEL_DB_FNAME",
    # "get_dll_directories",
    "find_lib_path",
]

# BRT root paths
HOME_PATH = pathlib.Path.home()
BRT_PKG_PATH = pathlib.Path(__file__).parent.parent
BRT_CACHE_PATH = pathlib.Path(
    os.getenv("BRT_CACHE_PATH", str(HOME_PATH / ".cache/brt"))
).absolute()
BRT_LOG_PATH = pathlib.Path(
    os.getenv("BRT_LOG_PATH", str(BRT_CACHE_PATH / "log"))
).absolute()

# brt root log file
BRT_LOG_FNAME = str(BRT_LOG_PATH / "brainstorm.log")

# checkpoint path
BRT_CKPT_PATH = BRT_CACHE_PATH / "ckpt"

# kernel tune
BRT_KERNEL_TUNE_LOG_PATH = BRT_LOG_PATH / "kernel_tune"
BRT_KERNEL_TEMPLATE_PATH = BRT_CACHE_PATH / "kernel_template"
BRT_KERNEL_DB_FNAME = BRT_CACHE_PATH / "kernel_db.sqlite"


BRT_CACHE_PATH.mkdir(parents=True, exist_ok=True)
BRT_LOG_PATH.mkdir(parents=True, exist_ok=True)
BRT_CKPT_PATH.mkdir(parents=True, exist_ok=True)
BRT_KERNEL_TUNE_LOG_PATH.mkdir(parents=True, exist_ok=True)
BRT_KERNEL_TEMPLATE_PATH.mkdir(parents=True, exist_ok=True)


def get_dll_directories():
    default_ddl_dir = BRT_PKG_PATH.parent.parent / "build"
    dll_dir = pathlib.Path(os.getenv("BRT_LIBRARY_PATH", str(default_ddl_dir)))
    return [dll_dir]


def find_lib_path(name=None, search_path: str = None):
    search_paths = get_dll_directories()
    if search_path is not None:
        if isinstance(search_path, str):
            search_paths.append(pathlib.Path(search_path))
        elif isinstance(search_path, list):
            search_paths.extend(pathlib.Path(p) for p in search_path)

    if name is not None:
        if isinstance(name, list):
            lib_dll_path = []
            for n in name:
                lib_dll_path += [p / n for p in search_paths]
        else:
            lib_dll_path = [p / name for p in search_paths]
        torchscript_dll_path = []
    else:
        lib_dll_names = ["libbrt.so"]
        torchscript_dll_names = ["libbrt_torchscript.so"]
        lib_dll_path = [p / n for p in search_paths for n in lib_dll_names]
        torchscript_dll_path = [
            p / n for p in search_paths for n in torchscript_dll_names
        ]
    lib_found = [p for p in lib_dll_path if p.exists()]
    lib_found += [p for p in torchscript_dll_path if p.exists()]
    return lib_found

# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import os
import pathlib

# BRT root paths
BRT_PKG_PATH = pathlib.Path(__file__).parent.parent
HOME_PATH = pathlib.Path.home()
BRT_CACHE_PATH = pathlib.Path(
    os.getenv("BRT_CACHE_PATH", str(HOME_PATH / ".cache/brt"))
).absolute()
BRT_LOG_PATH = pathlib.Path(
    os.getenv("BRT_LOG_PATH", str(BRT_CACHE_PATH / "log"))
).absolute()

# brt root log file
BRT_LOG_FILENAME = str(BRT_LOG_PATH / "brainstorm.log")

# checkpoint path
BRT_CKPT_PATH = BRT_CACHE_PATH / "ckpt"
BRT_ONNX_CKPT_PATH = BRT_CKPT_PATH / "onnx"
BRT_PYTORCH_CKPT_PATH = BRT_CKPT_PATH / "pytorch"

# kernel tune
BRT_KERNEL_TUNE_LOG_PATH = BRT_LOG_PATH / "kernel_tune"
BRT_KERNEL_TEMPLATE_PATH = BRT_CACHE_PATH / "kernel_template"


BRT_CACHE_PATH.mkdir(parents=True, exist_ok=True)
BRT_LOG_PATH.mkdir(parents=True, exist_ok=True)
BRT_CKPT_PATH.mkdir(parents=True, exist_ok=True)
BRT_ONNX_CKPT_PATH.mkdir(parents=True, exist_ok=True)
BRT_PYTORCH_CKPT_PATH.mkdir(parents=True, exist_ok=True)
BRT_KERNEL_TUNE_LOG_PATH.mkdir(parents=True, exist_ok=True)
BRT_KERNEL_TEMPLATE_PATH.mkdir(parents=True, exist_ok=True)

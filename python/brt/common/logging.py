# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pathlib

_BRT_MODULES = [
    "common",
    "primitive",
    "frontend",
    "backend",
    "router",
    "runtime",
    "transform",
]
_BRT_PKG_PATH = pathlib.Path(__file__).parent.parent

__all__ = [
    "set_level_to_debug",
    "set_level_to_info",
    "set_level_to_warn",
    "set_level_to_error",
    "set_modules_level",
    "get_module_logger",
]

def _to_list(modules):
    if isinstance(modules, tuple):
        return list(modules)
    elif isinstance(modules, list):
        return modules
    else:
        return [modules]


def set_level_to_debug():
    for logger_name in _BRT_MODULES:
        m_logger = logging.getLogger(logger_name)
        m_logger.setLevel(logging.DEBUG)


def set_level_to_info():
    for logger_name in _BRT_MODULES:
        m_logger = logging.getLogger(logger_name)
        m_logger.setLevel(logging.INFO)


def set_level_to_warn():
    for logger_name in _BRT_MODULES:
        m_logger = logging.getLogger(logger_name)
        m_logger.setLevel(logging.WARN)


def set_level_to_error():
    for logger_name in _BRT_MODULES:
        m_logger = logging.getLogger(logger_name)
        m_logger.setLevel(logging.ERROR)


def set_modules_level(modules, level):
    if modules is "BRT":
        modules = _BRT_MODULES
    else:
        modules = _to_list(modules)
    for module in modules:
        if module in _BRT_MODULES:
            m_logger = logging.getLogger(module)
            m_logger.setLevel(level=level)
        else:
            raise ValueError(f"{module} is not a valid module for setting logger level")


def get_module_logger(file_path: str):
    file_path = pathlib.Path(file_path)
    module = file_path.relative_to(_BRT_PKG_PATH).parts[0]
    if module in _BRT_MODULES:
        m_logger = logging.getLogger(module)
        return m_logger
    else:
        raise ValueError(f"{module} is not a valid module for getting logger")

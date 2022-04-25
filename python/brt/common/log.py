# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pathlib

_BRT_MODULES = [
    "user",
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
    "set_level",
    "get_logger",
]


def _to_list(modules):
    if isinstance(modules, tuple):
        return list(modules)
    elif isinstance(modules, list):
        return modules
    else:
        return [modules]


def set_level_to_debug():
    for module in _BRT_MODULES:
        m_logger = logging.getLogger(f"brainstorm.{module}")
        m_logger.setLevel(logging.DEBUG)


def set_level_to_info():
    for module in _BRT_MODULES:
        m_logger = logging.getLogger(f"brainstorm.{module}")
        m_logger.setLevel(logging.INFO)


def set_level_to_warn():
    for module in _BRT_MODULES:
        m_logger = logging.getLogger(f"brainstorm.{module}")
        m_logger.setLevel(logging.WARN)


def set_level_to_error():
    for module in _BRT_MODULES:
        m_logger = logging.getLogger(f"brainstorm.{module}")
        m_logger.setLevel(logging.ERROR)


def set_level(modules, level):
    if modules is "BRT":
        modules = _BRT_MODULES
    else:
        modules = _to_list(modules)
    for module in modules:
        if module in _BRT_MODULES:
            print(f"setting logger for brainstorm.{module} to {level} level")
            m_logger = logging.getLogger(f"brainstorm.{module}")
            m_logger.setLevel(level=level)

        else:
            raise ValueError(
                f"{module} is not a valid module for setting brainstorm logger level"
            )


def get_logger(file_path: str = None) -> logging.Logger:
    if file_path is None:
        module = "user"
    else:
        file_path = pathlib.Path(file_path)
        module = file_path.relative_to(_BRT_PKG_PATH).parts[0]
    if module in _BRT_MODULES:
        m_logger = logging.getLogger(f"brainstorm.{module}")
        return m_logger
    else:
        raise ValueError(
            f"{module} is not a valid module for getting brainstorm logger"
        )

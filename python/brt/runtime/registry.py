# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Callable, Dict, Type, TypeVar

from brt.common import log

logger = log.get_logger(__file__)

T = TypeVar("T")


class Registry:
    cls_registries: Dict[T, Dict[str, T]] = {}
    func_registries: Dict[str, Callable] = {}

    @classmethod
    def register_cls(cls, sub_cls_type: str, base_cls: T) -> Callable:
        def register_func(sub_cls) -> Type[T]:
            if base_cls not in cls.cls_registries:
                cls.cls_registries[base_cls] = {}

            if sub_cls_type in cls.cls_registries[base_cls]:
                logger.warning(f"{sub_cls_type} is already registered, overwrite it")

            if not issubclass(sub_cls, base_cls):
                raise ValueError(f"Fabric: {sub_cls} is not a subclass of {base_cls}")

            cls.cls_registries[base_cls][sub_cls_type] = sub_cls
            return sub_cls

        return register_func

    @classmethod
    def get_cls(cls, sub_cls_type, base_cls) -> Type[T]:
        if base_cls not in cls.cls_registries:
            raise ValueError(f"{base_cls} is not registered")

        if sub_cls_type not in cls.cls_registries[base_cls]:
            return None

        sub_cls = cls.cls_registries[base_cls][sub_cls_type]

        return sub_cls

    @classmethod
    def cls_exists(cls, sub_cls, base_cls) -> bool:
        if (
            issubclass(sub_cls, base_cls)
            and sub_cls in cls.cls_registries[base_cls].values()
        ):
            return True
        return False

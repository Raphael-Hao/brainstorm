# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Callable, Dict, Type, TypeVar, List

from brt.runtime import log

logger = log.get_logger(__file__)

T = TypeVar("T")


class Registry:
    sub_cls_registries: Dict[T, Dict[str, T]] = {}
    func_registries: Dict[str, Callable] = {}
    cls_registries: Dict[str, List[T]] = {}

    @classmethod
    def register_sub_cls(cls, sub_cls_type: str, base_cls: T) -> Callable:
        def register_func(sub_cls) -> Type[T]:
            if base_cls not in cls.sub_cls_registries:
                cls.sub_cls_registries[base_cls] = {}

            if sub_cls_type in cls.sub_cls_registries[base_cls]:
                logger.warning(f"{sub_cls_type} is already registered, overwrite it.")

            if not issubclass(sub_cls, base_cls):
                raise ValueError(f"{sub_cls} is not a subclass of {base_cls}.")

            cls.sub_cls_registries[base_cls][sub_cls_type] = sub_cls
            return sub_cls

        return register_func

    @classmethod
    def get_sub_cls(cls, sub_cls_type, base_cls) -> Type[T]:
        if base_cls not in cls.sub_cls_registries:
            raise ValueError(f"{base_cls} is not registered.")

        if sub_cls_type not in cls.sub_cls_registries[base_cls]:
            return None

        sub_cls = cls.sub_cls_registries[base_cls][sub_cls_type]

        return sub_cls

    @classmethod
    def sub_cls_exists_and_registered(cls, sub_cls, base_cls) -> bool:
        if base_cls not in cls.sub_cls_registries:
            raise ValueError(
                f"No base class of {base_cls} exists in the registry, register the base class first."
            )
        if (
            issubclass(sub_cls, base_cls)
            and sub_cls in cls.sub_cls_registries[base_cls].values()
        ):
            return True

        return False

    @classmethod
    def register_cls(cls, cls_type: str) -> Callable:
        def register_func(registered_cls) -> Type[T]:
            if cls_type not in cls.cls_registries:
                cls.cls_registries[cls_type] = []

            if registered_cls in cls.cls_registries[cls_type]:
                logger.warning(f"{cls_type} is already registered, overwrite it.")

            cls.cls_registries[cls_type] = registered_cls

            return registered_cls

        return register_func

    @classmethod
    def cls_exists_and_registered(cls, registered_cls, cls_type: str) -> bool:

        if cls_type not in cls.cls_registries:
            raise ValueError(
                f"No key of {cls_type} exists in the registry, register a key for this class type first."
            )

        if registered_cls in cls.cls_registries[cls_type]:
            return True

        return False




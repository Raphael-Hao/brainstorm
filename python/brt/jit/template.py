# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import pathlib

from brt.common import BRT_KERNEL_TEMPLATE_PATH, BRT_KERNEL_TUNE_LOG_PATH

from .module_func import ModuleFunction

class Templator:
    template_pool = dict()

    @classmethod
    def get_global_function(cls, kernel_name: str):
        template = cls.template_pool.get(kernel_name)
        if template is None:
            template_path = BRT_KERNEL_TEMPLATE_PATH / (kernel_name + ".cu")
            if template_path.exists():
                template = template_path.read_text()
                cls.template_pool[kernel_name] = template
            else:
                # TODO get the template by using the TVMTuner
                raise NotImplementedError(
                    "not implemented yet by getting template from TVMTuner"
                )
        return ModuleFunction(template)

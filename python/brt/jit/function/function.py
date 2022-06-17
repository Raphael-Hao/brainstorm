# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from brt.common import log

logger = log.get_logger(__file__)
from ..compiler import CUDACompiler

__all__ = ["Function"]


class Function:
    c_api_decorator = 'extern "C" '

    def __init__(self) -> None:
        pass

    def append_code(self, code: str, end=False):
        formated_code = code
        self.clean_code += formated_code
        if end == True:
            formated_code += self.new_line()
        return formated_code

    def add_single_c_api(self):
        formated_code = Function.c_api_decorator
        self.clean_code += formated_code
        return formated_code

    def new_c_api_block(self):
        formated_code = Function.c_api_decorator + "{\n"
        self.clean_code += formated_code
        return formated_code

    def end_c_api_block(self):
        formated_code = '} // extern "C"\n'
        self.clean_code += formated_code
        return formated_code

    def add_codeblock(self, codeblock: str):
        formated_code = codeblock
        self.clean_code += formated_code
        formated_code += self.new_line()
        return formated_code

    def add_line_with_indent(self, code: str, end=False) -> str:
        formated_code = "  " * self.indent
        formated_code += code
        self.clean_code += formated_code
        if end == True:
            formated_code += self.new_line()
        return formated_code

    def new_line(self):
        formated_code = "\n"
        self.clean_code += formated_code
        return formated_code

    def new_codeblock(self):
        formated_code = "{\n"
        self.clean_code += formated_code
        self.indent += 1
        return formated_code

    def close_codeblock(self):
        self.indent -= 1
        formated_code = "  " * self.indent
        formated_code += "}\n"
        self.clean_code += formated_code
        return formated_code

    def verify_code(self):
        try:
            assert self.indent == 0
        except AssertionError:
            logger.exception("Code verify failed")

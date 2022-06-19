import unittest

import torch
from brt.jit import CUDACompiler
from brt.jit.function import (
    HeteroFusedFunction,
    HomoFusedFunction,
    HorizFusedFunction,
    ModuleFunction,
)


class JitFunctionTest(unittest.TestCase):
    def run_test(self, function, input):
        pass

    def test_moduel_function(self):
        module_func = ModuleFunction("Linear", method="forward", input_infos={})
        pass

    def test_horiz_fused_function(self):
        candidates = [ModuleFunction("Linear", method="forward") for _ in range(10)]
        fused_function = HorizFusedFunction(candidates)
        pass

    def test_hetero_fused_function(self):
        pass

    def test_homo_fused_function(self):
        pass


if __name__ == "__main__":
    unittest.main()

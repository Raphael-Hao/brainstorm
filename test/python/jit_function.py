import unittest

import torch
from brt.jit import (
    CUDACompiler,
    HeteroFusedModuleFunction,
    HomoFusedModuleFunction,
    HorizFusedModuleFunction,
    ModuleFunction,
)


class JitFunctionTest(unittest.TestCase):
    def run_test(self, function, input):
        pass
    
    def test_moduel_function(self):
        ModuleFunction("Linear", input_infos={})
        pass

    def test_horiz_fused_function(self):
        pass

    def test_hetero_fused_function(self):
        pass

    def test_homo_fused_function(self):
        pass


if __name__ == "__main__":
    unittest.main()

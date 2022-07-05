import unittest

import torch
import torch.nn as nn
from brt.jit import CUDACompiler, make_jit_function, make_jit_kernel


class JitFunctionTest(unittest.TestCase):
    def run_test(self, function, input):
        pass

    def test_moduel_kernel(self):
        linear = nn.Linear(512, 1024)
        sample_inputs = torch.randn((16, 512))
        module_kernel = make_jit_kernel(linear, sample_inputs=sample_inputs)
        pt_out_cpu = linear(sample_inputs)
        sample_inputs = sample_inputs.cuda()
        linear = linear.cuda()
        pt_out_gpu = linear(sample_inputs)
        
        brt_out_gpu = torch.randn_like(
            pt_out_gpu, dtype=pt_out_gpu.dtype, device=pt_out_gpu.device
        )
        module_kernel(sample_inputs, linear.weight.cuda(), brt_out_gpu, linear.bias.cuda())
        brt_out_cpu = brt_out_gpu.cpu()
        print(pt_out_cpu)
        print(pt_out_gpu)
        print(brt_out_gpu)
        self.assertTrue(torch.allclose(brt_out_cpu, pt_out_cpu, atol=1e-6))

    def test_horiz_fused_function(self):

        candidates = [ModuleKernel("Linear", method="forward") for _ in range(10)]
        fused_function = HorizFusedKernel(candidates)
        pass

    def test_hetero_fused_function(self):
        pass

    def test_homo_fused_function(self):
        pass


if __name__ == "__main__":
    unittest.main()

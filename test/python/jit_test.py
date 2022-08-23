import unittest

import torch
import torch.nn as nn
from brt.jit import make_jit_function, make_jit_kernel


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
        module_kernel(
            sample_inputs, linear.weight.cuda(), brt_out_gpu, linear.bias.cuda()
        )
        brt_out_cpu = brt_out_gpu.cpu()
        self.assertTrue(torch.allclose(brt_out_cpu, pt_out_cpu, atol=1e-6))

    def test_horiz_fused_function(self):
        linears = nn.ModuleList(nn.Linear(512, 1024) for _ in range(4)).cuda()
        sample_inputs = [torch.randn((16, 512)).cuda() for _ in range(4)]
        sample_outputs = [linears[i](sample_inputs[i]) for i in range(4)]

        horiz_fused_kernel = make_jit_kernel(
            linears, sample_inputs=sample_inputs, opt_level="horiz_fuse"
        )

        horiz_fused_inputs = []
        for i in range(4):
            horiz_fused_inputs.append(sample_inputs[i])
            horiz_fused_inputs.append(linears[i].weight)
            horiz_fused_inputs.append(torch.empty_like(sample_outputs[i]))
            horiz_fused_inputs.append(linears[i].bias)

        horiz_fused_kernel(*horiz_fused_inputs)
        for i in range(4):
            print(horiz_fused_inputs[2 + i * 4])
        for i in range(4):
            self.assertTrue(
                torch.allclose(
                    sample_outputs[i], horiz_fused_inputs[2 + i * 4], atol=1e-6
                )
            )

    def test_hetero_fused_function(self):
        linears = nn.ModuleList(nn.Linear(512, 1024) for _ in range(4)).cuda()
        sample_inputs = [torch.randn((16, 512)).cuda() for _ in range(4)]
        hetero_fused_kernel = make_jit_kernel(
            linears, sample_inputs=sample_inputs, opt_level="hetero_fuse"
        )
        active_blocks = [0, 1, 0, 1]

        hetero_fused_inputs = []

        for i, active in enumerate(active_blocks):
            if active == 0:
                hetero_fused_inputs.append(torch.randn((0, 512)).cuda())
                hetero_fused_inputs.append(linears[i].weight)
                hetero_fused_inputs.append(torch.empty((0, 1024)).cuda())
                hetero_fused_inputs.append(linears[i].bias)
            else:
                hetero_fused_inputs.append(sample_inputs[i])
                hetero_fused_inputs.append(linears[i].weight)
                hetero_fused_inputs.append(torch.empty((16, 1024)).cuda())
                hetero_fused_inputs.append(linears[i].bias)
        hetero_fused_kernel(*hetero_fused_inputs, active_blocks=active_blocks)
        for i, active in enumerate(active_blocks):
            if active != 0:
                sample_out = linears[i](hetero_fused_inputs[i * 4])
                self.assertTrue(
                    torch.allclose(
                        sample_out, hetero_fused_inputs[2 + i * 4], atol=1e-6
                    )
                )

    def test_homo_fused_function(self):
        linears = nn.ModuleList(nn.Linear(512, 1024) for _ in range(4)).cuda()
        sample_inputs = [torch.randn((16, 512)).cuda() for _ in range(4)]
        hetero_fused_kernel = make_jit_kernel(
            linears, sample_inputs=sample_inputs, opt_level="homo_fuse"
        )
        active_blocks = [0, 1, 0, 1]

        hetero_fused_inputs = []

        for i, active in enumerate(active_blocks):
            if active == 0:
                hetero_fused_inputs.append(torch.randn((0, 512)).cuda())
                hetero_fused_inputs.append(linears[i].weight)
                hetero_fused_inputs.append(torch.empty((0, 1024)).cuda())
                hetero_fused_inputs.append(linears[i].bias)
            else:
                hetero_fused_inputs.append(sample_inputs[i])
                hetero_fused_inputs.append(linears[i].weight)
                hetero_fused_inputs.append(torch.empty((16, 1024)).cuda())
                hetero_fused_inputs.append(linears[i].bias)
        hetero_fused_kernel(*hetero_fused_inputs, active_blocks=active_blocks)
        for i, active in enumerate(active_blocks):
            if active != 0:
                sample_out = linears[i](hetero_fused_inputs[i * 4])
                self.assertTrue(
                    torch.allclose(
                        sample_out, hetero_fused_inputs[2 + i * 4], atol=1e-6
                    )
                )


if __name__ == "__main__":
    unittest.main()

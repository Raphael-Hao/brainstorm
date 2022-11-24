import unittest

import torch
import torch.nn as nn
from brt.jit import make_jit_function, make_jit_kernel

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


class CUDATimer:
    def __init__(self) -> None:
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.stream = torch.cuda.default_stream()

    def set_stream(self, stream: torch.cuda.Stream) -> None:
        self.stream = stream

    def start(self) -> None:
        torch.cuda.synchronize()
        self.start_event.record(self.stream)

    def stop(self, iterations, timer_name) -> None:
        self.end_event.record(self.stream)
        self.end_event.synchronize()
        print(
            "{} elapsed time: {:.3f}".format(
                timer_name, self.start_event.elapsed_time(self.end_event) / iterations
            )
        )


class JitKernelTest(unittest.TestCase):
    def test_module_kernel(self):
        cuda_timer = CUDATimer()
        linear = nn.Linear(512, 1024).cuda()
        sample_inputs = torch.randn((16, 512)).cuda()
        pt_out_gpu = linear(sample_inputs)
        cuda_timer.start()
        for i in range(100):
            pt_out_gpu = linear(sample_inputs)
        cuda_timer.stop(100, "pt_out_gpu")

        module_kernel = make_jit_kernel(linear, sample_inputs=sample_inputs)
        brt_out_gpu = torch.randn_like(
            pt_out_gpu, dtype=pt_out_gpu.dtype, device=pt_out_gpu.device
        )
        module_kernel(
            sample_inputs, linear.weight.cuda(), brt_out_gpu, linear.bias.cuda()
        )
        cuda_timer.start()
        for i in range(100):
            brt_out_gpu = torch.empty_like(
                pt_out_gpu, dtype=pt_out_gpu.dtype, device=pt_out_gpu.device
            )
            module_kernel(
                sample_inputs, linear.weight.cuda(), brt_out_gpu, linear.bias.cuda()
            )
        cuda_timer.stop(100, "brt_out_gpu")

        self.assertTrue(torch.allclose(brt_out_gpu, pt_out_gpu, atol=1e-6))

    def test_horiz_fused_kernel(self):
        cuda_timer = CUDATimer()
        linears = nn.ModuleList(nn.Linear(512, 1024) for _ in range(4)).cuda()
        sample_inputs = [torch.randn((16, 512)).cuda() for _ in range(4)]

        sample_outputs = [linears[i](sample_inputs[i]) for i in range(4)]

        cuda_timer.start()
        for i in range(10):
            sample_outputs = [linears[i](sample_inputs[i]) for i in range(4)]
        cuda_timer.stop(10, "pt_serial")

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

        cuda_timer.start()
        for i in range(10):
            horiz_fused_inputs = []
            for i in range(4):
                horiz_fused_inputs.append(sample_inputs[i])
                horiz_fused_inputs.append(linears[i].weight)
                horiz_fused_inputs.append(torch.empty_like(sample_outputs[i]))
                horiz_fused_inputs.append(linears[i].bias)

            horiz_fused_kernel(*horiz_fused_inputs)
        cuda_timer.stop(10, "brt_horiz_fusion")

        for i in range(4):
            self.assertTrue(
                torch.allclose(sample_outputs[i], horiz_fused_inputs[2 + i * 4], atol=1e-6)
            )

    def test_hetero_fused_kernel(self):
        cuda_timer = CUDATimer()
        linears = nn.ModuleList(nn.Linear(512, 1024) for _ in range(4)).cuda()
        sample_inputs = [torch.randn((16, 512)).cuda() for _ in range(4)]
        hetero_fused_kernel = make_jit_kernel(
            linears, sample_inputs=sample_inputs, opt_level="hetero_fuse"
        )
        active_blocks = [1, 1, 1, 1]

        hetero_fused_inputs = []

        for i, active in enumerate(active_blocks):
            if active == 0:
                hetero_fused_inputs.append(torch.randn((0, 512), device="cuda"))
                hetero_fused_inputs.append(linears[i].weight)
                hetero_fused_inputs.append(torch.empty((0, 1024), device="cuda"))
                hetero_fused_inputs.append(linears[i].bias)
            else:
                hetero_fused_inputs.append(sample_inputs[i])
                hetero_fused_inputs.append(linears[i].weight)
                hetero_fused_inputs.append(torch.empty((16, 1024), device="cuda"))
                hetero_fused_inputs.append(linears[i].bias)
        hetero_fused_kernel(*hetero_fused_inputs, active_blocks=active_blocks)

        cuda_timer.start()
        for i in range(100):
            for i, active in enumerate(active_blocks):
                if active == 0:
                    hetero_fused_inputs[2 + i * 4] = torch.empty(
                        (0, 1024), device="cuda"
                    )
                else:
                    hetero_fused_inputs[2 + i * 4] = torch.empty(
                        (16, 1024), device="cuda"
                    )
            hetero_fused_kernel(*hetero_fused_inputs, active_blocks=active_blocks)
        cuda_timer.stop(100, "brt_hetero_fusion")

        for i, active in enumerate(active_blocks):
            if active != 0:
                sample_out = linears[i](hetero_fused_inputs[i * 4])
                self.assertTrue(
                    torch.allclose(
                        sample_out, hetero_fused_inputs[2 + i * 4], atol=1e-6
                    )
                )

    def test_homo_fused_kernel(self):
        cuda_timer = CUDATimer()
        linears = nn.ModuleList(nn.Linear(512, 1024) for _ in range(4)).cuda()
        sample_inputs = [torch.randn((2**i, 512)).cuda() for i in range(5)]
        homo_fused_kernel = make_jit_kernel(
            linears, sample_inputs=sample_inputs, opt_level="homo_fuse"
        )

        shared_inputs = [
            torch.randn((16, 512), device="cuda"),
            torch.randn((16, 1024), device="cuda"),
        ]
        standalone_inputs = []
        for linear in linears:
            standalone_inputs.append(linear.weight)
            standalone_inputs.append(linear.bias)
        capacities = [2, 8, 4, 2]
        homo_fused_kernel(shared_inputs, standalone_inputs, capacities)
        cuda_timer.start()
        for i in range(100):
            shared_inputs[1] = torch.empty((16, 1024), device="cuda")
            homo_fused_kernel(shared_inputs, standalone_inputs, capacities)
        cuda_timer.stop(100, "brt_homo_fusion")

        start_idx = 0
        outputs = []
        for i, cap in enumerate(capacities):
            inputs = shared_inputs[0][start_idx : start_idx + cap, :]
            outputs.append(linears[i](inputs))
            start_idx += cap
        outputs = torch.cat(outputs)

        cuda_timer.start()
        for i in range(100):
            start_idx = 0
            outputs = []
            for i, cap in enumerate(capacities):
                inputs = shared_inputs[0][start_idx : start_idx + cap, :]
                outputs.append(linears[i](inputs))
                start_idx += cap
            outputs = torch.cat(outputs)
        cuda_timer.stop(100, "brt_serial_fusion")

        self.assertTrue(torch.allclose(outputs, shared_inputs[1], atol=1e-6))


class JitFunctionTest(unittest.TestCase):
    def test_module_kernel(self):
        cuda_timer = CUDATimer()
        linear = nn.Linear(512, 1024).cuda()
        sample_inputs = torch.randn((16, 512)).cuda()
        pt_out_gpu = linear(sample_inputs)
        cuda_timer.start()
        for i in range(100):
            pt_out_gpu = linear(sample_inputs)
        cuda_timer.stop(100, "pt_out_gpu")

        module_function = make_jit_function(linear, sample_inputs=sample_inputs)
        brt_out_gpu = module_function.apply(
            sample_inputs, linear.weight.cuda(), linear.bias.cuda()
        )
        cuda_timer.start()
        for i in range(100):
            brt_out_gpu = torch.empty_like(
                pt_out_gpu, dtype=pt_out_gpu.dtype, device=pt_out_gpu.device
            )
            brt_out_gpu = module_function.apply(
                sample_inputs, linear.weight.cuda(), linear.bias.cuda()
            )
        cuda_timer.stop(100, "brt_out_gpu")

        self.assertTrue(torch.allclose(brt_out_gpu[0], pt_out_gpu, atol=1e-6))

    def test_horiz_fused_function(self):
        cuda_timer = CUDATimer()
        linears = nn.ModuleList(nn.Linear(512, 1024) for _ in range(4)).cuda()
        sample_inputs = [torch.randn((16, 512)).cuda() for _ in range(4)]

        sample_outputs = [linears[i](sample_inputs[i]) for i in range(4)]

        cuda_timer.start()
        for i in range(10):
            sample_outputs = [linears[i](sample_inputs[i]) for i in range(4)]
        cuda_timer.stop(10, "pt_serial")

        horiz_fused_function = make_jit_function(
            linears, sample_inputs=sample_inputs, opt_level="horiz_fuse"
        )

        horiz_fused_inputs = []
        for i in range(4):
            horiz_fused_inputs.append(sample_inputs[i])
            horiz_fused_inputs.append(linears[i].weight)
            horiz_fused_inputs.append(linears[i].bias)

        brt_outputs = horiz_fused_function.apply(*horiz_fused_inputs)

        cuda_timer.start()
        for i in range(10):
            horiz_fused_inputs = []
            for i in range(4):
                horiz_fused_inputs.append(sample_inputs[i])
                horiz_fused_inputs.append(linears[i].weight)
                horiz_fused_inputs.append(linears[i].bias)

            brt_outputs = horiz_fused_function.apply(*horiz_fused_inputs)
        cuda_timer.stop(10, "brt_horiz_fusion")

        for i in range(4):
            self.assertTrue(
                torch.allclose(sample_outputs[i], brt_outputs[i], atol=1e-6)
            )

    def test_hetero_fused_function(self):
        cuda_timer = CUDATimer()
        linears = nn.ModuleList(nn.Linear(512, 1024) for _ in range(4)).cuda()
        sample_inputs = [torch.randn((16, 512)).cuda() for _ in range(4)]
        hetero_fused_function = make_jit_function(
            linears, sample_inputs=sample_inputs, opt_level="hetero_fuse"
        )
        active_blocks = [1, 0, 1, 0]

        hetero_fused_inputs = []

        for i, active in enumerate(active_blocks):
            if active == 0:
                hetero_fused_inputs.append(torch.randn((0, 512), device="cuda"))
                hetero_fused_inputs.append(linears[i].weight)
                hetero_fused_inputs.append(linears[i].bias)
            else:
                hetero_fused_inputs.append(sample_inputs[i])
                hetero_fused_inputs.append(linears[i].weight)
                hetero_fused_inputs.append(linears[i].bias)
        brt_outs = hetero_fused_function.apply(*hetero_fused_inputs, active_blocks=active_blocks)

        cuda_timer.start()
        for i in range(100):
            brt_outs = hetero_fused_function.apply(*hetero_fused_inputs, active_blocks=active_blocks)
        cuda_timer.stop(100, "brt_hetero_fusion")

        for i, active in enumerate(active_blocks):
            if active != 0:
                sample_out = linears[i](hetero_fused_inputs[i * 3])
                self.assertTrue(
                    torch.allclose(sample_out, brt_outs[i], atol=1e-6)
                )

    def test_homo_fused_function(self):
        cuda_timer = CUDATimer()
        linears = nn.ModuleList(nn.Linear(512, 1024) for _ in range(4)).cuda()
        sample_inputs = [torch.randn((2**i, 512)).cuda() for i in range(5)]
        homo_fused_function = make_jit_function(
            linears, sample_inputs=sample_inputs, opt_level="homo_fuse"
        )

        shared_inputs = [
            torch.randn((16, 512), device="cuda"),
        ]
        standalone_inputs = []
        for linear in linears:
            standalone_inputs.append(linear.weight)
            standalone_inputs.append(linear.bias)
        capacities = [2, 8, 4, 2]
        brt_outputs = homo_fused_function.apply(*shared_inputs, *standalone_inputs, capacities=capacities)
        cuda_timer.start()
        for i in range(100):
            brt_outputs = homo_fused_function.apply(*shared_inputs, *standalone_inputs, capacities=capacities)
        cuda_timer.stop(100, "brt_homo_fusion")

        start_idx = 0
        outputs = []
        for i, cap in enumerate(capacities):
            inputs = shared_inputs[0][start_idx : start_idx + cap, :]
            outputs.append(linears[i](inputs))
            start_idx += cap

        cuda_timer.start()
        for i in range(100):
            start_idx = 0
            outputs = []
            for i, cap in enumerate(capacities):
                inputs = shared_inputs[0][start_idx : start_idx + cap, :]
                outputs.append(linears[i](inputs))
                start_idx += cap
        cuda_timer.stop(100, "brt_serial_fusion")

        for i in range(len(capacities)):
            self.assertTrue(torch.allclose(outputs[i], brt_outputs[1], atol=1e-6))



if __name__ == "__main__":
    unittest.main()

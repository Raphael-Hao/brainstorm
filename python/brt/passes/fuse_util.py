from typing import Type, List, Union, Tuple
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from brt.jit import make_jit_kernel
from brt.runtime.benchmark import (
    BenchmarkArgumentManager,
    Benchmarker,
    CUDATimer,
    MemoryStats,
    profile,
)

class _ObjectiveFuncContext:
    curr = "fastest"

    def __init__(self, objective_func: str) -> None:
        self.prev = type(self).curr
        type(self).curr = objective_func

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        type(self).curr = self.prev


def set_objective_func(objective_func) -> _ObjectiveFuncContext:
    return _ObjectiveFuncContext(objective_func)


def get_objective_func() -> str:
    return _ObjectiveFuncContext.curr


class TunedKernel(nn.Module):
    # Support Conv2dBias, Conv2dBiasReLU, Conv2dBiasSigmoid and AdaptiveAvgPool2d
    def __init__(
        self,
        model: nn.Module,
        input_shape: torch.Size,
        output_shape: torch.Size,
    ):
        super().__init__()
        if isinstance(model, nn.Sequential) and len(model) == 1:
            model = model[0]
        self.input_shape = input_shape
        self.output_shape = output_shape
        self._extra_repr = str(model)
        # Generate model name
        if isinstance(model, nn.Conv2d) and model.bias is not None:
            self.module_name = "Conv2dBias"
        elif isinstance(model, nn.Sequential):
            if isinstance(model[0], nn.Conv2d) and model[0].bias is not None:
                if len(model) == 1:
                    self.module_name = "Conv2dBias"
                elif len(model) == 2 and isinstance(model[1], nn.ReLU):
                    self.module_name = "Conv2dBiasReLU"
                elif len(model) == 2 and isinstance(model[1], nn.Sigmoid):
                    self.module_name = "Conv2dBiasSigmoid"
                elif len(model) == 3 and isinstance(model[1], nn.BatchNorm2d) and isinstance(model[2], nn.ReLU):
                    self.module_name = "Conv2dBiasBatchNormReLU"
                else:
                    raise NotImplementedError(f"{model}")
            elif isinstance(model[0], nn.Conv2d) and model[0].bias is None:
                if len(model) == 1:
                    self.module_name = "Conv2d"
                elif len(model) == 2 and isinstance(model[1], nn.ReLU):
                    self.module_name = "Conv2dReLU"
                elif len(model) == 2 and isinstance(model[1], nn.Sigmoid):
                    self.module_name = "Conv2dSigmoid"
                elif len(model) == 3 and isinstance(model[1], nn.BatchNorm2d) and isinstance(model[2], nn.ReLU):
                    self.module_name = "Conv2dBatchNormReLU"
                else:
                    raise NotImplementedError(f"{model}")
            
            else:
                raise NotImplementedError(f"{model}")
        else:
            self.module_name = "ERROR"
            raise NotImplementedError(f"{model}")
        # Make fused kernel
        for name, tensor in model.named_parameters():
            self.register_parameter(name.replace(".", "_"), tensor)
        for name, tensor in model.named_buffers():
            self.register_buffer(name.replace(".", "_"), tensor)
        sample_input = torch.randn(self.input_shape).cuda()
        model=model.cuda()
        self.fused_kernel = make_jit_kernel(model, sample_input)
        # Make fused inputs
        self.inputs_templete = {}
        self.inputs_templete["forward"] = []
        if self.module_name == "Conv2dBias":
            self.inputs_templete["forward"].extend(
                [
                    None,
                    self.get_parameter(f"weight"),
                    # None,
                    torch.empty(self.output_shape, device="cuda"),
                    self.get_parameter(f"bias"),
                ]
            )
        elif self.module_name in ["Conv2dBiasReLU", "Conv2dBiasSigmoid"]:
            ## TODO
            self.inputs_templete["forward"].extend(
                [
                    None,
                    self.get_parameter(f"0_weight"),
                    # None,
                    torch.empty(self.output_shape, device="cuda"),
                    self.get_parameter(f"0_bias"),
                ]
            )
        elif self.module_name in ["Conv2dBiasBatchNormReLU","Conv2dBatchNormReLU"]:
            w_conv = self.get_parameter(f"0_weight").clone().view(model[0].out_channels, -1).cuda()
            w_bn = torch.diag(self.get_parameter(f"1_weight").div(torch.sqrt(model[1].eps+ model[1].running_var))).cuda()
            fusedweight=( torch.mm(w_bn, w_conv).view(model[0].weight.size()).cuda() )
            if model[0].bias is not None and self.module_name=="Conv2dBiasBatchNormReLU":
                b_conv = model[0].bias
                self.inputs_templete["forward"].extend(
                [
                    None,
                    fusedweight,
                    # None,
                    torch.empty(self.output_shape, device="cuda"),
                    self.get_parameter(f"0_bias"),
                    self.get_parameter(f"1_bias"),
                ]
            )
            elif model[0].bias is None and self.module_name=="Conv2dBatchNormReLU":
                b_conv = torch.zeros(model[0].weight.size(0) ).cuda()
                b_conv = torch.mm(w_bn, b_conv.view(-1, 1)).view(-1).cuda()
                b_bn = model[1].bias - model[1].weight.mul(model[1].running_mean).div(torch.sqrt(model[1].running_var + model[1].eps)).cuda()
                fusedbias=( b_conv + b_bn ).cuda()
                self.inputs_templete["forward"].extend(
                [
                    None,
                    fusedweight,
                    # None,
                    torch.empty(self.output_shape, device="cuda"),
                    fusedbias,
                ]
            )
            else :
                raise NotImplementedError(f"{self.module_name}")
                
            
            
            
            # self.inputs_templete["forward"].extend(
            #     [
            #         None,
            #         fusedweight,
            #         # None,
            #         torch.empty(self.output_shape, device="cuda"),
            #         fusedbias,
            #     ]
            # )
        elif self.module_name in ["Conv2dBatchNormReLU"]:
            self.inputs_templete["forward"].extend(
                [
                    None,
                    self.get_parameter(f"0_weight"),
                    # None,
                    torch.empty(self.output_shape, device="cuda"),
                    self.get_parameter(f"0_bias"),
                ]
            )
        else:
            raise NotImplementedError(f"{self.module_name}")
        # Test forward & warmup
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        # if (self.forward(sample_input)).equal(model(sample_input))==0:
        #     raise NotImplementedError(f"{self.forward(sample_input)} != {model(sample_input)}")
        
        
        # import pdb; pdb.set_trace()
        self.forward(sample_input)
        # import pdb; pdb.
        # set_trace()

    def forward(self, input: torch.Tensor):
        self.inputs_templete["forward"][0] = input
        self.inputs_templete["forward"][2] = torch.empty(
            self.output_shape, device="cuda"
        )
        # print("self inputs templete",self.input_shape)
        # print("input shape",input.shape)
        if input.size(0)==0:
            import copy
            newout_shape = copy.deepcopy(self.output_shape)
            newout_shape[0]=0
            return torch.zeros(newout_shape, device="cuda")
        self.fused_kernel(*self.inputs_templete["forward"])
        output = self.inputs_templete["forward"][2]
        # self.inputs_templete["forward"][0] = None
        # self.inputs_templete["forward"][2] = None
        return output

    def extra_repr(self):
        return self._extra_repr


class FusedLayer(nn.Module):
    # Support Conv2dBias, Conv2dBiasReLU, Conv2dBiasSigmoid and AdaptiveAvgPool2d
    def __init__(
        self,
        models: Union[List[nn.Module], nn.ModuleList],
        input_shapes: List[torch.Size],
        output_shapes: List[torch.Size],
    ):
        super().__init__()
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.num_submodels = len(models)
        if not isinstance(models, nn.ModuleList):
            models = nn.ModuleList(models)
        for i in range(self.num_submodels):
            if isinstance(models[i], nn.Sequential) and len(models[i]) == 1:
                models[i] = models[i][0]
        self._extra_repr = "\n".join([str(m) for m in models])
        assert len(input_shapes) == self.num_submodels
        assert len(output_shapes) == self.num_submodels
        # Generate model name
        self.module_names = []
        for model in models:
            if isinstance(model[0], nn.Conv2d) and model[0].bias is not None:
                if len(model) == 1:
                    self.module_names.append("Conv2dBias") 
                elif len(model) == 2 and isinstance(model[1], nn.ReLU):
                    self.module_names.append("Conv2dBiasReLU")
                elif len(model) == 2 and isinstance(model[1], nn.Sigmoid):
                    self.module_names.append("Conv2dBiasSigmoid")
                elif len(model) == 3 and isinstance(model[1], nn.BatchNorm2d) and isinstance(model[2], nn.ReLU):
                    self.module_names.append("Conv2dBiasBatchNormReLU")
                else:
                    raise NotImplementedError(f"{models}")
            elif isinstance(model[0], nn.Conv2d) and model[0].bias is None:
                if len(model) == 1:
                    self.module_names.append("Conv2d")
                elif len(model) == 2 and isinstance(model[1], nn.ReLU):
                    self.module_names.append("Conv2dReLU")
                elif len(model) == 2 and isinstance(model[1], nn.Sigmoid):
                    self.module_names.append("Conv2dSigmoid")
                elif len(model) == 3 and isinstance(model[1], nn.BatchNorm2d) and isinstance(model[2], nn.ReLU):
                    self.module_names.append("Conv2dBatchNormReLU")
                else:
                    raise NotImplementedError(f"{model}")
            
            else:
                raise NotImplementedError(f"{models}")
        # Make fused kernel
        for i, model in enumerate(models):
            for name, tensor in model.named_parameters(f"m{i}"):
                self.register_parameter(name.replace(".", "_"), tensor)
            for name, tensor in model.named_buffers(f"m{i}"):
                self.register_buffer(name.replace(".", "_"), tensor)
        sample_inputs = [torch.randn(shp).cuda() for shp in input_shapes]
        self.fused_kernel = make_jit_kernel(
            models,
            sample_inputs,
            opt_level="horiz_fuse",
            objective_func=get_objective_func(),
        )
        # Make fused inputs
        self.inputs_templete = {}
        self.inputs_templete["forward"] = []
        self.input_indices = []
        self.output_indices = []
        for i, (module_name, output_shape,iter_module) in enumerate(zip(self.module_names, self.output_shapes,models)):
            # TODO: module_name parser
            if module_name == "Conv2dBias":
                self.input_indices.append(len(self.inputs_templete["forward"]))
                self.output_indices.append(len(self.inputs_templete["forward"]) + 2)
                self.inputs_templete["forward"].extend(
                    [
                        None,
                        self.get_parameter(f"m{i}_weight"),
                        # None,
                        torch.empty(output_shape, device="cuda"),
                        self.get_parameter(f"m{i}_bias"),
                    ]
                )
            elif module_name in ["Conv2dBiasReLU", "Conv2dBiasSigmoid"]:
                self.input_indices.append(len(self.inputs_templete["forward"]))
                self.output_indices.append(len(self.inputs_templete["forward"]) + 2)
                self.inputs_templete["forward"].extend(
                    [
                        None,
                        self.get_parameter(f"m{i}_0_weight"),
                        # None,
                        torch.empty(output_shape, device="cuda"),
                        self.get_parameter(f"m{i}_0_bias"),
                    ]
                )
            elif module_name in ["Conv2dBiasBatchNormReLU","Conv2dBatchNormReLU"]:
                self.input_indices.append(len(self.inputs_templete["forward"]))
                self.output_indices.append(len(self.inputs_templete["forward"]) + 2)
                w_conv = self.get_parameter(f"m{i}_0_weight").clone().view(iter_module[0].out_channels, -1).cuda()
                w_bn = torch.diag(self.get_parameter(f"m{i}_1_weight").div(torch.sqrt(iter_module[1].eps+ iter_module[1].running_var))).cuda()
                fusedweight=( torch.mm(w_bn, w_conv).view(iter_module[0].weight.size()).cuda() )
                if iter_module[0].bias is not None and module_name=="Conv2dBiasBatchNormReLU":
                    b_conv = iter_module[0].bias
                    self.inputs_templete["forward"].extend(
                    [
                        None,
                        fusedweight,
                        # None,
                        torch.empty(output_shape, device="cuda"),
                        self.get_parameter(f"m{i}_0_bias"),
                        self.get_parameter(f"m{i}_1_bias"),
                    ]
                )
                elif iter_module[0].bias is None and module_name=="Conv2dBatchNormReLU":
                    b_conv = torch.zeros(iter_module[0].weight.size(0) ).cuda()
                    b_conv = torch.mm(w_bn, b_conv.view(-1, 1)).view(-1).cuda()
                    b_bn = iter_module[1].bias - iter_module[1].weight.mul(iter_module[1].running_mean).div(torch.sqrt(iter_module[1].running_var + iter_module[1].eps)).cuda()
                    fusedbias=( b_conv + b_bn ).cuda()
                    self.inputs_templete["forward"].extend(
                    [
                        None,
                        fusedweight,
                        # None,
                        torch.empty(output_shape, device="cuda"),
                        fusedbias,
                    ]
                )
                else :
                    raise NotImplementedError(f"{self.module_names}")
                
            
            else:
                raise NotImplementedError(f"{self.module_names}")
        # Test forward & warmup
        self.forward(sample_inputs)

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        
        ## TODO cope with bs==0
        for i in range(self.num_submodels):
            self.inputs_templete["forward"][self.input_indices[i]] = inputs[i]
            # self.inputs_templete["forward"][self.output_indices[i]] = torch.empty(
            #     self.output_shapes[i], device="cuda"
            # )
        
            
        self.fused_kernel(*self.inputs_templete["forward"])
        outputs = [
            self.inputs_templete["forward"][index] for index in self.output_indices
        ]
        
        
        for i,input in  enumerate(inputs):
            if input.size(0)==0:
                import copy
                newout_shape = copy.deepcopy(self.output_shapes[i])
                newout_shape[0]=0
                outputs[i]=torch.empty(newout_shape, device="cuda")
        # for i in range(self.num_submodels):
        #     self.inputs_templete["forward"][self.input_indices[i]] = None
        #     self.inputs_templete["forward"][self.output_indices[i]] = None
        return outputs

    def extra_repr(self) -> str:
        return self._extra_repr


class GroupFusedLayer(nn.ModuleList):
    def __init__(
        self,
        models: Union[List[nn.Module], nn.ModuleList],
        input_shapes: List[torch.Size],
        output_shapes: List[torch.Size],
        group_indices: List[List[int]],
    ):
        import pdb;pdb.set_trace()
        super().__init__()
        self.group_indices = group_indices
        self.num_submodels = len(models)
        self.num_sublayers = len(self.group_indices)
        self.kernels = []
        for group_index in self.group_indices:
            import pdb;pdb.set_trace()
            self.append(
                FusedLayer(
                    [models[i] for i in group_index],
                    [input_shapes[i] for i in group_index],
                    [output_shapes[i] for i in group_index],
                )
            )

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = [None for _ in range(self.num_submodels)]
        # for i, group_index in enumerate(self.group_indices):
        for group_index, sublayer in zip(self.group_indices, self):
            sub_outputs = sublayer([inputs[i] for i in group_index])
            for i, j in enumerate(group_index):
                outputs[j] = sub_outputs[i]
        return outputs

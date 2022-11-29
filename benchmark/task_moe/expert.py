# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch

class FusedExpertsNetwork(torch.nn.Module):
    def __init__(
        self,
        model_dim,
        hidden_size,
        local_experts,
        ffn_zero_group,
        mlpfp32=False,
        has_fc2_bias=True,
    ):
        super().__init__()

        fc1_weight = torch.empty(
            1, local_experts, hidden_size, model_dim
        )
        fc2_weight = torch.empty(
            1, local_experts, hidden_size, model_dim
        )
        fc1_bias = torch.empty(1, local_experts, 1, hidden_size)
        fc2_bias = (
            torch.empty(
                1,
                local_experts,
                1,
                (model_dim + sharded_count - 1) // sharded_count,
            )
            if self.has_fc2_bias
            else None
        )

        for i in range(local_experts):
            fc1 = torch.nn.Linear(model_dim, hidden_size)
            fc2 = torch.nn.Linear(
                hidden_size, model_dim, bias=self.has_fc2_bias
            )
            fc1_weight[0, i, :, :], fc1_bias[0, i, :, :] = (
                fc1.weight,
                fc1.bias,
            )
            fc2_weight[0, i, :, :] = fc2.weight.t()
            if self.has_fc2_bias:
                fc2_bias[0, i, :, :] = fc2.bias[: fc2_bias.size(-1)]

        self.model_dim, self.hidden_size, self.local_experts = (
            model_dim,
            hidden_size,
            local_experts,
        )
        self.ffn_zero_group = ffn_zero_group
        if self.ffn_zero_group is not None:
            assert self.local_experts == 1
            fc1_weight = fc1_weight.view(
                self.hidden_size, self.model_dim
            )
            fc2_weight = fc2_weight.view(
                self.hidden_size, self.model_dim
            )
            fc1_bias = fc1_bias.view(self.hidden_size)
            if self.has_fc2_bias:
                fc2_bias = fc2_bias.view(-1)
        elif self.local_experts == 1:
            fc1_weight = fc1_weight.view(
                self.hidden_size, self.model_dim
            )
            fc2_weight = fc2_weight.view(
                self.hidden_size, self.model_dim
            )
            fc1_bias = fc1_bias.view(self.hidden_size)
            if self.has_fc2_bias:
                fc2_bias = fc2_bias.view(-1)
        else:
            fc1_weight = fc1_weight.view(
                self.local_experts, self.hidden_size, self.model_dim
            )
            fc2_weight = fc2_weight.view(
                self.local_experts, self.hidden_size, self.model_dim
            )
            fc1_bias = fc1_bias.view(
                self.local_experts, 1, self.hidden_size
            )
            if self.has_fc2_bias:
                fc2_bias = fc2_bias.view(self.local_experts, 1, -1)

        self.register_parameter(
            name="fc1_weight", param=torch.nn.Parameter(fc1_weight)
        )
        self.register_parameter(
            name="fc2_weight", param=torch.nn.Parameter(fc2_weight)
        )
        self.register_parameter(
            name="fc1_bias", param=torch.nn.Parameter(fc1_bias)
        )
        if self.has_fc2_bias:
            self.register_parameter(
                name="fc2_bias", param=torch.nn.Parameter(fc2_bias)
            )

        if implicit_dropout_p:
            self.dropout_fc1 = torch.nn.Dropout(p=implicit_dropout_p)
            self.dropout_fc2 = torch.nn.Dropout(p=implicit_dropout_p)
        else:
            self.dropout_fc1 = self.dropout_fc2 = lambda x: x

    def extra_repr(self):
        return (
            "model_dim=%d, hidden_size=%d, local_experts=%d, bias=%s"
            % (
                self.model_dim,
                self.hidden_size,
                self.local_experts,
                self.fc1_bias is not None,
            )
        )

    def forward(self, x):
        if self.skip_expert:
            return x
        if fused_custom_fn is not None:
            return fused_custom_fn(self, x)

        fc1_weight, fc2_weight, fc1_bias = (
            self.fc1_weight,
            self.fc2_weight,
            self.fc1_bias,
        )
        if self.has_fc2_bias:
            fc2_bias = self.fc2_bias

        if self.ffn_zero_group is not None:
            fc1_weight = C.PreAllreduceSum.apply(
                self.ffn_zero_group, self.fc1_weight
            )
            fc2_weight = C.PreAllreduceSum.apply(
                self.ffn_zero_group, self.fc2_weight
            )
            fc1_bias = C.PreAllreduceSum.apply(
                self.ffn_zero_group, self.fc1_bias
            )
            if self.has_fc2_bias:
                fc2_bias = C.PreAllreduceSum.apply(
                    self.ffn_zero_group, self.fc2_bias
                )
                if fc2_bias.size(-1) != self.model_dim:
                    fc2_bias = fc2_bias[:, : self.model_dim]

        if self.local_experts == 1:
            original_shape, x = x.shape, x.view(-1, self.model_dim)

            with torch.cuda.amp.autocast(enabled=False):
                x = torch.addmm(
                    fc1_bias.unsqueeze(0).float(),
                    x.float(),
                    fc1_weight.t().float(),
                )
            x = activation_fn(x.unsqueeze(0)).squeeze(0)
            x = self.dropout_fc1(x)
            if self.mlpfp32:
                with torch.cuda.amp.autocast(enabled=False):
                    if self.has_fc2_bias:
                        x = torch.addmm(
                            fc2_bias.unsqueeze(0).float(),
                            x.float(),
                            fc2_weight.float(),
                        )
                    else:
                        x = torch.matmul(x.float(), fc2_weight.float())
            else:
                if self.has_fc2_bias:
                    x = torch.addmm(
                        fc2_bias.unsqueeze(0), x, fc2_weight
                    )
                else:
                    x = torch.matmul(x, fc2_weight)
            x = self.dropout_fc2(x)
            x = x.view(original_shape)
        else:
            x = x.permute(1, 0, 2, 3)
            original_shape, x = x.shape, x.reshape(
                self.local_experts, -1, self.model_dim
            )
            with torch.cuda.amp.autocast(enabled=False):
                x = (
                    torch.matmul(
                        x.float(), fc1_weight.swapaxes(1, 2).float()
                    )
                    + fc1_bias.float()
                )
            x = activation_fn(x)
            x = self.dropout_fc1(x)

            if self.mlpfp32:
                with torch.cuda.amp.autocast(enabled=False):
                    x = torch.matmul(x.float(), fc2_weight.float())
                    if self.has_fc2_bias:
                        x = x + fc2_bias.float()
            else:
                x = torch.matmul(x, fc2_weight)
                if self.has_fc2_bias:
                    x = x + fc2_bias
            x = self.dropout_fc2(x)
            x = x.reshape(
                self.local_experts,
                original_shape[1],
                original_shape[2],
                self.model_dim,
            )
            x = x.permute(1, 0, 2, 3)
        return x

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.fc1_weight = self.fc1_weight.to(*args, **kwargs)
        self.fc2_weight = self.fc2_weight.to(*args, **kwargs)
        self.fc1_bias = self.fc1_bias.to(*args, **kwargs)
        if self.has_fc2_bias:
            self.fc2_bias = self.fc2_bias.to(*args, **kwargs)
        return self
# %%f
import torch

class recur_module(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x):
        if x == 0:
            return x
        return self.forward(x - 1)

mod = recur_module()

x = torch.Tensor([10])
y = mod(x)
print(y)
# %%

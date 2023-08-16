import torch
from torch import nn


class PumbaaNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = input + 1
        return output


pumbaaNN = PumbaaNN()
x = torch.tensor(1.0)
output = pumbaaNN(x)
print(output)

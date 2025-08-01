from pydantic import Field
from torch import nn

from serialization import SerializableModule


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(-1, *self.shape,)


class ToggleDropout2d(nn.Dropout2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enabled = True
    def forward(self, x):
        if self.enabled and self.p != 0:
            return super().forward(x)
        return x


class Mean(nn.Module):
    def __init__(self, dims=(2, 3)):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return x.mean(axis=self.dims)


class SEBlock(SerializableModule):
    in_channels: int = Field(..., description="Number of input channels")
    reduction_ratio: int = Field(16, description="Reduction ratio for bottleneck")
    inplace: bool = Field(False, description="Whether ReLU modifies input in-place")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden_size = max(self.in_channels // self.reduction_ratio, 1)
        self.fc = nn.Sequential(
            nn.Linear(self.in_channels, hidden_size),
            nn.ReLU(inplace=self.inplace),
            nn.Linear(hidden_size, self.in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Squeeze-and-Excitation scaling
        scale = self.fc(self.avg_pool(x).flatten(1)).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class ResConn(nn.Module):
    def __init__(self, inp_transform, residual=1):
        super().__init__()
        self.inp_transform = inp_transform
        self.residual = residual
        self.residual_transform = lambda result, inp: inp + result
        if callable(residual):
            self.residual = nn.Parameter(residual())
            self.residual_transform = lambda result, inp: result + self.residual * inp
        elif type(residual) in [float, int]:
            self.residual = nn.Parameter(torch.full((1,), float(residual))[0])
            self.residual_transform = lambda result, inp: result + self.residual * inp
    def forward(self, x):
        return self.residual_transform(self.inp_transform(x), x)

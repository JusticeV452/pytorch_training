import torch

from pydantic import Field
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm as torch_spectral_norm

from serialization import SerializableModule
from utils import next_largest_dividend


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, shape=None):
        super().__init__()
        self.shape = shape
    def forward(self, input):
        if self.shape is None:
            return input.view(input.shape[0], -1, 1, 1)
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
        self.inp_transform = (
            nn.Sequential(*inp_transform)
            if isinstance(inp_transform, list | tuple)
            else inp_transform
        )
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


def construct_classifier(
        in_channels, depth=2, reduction_factor=4,
        norm_layer=nn.BatchNorm1d, layer_transform=nn.ReLU, **kwargs
    ):
    use_out_bias = kwargs.get("use_out_bias", False)
    out_activ = kwargs.get("out_activ", nn.Sigmoid())
    spectral_norm = kwargs.get("spectral_norm", False)
    channel_div = kwargs.get("channel_div", 1)

    # Construct classifier
    classif_layers = []
    def linear_layer(in_dim, out_dim):
        layer = [nn.Linear(in_dim, out_dim)]
        if spectral_norm:
            layer[0] = torch_spectral_norm(layer[0])
        if norm_layer:
            layer.append(norm_layer(out_dim))
        layer.append(layer_transform())
        return layer

    for _ in range(depth):
        out_dim = next_largest_dividend(in_channels // reduction_factor, channel_div)
        if out_dim == 0:
            raise Exception(f"Too many layers or too high layer reduction: in_dim = {in_channels}")
        classif_layers.extend(linear_layer(in_channels, out_dim))
        in_channels = out_dim

    classif_layers.append(
        nn.Linear(out_dim, 1, bias=use_out_bias)
    )
    if out_activ is not None:
        classif_layers.append(out_activ)
    return nn.Sequential(*classif_layers)


def construct_conv_classifier(
        in_channels, depth=2, reduction_factor=4, norm_layer=nn.BatchNorm2d,
        layer_transform=nn.LeakyReLU, **kwargs
    ):
    kernel_size = kwargs.get("kernel_size", 1)
    channel_div = kwargs.get("channel_div", 1)
    spectral_norm = kwargs.get("spectral_norm", False)
    dim_reducer = kwargs.get("dim_reducer", Mean)
    out_activ = kwargs.get("out_activ", nn.Sigmoid())

    # Construct classifier
    classif_layers = []
    def layer(in_channels, out_channels):
        layer = [nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2
        )]
        if spectral_norm:
            layer[0] = torch_spectral_norm(layer[0])
        if norm_layer:
            layer.append(norm_layer(out_channels))
        layer.append(layer_transform())
        return layer

    for _ in range(depth - 1):
        out_channels = next_largest_dividend(in_channels // reduction_factor, channel_div)
        classif_layers.extend(layer(in_channels, out_channels))
        in_channels = out_channels

    classif_layers.extend([
        nn.Conv2d(
            in_channels, 1, kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2
        ),
        dim_reducer()
    ])
    if out_activ is not None:
        classif_layers.append(out_activ)
    return nn.Sequential(*classif_layers)

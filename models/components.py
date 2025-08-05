import torch
import torch.nn.functional as F

from typing import Optional, Union
from pydantic import Field
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential

from models.components import SEBlock
from models.convs.dynamic import FILTER_TYPES, DynamicConv2d
from serialization import SerializableModule, AutoLambda, Lambda
from utils import not_none


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


class ConvLayer(SerializableModule):
    in_channels: int = Field(..., description="Number of input channels")
    out_channels: int = Field(..., description="Number of output channels")
    mid_channels: Optional[int] = Field(
        None, description="Intermediate channel size (defaults to out_channels if None)"
    )
    kernel_size: int = Field(3, description="Kernel size for convolutions")
    num_convs: int = Field(2, description="Number of convolution layers")
    norm_layer: Optional[AutoLambda[nn.Module]] = Field(
        nn.BatchNorm2d, description="Normalization layer class"
    )
    activ_type: AutoLambda[nn.Module] = Field(
        Lambda(nn.ReLU, inplace=True),
        description="Activation function factory"
    )
    conv_type: AutoLambda[nn.Module] = Field(
        nn.Conv2d, description="Convolution layer class"
    )
    residual_cfg: Optional[Union[str, dict, float, int, AutoLambda]] = Field(
        None, description="Residual connection type or configuration"
    )
    use_bias: bool = Field(False, description="Whether to use bias in convolutions")
    use_checkpointing: bool = Field(False, description="Enable checkpointing for convolutions")
    filter_gen: Optional[Union[str, AutoLambda[nn.Module]]] = Field(
        None, description="Filter generator class used if dynamic convolution is enabled"
    )
    se_param: Optional[Union[int, AutoLambda[nn.Module]]] = Field(
        None, description="Int: acts as reduction_ratio for SEBlock, func: factory for se block"
    )

    def __init__(self, in_channels, out_channels, mid_channels=None, num_convs=2, filter_gen=None, **kwargs):
        if not mid_channels or num_convs:
            mid_channels = out_channels
        if filter_gen:
            filter_gen = FILTER_TYPES[filter_gen] if isinstance(filter_gen, str) else filter_gen
            kwargs["conv_type"] = Lambda(DynamicConv2d, filt_gen=filter_gen)
        super().__init__(
            in_channels=in_channels, out_channels=out_channels,
            mid_channels=mid_channels, num_convs=num_convs,
            filter_gen=filter_gen, **kwargs
        )

        residual_cfg = self.residual_cfg
        self.residual_transform = None
        self.residual_type = None
        if callable(residual_cfg):
            self.residual_type = "callable"
            self.residual = nn.Parameter(residual_cfg())
            self.residual_transform = lambda result, inp: result + self.residual * inp
        elif type(residual_cfg) is dict:
            self.residual_type = "conv"
            self.residual = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, **residual_cfg)
            self.residual_transform = lambda result, inp: result + self.residual(inp)
        elif type(residual_cfg) in [float, int]:
            self.residual_type = "constant"
            self.residual = nn.Parameter(torch.full((1,), float(residual_cfg))[0])
            self.residual_transform = lambda result, inp: result + self.residual * inp
        elif not_none(residual_cfg) and "pad" in residual_cfg:
            self.residual_type = residual_cfg
            self.residual = nn.Parameter(torch.ones(in_channels))
            self.residual_transform = self.residual_pad

        def sub_layer(in_channels, out_channels):
            layer = [self.conv_type(
                in_channels, out_channels, kernel_size=self.kernel_size,
                padding=(self.kernel_size - 1) // 2, bias=self.use_bias
            )]
            if self.norm_layer:
                layer.append(self.norm_layer(out_channels))
            layer.append(self.activ_type())
            return layer

        convs = sub_layer(in_channels, mid_channels)
        for i in range(num_convs - 2):
            convs.extend(sub_layer(mid_channels, mid_channels))
        if num_convs > 1:
            convs.extend(sub_layer(mid_channels, out_channels))
        self.convs = nn.Sequential(*convs,)
        self.se_block = None
        se_param = self.se_param
        if type(se_param) is int:
            self.se_block = SEBlock(out_channels, reduction_ratio=se_param)
        elif callable(se_param):
            self.se_block = se_param()

    def residual_pad(self, result, inp):
        res = self.residual
        if "sigmoid" in self.residual_type:
            res = nn.Sigmoid()(res)
        inp = res.unsqueeze(0).unsqueeze(2).unsqueeze(2) * inp
        num_dims = len(result.shape)
        for i in range(1, num_dims):
            if inp.shape[i] > result.shape[i]:
                indicies = torch.tensor(
                    range(0, result.shape[i]),
                    device=inp.device
                )
                inp = torch.index_select(inp, i, indicies)
            elif inp.shape[i] < result.shape[i]:
                diff = result.shape[i] - inp.shape[i]
                remaining = diff % 2
                padding = (num_dims - i - 1) * [0, 0] + [diff // 2, diff // 2 + remaining] + (i - 1) * [0, 0]
                inp = F.pad(inp, padding)
        return result + inp

    def forward(self, x):
        convs = (
            (lambda inp: checkpoint_sequential(self.convs, self.num_convs - 1, inp))
            if self.use_checkpointing and self.num_convs > 1 else self.convs
        )
        result = convs(x)
        if type(self.residual_transform) is not type(None):
            result = self.residual_transform(result, x)
        return result

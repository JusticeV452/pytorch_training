import torch

from typing import Any, List, Optional, Tuple, Type, Union
from pydantic import Field
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm as torch_spectral_norm

from serialization import SerializableModule, AutoLambda


class MultiKernelConv(SerializableModule):

    in_channels: int = Field(..., description="Number of input channels.")
    out_channels: int = Field(..., description="Number of output channels.")
    kernel_sizes: Union[int, Tuple[int, int], List[Tuple[int, int]]] = Field(
        ..., description="Single kernel size or list of kernel sizes."
    )
    stride: Union[int, Tuple[int, int]] = Field(1, description="Stride for convolution.")
    padding: Union[int, Tuple[int, int], str] = Field(
        "same", description="Padding mode: int, tuple, or 'same'."
    )
    dilation: Union[int, Tuple[int, int]] = Field(1, description="Dilation for convolution.")
    groups: int = Field(1, description="Number of blocked connections.")
    bias: bool = Field(True, description="If True, adds a learnable bias.")
    padding_mode: str = Field("zeros", description="Padding mode: 'zeros', 'reflect', etc.")
    conv_type: AutoLambda[Type[nn.Module]] = Field(nn.Conv2d, description="Convolution type (default nn.Conv2d).")
    spectral_norm: bool = Field(False, description="Apply spectral normalization to layers.")
    device: Optional[torch.device] = Field(None, description="Device for tensor allocation.")
    dtype: Optional[torch.dtype] = Field(None, description="Data type for tensors.")

    def __init__(self, in_channels, out_channels, kernel_sizes, **kwargs):
        if type(kernel_sizes) in [int, float]:
            kernel_sizes = (int(kernel_sizes), int(kernel_sizes))
        if type(kernel_sizes[0]) is int:
            kernel_sizes = [kernel_sizes]
        super().__init__(
            in_channels=in_channels, out_channels=out_channels,
            kernel_sizes=kernel_sizes, **kwargs
        )
        base_out_size = self.out_channels // len(kernel_sizes)
        conv_outs = [base_out_size + self.out_channels % len(kernel_sizes)] + [base_out_size] * (len(kernel_sizes) - 1)
        padding = self.padding
        self.convs = nn.ModuleList([
            (torch_spectral_norm if self.spectral_norm else lambda x: x)(self.conv_type(
                self.in_channels, conv_outs[i], kernel_size,
                padding=(kernel_size[0] // 2, kernel_size[1] // 2) if padding == "same" else padding,
                stride=self.stride, dilation=self.dilation,
                groups=self.groups, bias=self.bias, padding_mode=self.padding_mode
            ))
            for i, kernel_size in enumerate(kernel_sizes)
        ])
    def forward(self, x):
        return torch.cat([conv(x) for conv in self.convs], axis=1)

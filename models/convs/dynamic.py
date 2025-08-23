import torch
import torch.nn.functional as F

from typing import Optional
from pydantic import Field
from torch import nn

from models.components import SEBlock
from serialization import AutoLambda, SerializableModule
from serialization.core import PMAutoCaster
from utils import int_round, next_largest_dividend, parse_filter_size


class Size2D(PMAutoCaster):
    @classmethod
    def _PM_auto_cast(cls, v):
        assert isinstance(v, int | tuple | list), f"Cannot cast {v} to Size2d"
        return parse_filter_size(v)


def split_into_patches(x, P):
    """
    Splits a tensor of shape (N, C, H, W) into patches with different numbers of patches
    along the height and width axes, padding as necessary.

    Args:
        x (torch.Tensor): Input tensor of shape (N, C, H, W).
        P_H (int): Number of patches along the height axis.
        P_W (int): Number of patches along the width axis.

    Returns:
        torch.Tensor: Output tensor of shape (N, P_H * P_W, C, PY, PX).
    """
    *_, H, W = x.shape
    P_H, P_W = P if isinstance(P, tuple | list) else (P, P)

    # Calculate patch sizes (PY, PX)
    PY = (H + P_H - 1) // P_H  # Ceiling division for height
    PX = (W + P_W - 1) // P_W  # Ceiling division for width

    # Calculate necessary padding
    pad_h = (PY * P_H - H)
    pad_w = (PX * P_W - W)
    x_padded = x
    if pad_h > 0 or pad_w > 0:
        padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
        # Pad the input tensor
        x_padded = F.pad(x, padding)

    orig_device = x.device
    # orig_device != "cpu" and (x_padded.shape[-2] > 256 or x_padded.shape[-3] > 256)
    cpu_offload = False
    if cpu_offload:
        x = x.cpu()

    # Split the tensor into patches
    patches = x_padded.unfold(2, PY, PY).unfold(3, PX, PX)  # (N, C, P_H, P_W, PY, PX)

    # Reshape and permute to get (N, P_H * P_W, C, PY, PX)
    patches = patches.permute(0, 2, 3, 1, 4, 5)
    # Combine patch indices into one dimension (N, P_H * P_W, C, PY, PX)
    patches = patches.flatten(1, 2)
    if cpu_offload:
        patches = patches.to(device=orig_device)

    return patches


class PatchSplit(nn.Module):
    def __init__(self, grid_size, flatten_channels=True):
        super().__init__()
        self.grid_size = grid_size
        self.flatten_channels = flatten_channels

    def forward(self, x):
        patches = split_into_patches(x, self.grid_size)
        if self.flatten_channels:
            patches = patches.flatten(1, 2)
        return patches


class NoParamBilinear(nn.Module):
    def __init__(self, h, w, align_corners=False):
        super().__init__()
        self.align_corners = align_corners
        self.output_shape = (h, w)

    def forward(self, x):
        return F.interpolate(
            x, size=self.output_shape,
            mode="bilinear", align_corners=self.align_corners
        )


class SpatialFC(nn.Module):
    def __init__(self, h, w):
        super().__init__()
        self.fc = nn.Linear(h * w, h * w)

    def forward(self, x):
        *_, H, W = x.shape
        return self.fc(x.view(*_, H * W)).view(*_, H, W)


class FilterGen(SerializableModule):
    in_channels: int = Field(..., description="Number of input channels")
    out_channels: int = Field(..., description="Number of output channels")
    kernel_size: Size2D = Field(..., description="Kernel size for dynamic filters")

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        kernel_size = parse_filter_size(kernel_size)
        super().__init__(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, **kwargs
        )
        self.fy, self.fx = kernel_size

    def weight_gen(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.weight_gen(x).view(-1, x.shape[1], self.fy, self.fx)


class FCFilterGen(FilterGen):
    input_filter_dim: int = Field(4, description="Intermediate embedding size in filter generation")
    num_pc: int = Field(0, description="Number of pre-conv layers before filter generation")
    pc_start: int = Field(32, description="Initial output channels for pre-conv layers")
    pc_kernel_size: int = Field(3, description="Kernel size for pre-conv layers")
    pc_growth_rate: float = Field(2., description="Multiplicative growth rate for pre-conv channels")
    pc_activ: AutoLambda[nn.Module] = Field(nn.ReLU, description="Activation module for pre-conv layers")
    pc_norm: AutoLambda[nn.Module] = Field(nn.BatchNorm2d, description="Normalization module for pre-conv layers")

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, **kwargs
        )
        filter_gen_layers = []
        # Add pre convs (pc)
        pc_in_channels = in_channels
        for _ in range(self.num_pc):
            out_channels = int_round(self.pc_growth_rate * pc_in_channels)
            filter_gen_layers.extend([
                nn.Conv2d(
                    pc_in_channels, self.pc_start, kernel_size=self.pc_kernel_size,
                    stride=2, padding=kernel_size // 2
                ),
                self.pc_norm(out_channels),
                self.pc_activ()
            ])
            pc_in_channels = out_channels
        filter_gen_layers += [
            nn.AdaptiveAvgPool2d(1),  # Global pooling for input summary
            nn.Flatten(),
            nn.Linear(pc_in_channels, self.input_filter_dim),
            nn.ReLU(),
            nn.Linear(
                self.input_filter_dim,
                out_channels * in_channels * self.kernel_size[0] * self.kernel_size[1]
            )
        ]
        self.backbone = nn.Sequential(*filter_gen_layers)
    
    def weight_gen(self, x):
        return self.backbone(x)


class MatmulConvFilterGen(FilterGen):
    filt_kernel_size: Size2D = Field(
        3, description="Kernel size used in internal filter generation convolutions"
    )
    zero_pad: bool = Field(
        False, description="If True, uses padding to maintain resolution in internal convolutions"
    )
    use_in_conv: bool = Field(True, description="Whether to apply a pre-filter convolution on the input")

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, **kwargs
        )
        filt_kernel_size = self.filt_kernel_size
        zero_pad = self.zero_pad
        self.in_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=filt_kernel_size,
            padding=tuple(filt_kernel_size[i] // 2 for i in range(2)) if zero_pad else 0,
        ) if self.use_in_conv else lambda x: x
        self.out_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=filt_kernel_size,
            padding=tuple(filt_kernel_size[i] // 2 for i in range(2)) if zero_pad else 0,
        )
        self.downsample_pad = [
            0 if zero_pad else filt_kernel_size[i] - 1 for i in range(2)
        ]

    def weight_gen(self, x):
        x = F.interpolate(
            x, size=(
                self.fy +
                self.downsample_pad[0], self.fx + self.downsample_pad[1]
            ),
            mode="bilinear", align_corners=False
        )
        in_conv_inp = x
        if not self.use_in_conv and any(self.downsample_pad):
            # Need to make in_conv input expected size if not using convolution
            in_conv_inp = F.interpolate(
                x, size=(self.fy, self.fx),
                mode="bilinear", align_corners=False
            )
        N, C, *_, = x.shape
        in_conv_result = self.in_conv(in_conv_inp)
        return torch.matmul(
            in_conv_result.view(N, self.fy, self.fx, C, 1),
            self.out_conv(x).view(N, self.fy, self.fx, 1, self.out_channels)
        )


class BottleneckConvFilterGen(FilterGen):
    filt_kernel_size: Size2D = Field(3, description="Internal filter kernel size")
    bneck_div: int = Field(16, description="Bottleneck channel division factor")
    zero_pad: bool = Field(False, description="Whether to use zero padding")
    se_div: int = Field(4, description="Squeeze-excitation reduction ratio (0 disables SE block)")
    bneck_groups: int = Field(1, description="Groups for bottleneck input conv")
    inv_bneck_groups: int = Field(1, description="Groups for bottleneck output conv")
    bneck_channel_div: int = Field(1, description="Ensure bottleneck channels divisible by this")
    bneck_activ: Optional[AutoLambda[nn.Module]] = Field(None, description="Activation after bottleneck input conv")
    activ: Optional[AutoLambda[nn.Module]] = Field(None, description="Activation after bottleneck mid conv")
    hidden_fc: bool = Field(False, description="Whether to apply spatial fully-connected layer")

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, **kwargs
        )
        filt_kernel_size = self.filt_kernel_size
        bneck_channels = next_largest_dividend(
            max(in_channels // self.bneck_div, 1), self.bneck_channel_div
        )
        backbone = [nn.Conv2d(
            in_channels, bneck_channels,
            kernel_size=1, groups=self.bneck_groups
        )]
        if self.bneck_activ:
            backbone.append(self.bneck_activ())
        backbone.append(nn.Conv2d(
            bneck_channels, bneck_channels,
            kernel_size=filt_kernel_size,
            padding=tuple(filt_kernel_size[i] // 2 for i in range(2)) if self.zero_pad else 0,
        ))
        if self.hidden_fc:
            backbone.append(SpatialFC(self.fy, self.fx))
        if self.activ:
            backbone.append(self.activ())
        if self.se_div:
            backbone.append(SEBlock(bneck_channels, self.se_div))
        backbone.append(nn.Conv2d(
            bneck_channels, in_channels * out_channels, kernel_size=1, groups=self.inv_bneck_groups
        ))
        self.backbone = nn.Sequential(*backbone)
        self.downsample_pad = [
            0 if self.zero_pad else filt_kernel_size[i] - 1 for i in range(2)
        ]

    def weight_gen(self, x):
        return self.backbone(F.interpolate(
            x, size=(
                self.fy + self.downsample_pad[0], self.fx + self.downsample_pad[1]),
            mode="bilinear", align_corners=False
        ))


class PatchbasedBottleneckConvFilterGen(BottleneckConvFilterGen):
    grid_size: Size2D = Field(4, description="Grid size for patch splitting")
    reduction_channels: Optional[int] = Field(None, description="Optional channel reduction before patch split")
    channel_reduce_kernel_size: int = Field(1, description="Kernel size for optional channel reduction")
    channel_reduce_groups: int = Field(1, description="Groups for optional channel reduction")

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        FilterGen.__init__(
            self, in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, **kwargs
        )
        grid_size = self.grid_size
        filt_kernel_size = self.filt_kernel_size
        reduction_channels = self.reduction_channels

        bneck_channels = next_largest_dividend(
            max(in_channels // self.bneck_div, 1), self.bneck_channel_div
        )
        self.downsample_pad = [0 if self.zero_pad else filt_kernel_size[i] - 1 for i in range(2)]

        backbone = []
        if reduction_channels:
            backbone.append(nn.Conv2d(
                in_channels, reduction_channels,
                kernel_size=self.channel_reduce_kernel_size,
                groups=self.channel_reduce_groups
            ))
        backbone.append(PatchSplit(grid_size=grid_size))
        bneck_in_channels = grid_size[0] * grid_size[1] * (reduction_channels if reduction_channels else in_channels)
        backbone.extend([
            nn.Conv2d(bneck_in_channels, bneck_channels, kernel_size=1, groups=self.bneck_groups),
            NoParamBilinear(self.fy + self.downsample_pad[0], self.fx + self.downsample_pad[1])
        ])
        if self.bneck_activ:
            backbone.append(self.bneck_activ())
        backbone.append(nn.Conv2d(
            bneck_channels, bneck_channels,
            kernel_size=filt_kernel_size,
            padding=tuple(filt_kernel_size[i] // 2 for i in range(2)) if self.zero_pad else 0,
        ))
        if self.hidden_fc:
            backbone.append(SpatialFC(self.fy, self.fx))
        if self.activ:
            backbone.append(self.activ())
        if self.se_div:
            backbone.append(SEBlock(bneck_channels, reduction_ratio=self.se_div))
        backbone.append(nn.Conv2d(
            bneck_channels, in_channels * out_channels,
            kernel_size=1, groups=self.inv_bneck_groups
        ))
        self.backbone = nn.Sequential(*backbone)

    def weight_gen(self, x):
        return self.backbone(x)


class DynamicConv2d(SerializableModule):
    in_channels: int = Field(..., description="Number of input channels")
    out_channels: int = Field(..., description="Number of output channels")
    kernel_size: Size2D = Field(..., description="Size of the convolution kernel")
    stride: Size2D = Field(1, description="Stride of the convolution")
    padding: Size2D = Field(0, description="Padding added to both sides of input")
    dilation: Size2D = Field(1, description="Spacing between kernel elements")
    groups: int = Field(1, description="Number of groups for convolution")
    bias: bool = Field(False, description="Whether to include bias")
    filter_gen: AutoLambda[nn.Module] = Field(FCFilterGen, description="Filter generation network class")
    filter_gen_kwargs: dict = Field(default_factory=dict, description="Keyword args for filter generator")

    # ChatGPT + https://discuss.pytorch.org/t/how-to-apply-different-kernels-to-each-example-in-a-batch-when-using-convolution/84848/4
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        """
        A dynamic 2D convolutional layer that generates filters based on the input.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolution kernel.
            input_filter_dim (int): Dimensionality of intermediate embedding for filter generation.
            stride (int or tuple): Stride of the convolution.
            padding (int or tuple): Zero-padding added to both sides of the input.
            dilation (int or tuple): Spacing between kernel elements.
        """
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

        # Filter generation network
        self.filter_generator = self.filter_gen(
            self.in_channels, self.out_channels, self.kernel_size,
            **self.filter_gen_kwargs
        )

    def forward(self, x):
        N, _, H, W = x.shape
        # Generate filters based on input
        filters = self.filter_generator(x)
        # Apply convolution for each batch
        x = x.reshape(1, -1, H, W)
        # Apply grouped conv
        outputs_grouped = F.conv2d(
            x, filters, stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=N
        )
        in_dims = [H, W]
        out_dims = [
            (d + 2 * self.padding[i] - self.dilation[i] * (self.kernel_size[i] - 1) - 1) // self.stride[i] + 1
            for i, d in enumerate(in_dims)
        ]
        return outputs_grouped.view(N, self.out_channels, *out_dims)


FILTER_TYPES = {
    "fc": FCFilterGen,
    "matmul": MatmulConvFilterGen,
    "bottleneck": BottleneckConvFilterGen,
    "pb_bottleneck": PatchbasedBottleneckConvFilterGen
}

import torch.nn.functional as F

from pydantic import Field
from torch import nn

from serialization import SerializableModule, AutoLambda
from utils import int_round, parse_filter_size


class FilterGen(SerializableModule):
    in_channels: int = Field(..., description="Number of input channels")
    out_channels: int = Field(..., description="Number of output channels")
    kernel_size: int | tuple[int, int] = Field(..., description="Kernel size for dynamic filters")

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


class DynamicConv2d(nn.Module):
    # ChatGPT + https://discuss.pytorch.org/t/how-to-apply-different-kernels-to-each-example-in-a-batch-when-using-convolution/84848/4
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
            groups=1, bias=False, filter_gen=FCFilterGen, filter_gen_kwargs=None):
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
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = parse_filter_size(kernel_size)
        self.stride = parse_filter_size(stride)
        self.padding = parse_filter_size(padding)
        self.dilation = parse_filter_size(dilation)
        filter_gen_kwargs = filter_gen_kwargs if filter_gen_kwargs else {}

        # Filter generation network
        self.filter_generator = filter_gen(in_channels, out_channels, kernel_size, **filter_gen_kwargs)

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
}

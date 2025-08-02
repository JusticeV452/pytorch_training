import torch.nn.functional as F

from torch import nn

from utils import int_round, parse_filter_size


class DynamicConv2d(nn.Module):
    # ChatGPT + https://discuss.pytorch.org/t/how-to-apply-different-kernels-to-each-example-in-a-batch-when-using-convolution/84848/4
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, input_filter_dim=4,
            num_pc=0, pc_start=32, pc_kernel_size=3, pc_growth_rate=2, pc_activ=nn.ReLU, pc_norm=nn.BatchNorm2d, weight_norm=False):
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
        super(DynamicConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = parse_filter_size(kernel_size)
        self.stride = parse_filter_size(stride)
        self.padding = parse_filter_size(padding)
        self.dilation = parse_filter_size(dilation)
        self.weight_norm = weight_norm

        # Filter generation network
        filter_gen_layers = []
        # Add pre convs (pc)
        pc_in_channels = in_channels
        for _ in range(num_pc):
            out_channels = int_round(pc_growth_rate * pc_in_channels)
            filter_gen_layers.extend([
                nn.Conv2d(
                    pc_in_channels, pc_start, kernel_size=pc_kernel_size,
                    stride=2, padding=kernel_size // 2
                ),
                pc_norm(out_channels),
                pc_activ()
            ])
            pc_in_channels = out_channels
        filter_gen_layers += [
            nn.AdaptiveAvgPool2d(1),  # Global pooling for input summary
            nn.Flatten(),
            nn.Linear(pc_in_channels, input_filter_dim),
            nn.ReLU(),
            nn.Linear(
                input_filter_dim, out_channels * in_channels * self.kernel_size[0] * self.kernel_size[1])
        ]
        self.filter_generator = nn.Sequential(*filter_gen_layers)

    def forward(self, x):
        batch_size, _, height, width = x.size()

        # Generate filters based on input
        filters = self.filter_generator(x).view(
            -1, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        )
        # if self.weight_norm:
        #     weight_mean = filters.mean(dim=(1, 2, 3), keepdim=True)
        #     weight_std = filters.std(dim=(1, 2, 3), keepdim=True)
        #     filters = (filters - weight_mean) / (weight_std + 1e-5)

        # Apply convolution for each batch
        x = x.reshape(1, -1, height, width)  # move batch dim into channels
        outputs_grouped = F.conv2d(  # Apply grouped conv
            x, filters, stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=batch_size
        )
        hout = (height + 2 * self.padding[0] - self.dilation[0]
                * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        wout = (width + 2 * self.padding[1] - self.dilation[1]
                * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        return outputs_grouped.view(batch_size, self.out_channels, hout, wout)


FILTER_TYPES = {}

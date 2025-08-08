import torch
import torch.nn as nn
import torch.nn.functional as F

from pydantic import Field
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from typing import Optional, Tuple, Union

from models.components import Flatten, ToggleDropout2d
from models.conv_layer import ConvLayer
from models.convs.dynamic import FILTER_TYPES, DynamicConv2d
from serialization import AutoLambda, Lambda, SerializableModel, SerializableModule
from utils import is_none, next_largest_dividend, rec_int_pow


class ConvPixelShuffleUpscale(SerializableModule):
    in_channels: int = Field(..., description="Input channels")
    out_channels: int = Field(..., description="Output channels")
    scale_factor: int = Field(2, description="Pixel shuffle scale factor")
    groups: int = Field(-1, description="Number of groups for GroupNorm (-1 means in_channels)")
    kernel_size: int = Field(1, description="Kernel size for pre_conv")
    out_kernel_size: Optional[int] = Field(None, description="Kernel size for post_conv")
    pre_conv_bias: bool = Field(True, description="Enable bias for pre convolutions")
    post_conv_bias: bool = Field(True, description="Enable bias for pre convolutions")

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.padding = (self.kernel_size - 1) // 2
        if self.out_kernel_size is None:
            self.out_kernel_size = self.kernel_size
        in_channels = self.in_channels
        pre_conv_out = in_channels * (self.scale_factor ** 2)
        pre_conv = nn.Conv2d(
            in_channels, pre_conv_out, kernel_size=self.kernel_size,
            padding=self.padding, bias=self.pre_conv_bias
        )
        groups = in_channels if groups == -1 else groups
        self.pre_conv = pre_conv if groups is None else nn.Sequential(
            pre_conv,
            nn.GroupNorm(groups, pre_conv_out)
        )
        self.pixel_shuffle = nn.PixelShuffle(self.scale_factor)
        self.post_conv = nn.Conv2d(
            in_channels, self.out_channels, kernel_size=self.out_kernel_size,
            padding=(self.out_kernel_size - 1) // 2, bias=self.post_conv_bias
        )

    def forward(self, x):
        x = self.pre_conv(x)  # Expand channels
        x = self.pixel_shuffle(x)  # Rearrange to higher spatial resolution
        return self.post_conv(x)
    
    def _initialize_weights(self):
        for m in self.modules():
            if not isinstance(m, nn.Conv2d):
                continue
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is None:
                continue
            nn.init.constant_(m.bias, 0)


class FlexDown(SerializableModule):
    in_channels: int = Field(..., description="Number of input channels")
    out_channels: int = Field(..., description="Number of output channels")
    kernel_size: int = Field(3, description="Kernel size for ConvLayer")
    conv_layer_size: int = Field(2, description="Number of convolutions per ConvLayer")
    num_conv_layers: int = Field(1, description="Number of ConvLayer blocks after downscaling")
    norm_layer: Optional[AutoLambda[nn.Module]] = Field(
        nn.BatchNorm2d, description="Normalization layer type (default: BatchNorm2d)"
    )
    activ_type: AutoLambda[nn.Module] = Field(
        Lambda(nn.ReLU, inplace=True), description="Activation function factory"
    )
    downscaler: AutoLambda[Tuple[int], nn.Module] = Field(
        Lambda(nn.MaxPool2d, kernel_size=2, ignore_args_=True),
        description="Downscaling layer (default: MaxPool2d)"
    )
    conv_se_param: Optional[Union[int, AutoLambda[nn.Module]]] = Field(
        None, description="Squeeze-and-Excitation parameters for ConvLayer"
    )
    conv_type: AutoLambda[nn.Module] = Field(
        nn.Conv2d, description="Convolution layer type (default: Conv2d)"
    )
    conv_residual: Optional[Union[str, dict, float, int, AutoLambda]] = Field(
        None, description="Residual connection configuration for ConvLayer"
    )
    use_checkpointing: bool = Field(
        False, description="Enable checkpointing to reduce memory usage"
    )

    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, **kwargs)
        out_channels = self.out_channels
        def get_conv_layer(in_channels, out_channels):
            return ConvLayer(
                in_channels, out_channels, kernel_size=self.kernel_size,
                num_convs=self.conv_layer_size, norm_layer=self.norm_layer,
                activ_type=self.activ_type, conv_type=self.conv_type,
                residual_cfg=self.conv_residual, se_param=self.conv_se_param,
                use_checkpointing=self.use_checkpointing
            )
        conv_layers = [
            get_conv_layer(out_channels, out_channels)
            for _ in range(self.num_conv_layers - 1)
        ]
        self.use_checkpointing = self.use_checkpointing
        self.maxpool_conv = nn.Sequential(
            self.downscaler(self.in_channels), 
            get_conv_layer(self.in_channels, out_channels),
            *conv_layers,
        )

    def forward(self, x):
        if self.use_checkpointing and len(self.maxpool_conv) > 2:
            return checkpoint_sequential(self.maxpool_conv, len(self.maxpool_conv) - 2, x)
        return self.maxpool_conv(x)


class FlexUp(SerializableModule):
    """Upscaling then double conv"""

    in_channels: int = Field(..., description="Number of input channels")
    out_channels: int = Field(..., description="Number of output channels")
    kernel_size: int = Field(3, description="Kernel size for ConvLayer")
    conv_layer_size: int = Field(2, description="Number of convolutions per ConvLayer")
    num_conv_layers: int = Field(1, description="Number of ConvLayer blocks after upscaling")
    bilinear: bool = Field(True, description="Use bilinear upsampling instead of ConvTranspose2d")
    norm_layer: Optional[AutoLambda[nn.Module]] = Field(
        nn.BatchNorm2d, description="Normalization layer type (default: BatchNorm2d)"
    )
    activ_type: AutoLambda[nn.Module] = Field(
        Lambda(nn.ReLU, inplace=True), description="Activation function factory"
    )
    conv_type: AutoLambda[nn.Module] = Field(
        nn.Conv2d, description="Convolution layer type (default: Conv2d)"
    )
    use_checkpointing: bool = Field(
        False, description="Enable checkpointing to reduce memory usage"
    )
    upsample_mode: Optional[str] = Field(
        '', description="Mode for upsampling ('' for default, 'cpsu-*' for pixel shuffle)"
    )
    conv_residual: Optional[Union[str, dict, float, int, AutoLambda]] = Field(
        None, description="Residual connection configuration for ConvLayer"
    )
    conv_se_param: Optional[Union[int, AutoLambda[nn.Module]]] = Field(
        None, description="Squeeze-and-Excitation parameters for ConvLayer"
    )

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, **kwargs)
        in_channels = self.in_channels
        out_channels = self.out_channels
        upsample_mode = self.upsample_mode or ""
        if self.bilinear:
            out_channels //= 2
        def get_conv_layer(in_channels, out_channels, mid_channels=None):
            return ConvLayer(
                in_channels, out_channels, mid_channels=mid_channels, kernel_size=self.kernel_size,
                num_convs=self.conv_layer_size, norm_layer=self.norm_layer,
                activ_type=self.activ_type, conv_type=self.conv_type,
                residual_cfg=self.conv_residual, se_param=self.conv_se_param,
                use_checkpointing=self.use_checkpointing
            )

        # if bilinear, use the normal convolutions to reduce the number of channels
        mode = upsample_mode if upsample_mode else "bilinear"
        use_bilinear = self.bilinear or upsample_mode
        conv_mid_channels = (in_channels // 2) if use_bilinear and "cpsu" not in upsample_mode else None
        
        def parse_kernel_sizes(upsample_mode):
            return {
                key: int(val)
                for key, val in zip(["kernel_size", "out_kernel_size"], upsample_mode.split('-')[1:])
            }

        self.up = (
            (
                ConvPixelShuffleUpscale(in_channels, out_channels, **parse_kernel_sizes(upsample_mode))
                if "cpsu" in upsample_mode
                else nn.Upsample(scale_factor=2, mode=mode, align_corners=True)
            )
            if use_bilinear
            else nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
        )
        
        self.conv = nn.Sequential(
            get_conv_layer(in_channels, out_channels, mid_channels=conv_mid_channels),
            *[
                get_conv_layer(out_channels, out_channels)
                for i in range(self.num_conv_layers - 1)
            ],
        )

    def forward(self, x1, x2, glue_conv=None):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        if callable(glue_conv):
            x = glue_conv(x)
        checkpoints = len(self.conv)
        use_checkpointing = self.use_checkpointing and checkpoints > 0
        conv = (
            (lambda inp: checkpoint_sequential(self.conv, checkpoints, inp))
            if use_checkpointing else self.conv
        )
        return conv(x)


class OutConv(SerializableModule):
    in_channels: int = Field(..., description="Number of input channels")
    out_channels: int = Field(..., description="Number of output channels")
    kernel_size: int = Field(1, description="Kernel size for final convolution")
    depth: int = Field(0, description="Number of pre-output ConvLayers")
    norm_layer: Optional[AutoLambda[nn.Module]] = Field(
        nn.BatchNorm2d, description="Normalization layer type (default: BatchNorm2d)"
    )
    activ_type: AutoLambda[nn.Module] = Field(
        Lambda(nn.ReLU, inplace=True), description="Activation function factory"
    )
    conv_type: AutoLambda[nn.Module] = Field(
        nn.Conv2d, description="Convolution layer type (default: Conv2d)"
    )
    use_checkpointing: bool = Field(
        False, description="Enable checkpointing for pre-output layers"
    )

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, **kwargs)
        in_channels = self.in_channels
        kernel_size = self.kernel_size
        self.pre_out = nn.Sequential(*[
            ConvLayer(
                in_channels, in_channels, conv_type=self.conv_type,
                kernel_size=kernel_size, norm_layer=self.norm_layer,
                activ_type=self.activ_type,
                use_checkpointing=self.use_checkpointing
            ) for _ in range(self.depth)
        ],)
        self.conv = nn.Conv2d(
            in_channels, self.out_channels, kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2
        )

    def forward(self, x):
        checkpoints = self.depth - 1
        use_checkpointing = self.use_checkpointing and checkpoints > 0
        pre_out = (
            (lambda inp: checkpoint_sequential(self.pre_out, checkpoints, inp))
            if use_checkpointing else self.pre_out
        )
        return self.conv(pre_out(x))


class UNetDown(SerializableModel):
    num_channels: int = Field(..., description="Number of input channels")
    conv_start: int = Field(64, description="Initial number of convolution channels")
    kernel_size: int = Field(3, description="Kernel size for convolutions")
    conv_layer_size: int = Field(2, description="Number of convolutions in the input layer")
    num_layers: int = Field(4, description="Number of downsampling layers")
    conv_block_layers: int = Field(1, description="Number of convolutions per downsampling block")
    norm_layer: AutoLambda[nn.Module] = Field(nn.BatchNorm2d, description="Normalization layer type")
    activation: AutoLambda[nn.Module] = Field(Lambda(nn.ReLU, inplace=True), description="Activation function")
    bilinear: bool = Field(True, description="Whether bilinear upsampling is used")
    conv_type: AutoLambda[nn.Module] = Field(nn.Conv2d, description="Base convolution layer type")
    growth_rate: float = Field(2.0, description="Channel growth rate per layer")
    use_checkpointing: bool = Field(False, description="Enable activation checkpointing")
    layer_sizes: Optional[list[int]] = Field(None, description="Predefined layer sizes")
    conv_residual: Optional[dict] = Field(None, description="Residual block configuration for conv layers")
    channel_div: int = Field(1, description="Ensure channels divisible by this value")
    filter_gen: Optional[Union[str, AutoLambda[nn.Module]]] = Field(None, description="Optional dynamic filter generator")
    conv_se_param: Optional[Union[int, AutoLambda[nn.Module]]] = Field(None, description="SE block reduction ratio")
    downscaler: AutoLambda[nn.Module] = Field(
        Lambda(nn.MaxPool2d, kernel_size=2, ignore_args_=True),
        description="Layer used to downsample feature maps"
    )

    def __init__(self, num_channels, **kwargs):
        if filter_gen := kwargs.get("filter_gen"):
            kwargs["filter_gen"] = FILTER_TYPES[filter_gen] if isinstance(filter_gen, str) else filter_gen
            kwargs["conv_type"] = Lambda(DynamicConv2d, filter_gen=filter_gen)
        super().__init__(num_channels=num_channels, **kwargs)
        kernel_sizes = self.kernel_size
        num_layers = self.num_layers
        conv_start = self.conv_start
        activation = self.activation
        conv_residual = self.conv_residual
        use_checkpointing = self.use_checkpointing
        conv_type = self.conv_type
        conv_se_param = self.conv_se_param
        norm_layer = self.norm_layer
        conv_layer_size = self.conv_layer_size

        if type(self.kernel_size) is int:
            self.kernel_sizes = kernel_sizes = [self.kernel_size for _ in range(num_layers + 1)]
        if self.layer_sizes is None and conv_start:
            self.layer_sizes = layer_sizes = [
                next_largest_dividend(rec_int_pow(conv_start, self.growth_rate, i), self.channel_div)
                for i in range(num_layers + 1)
            ]

        self.input_layer = ConvLayer(
            num_channels, layer_sizes[0], kernel_size=kernel_sizes[0], num_convs=conv_layer_size,
            norm_layer=norm_layer, activ_type=activation, conv_type=conv_type,
            residual_cfg=conv_residual, se_param=conv_se_param, use_checkpointing=use_checkpointing
        )
        self.downs = nn.ModuleList()
        factor = 2 if self.bilinear else 1
        self.out_channel_sizes = layer_sizes
        for l in range(len(layer_sizes) - 1):
            in_channels = layer_sizes[l]
            out_channels = layer_sizes[l + 1]
            if l >= num_layers - 1:
                out_channels //= factor
            self.downs.append(FlexDown(
                in_channels, out_channels, kernel_size=kernel_sizes[l + 1],
                conv_layer_size=conv_layer_size, num_conv_layers=self.conv_block_layers,
                norm_layer=norm_layer, activ_type=activation, conv_type=conv_type,
                conv_residual=conv_residual, use_checkpointing=use_checkpointing,
                downscaler=self.downscaler, conv_se_param=conv_se_param
            ))

    def forward(self, x):
        out = [self.input_layer(x)]
        for down_layer in self.downs:
            last = out[-1]
            out.append(self.layer_forward(
                down_layer, last,
                use_checkpoint=self.use_checkpointing
            ))
        return out


class MultiInpFlexUNet(SerializableModel):
    num_channels: int = Field(..., description="Number of input channels")
    num_classes: Optional[int] = Field(num_channels, description="Number of output channels (defaults to input channels)")
    input_allocs: Optional[list[int]] = Field(None, description="Starting input channels for each UNet branch")
    channel_allocs: Optional[list[int]] = Field(None, description="Channels per branch")
    allow_glue: bool = Field(False, description="Allow glue convs to align shapes between down and up paths")
    norm_layer: AutoLambda[nn.Module] = Field(nn.BatchNorm2d, description="Normalization layer to use in convs")
    use_checkpoints: bool = Field(False, description="Use PyTorch checkpointing")
    channel_div: int = Field(1, description="Ensure channel sizes divisible by this number")
    unet_down_module: AutoLambda[nn.Module] = Field(UNetDown, description="Down path block type")
    bilinear: bool = Field(True, description="Whether to use bilinear upsampling")
    out_activ: AutoLambda[nn.Module] = Field(
        Lambda(nn.Sigmoid, parent_kwargs_={}), description="Activation function applied to final output"
    )

    # Exclude
    kernel_size: int = Field(3, exclude=True, description="Kernel size for all convolutions")
    conv_block_layers: int = Field(1, exclude=True, description="Conv layers per stage")
    layer_size: int = Field(2, exclude=True, description="Number of convs per conv layer in down block")
    activ_type: AutoLambda[nn.Module] = Field(
        Lambda(nn.ReLU, inplace=True), exclude=True, description="Default activation function"
    )

    start: int = Field(64, description="Number of base channels in first UNet layer")
    num_layers: int = Field(4, description="Number of UNet downsample/upsample layers")
    down_kernel_size: int = Field(kernel_size, description="Kernel size override for down path convolutions")
    up_kernel_size: int = Field(kernel_size, description="Kernel size override for up path convolutions")
    glue_kernel_size: int = Field(1, description="Kernel size for glue convolutions")
    out_kernel_size: int = Field(1, description="Kernel size of final output convolution")
    out_depth: int = Field(0, description="Number of output layers in final output block")
    down_block_layers: int = Field(conv_block_layers, description="Conv layers per stage in down path")
    up_block_layers: int = Field(conv_block_layers, description="Conv layers per stage in up path")
    down_layer_size: int = Field(layer_size, description="Number of convs per conv layer in down block")
    up_layer_size: int = Field(layer_size, description="Number of convs per conv layer in up block")
    down_activ: Optional[AutoLambda[nn.Module]] = Field(activ_type, description="Optional override for down path activation")
    up_activ: Optional[AutoLambda[nn.Module]] = Field(activ_type, description="Optional override for up path activation")
    growth_rate: float = Field(2.0, description="Channel width multiplier between layers")
    residual: Optional[bool] = Field(None, description="Whether to use residual connections")
    conv_se_param: Optional[Union[int, AutoLambda[nn.Module]]] = Field(None, description="SE layer reduction ratio (if used)")
    post_glue_activ: Optional[AutoLambda[nn.Module]] = Field(None, description="Activation after glue layer")
    post_glue_norm: Optional[AutoLambda[nn.Module]] = Field(None, description="Normalization after glue layer")
    upsample_mode: Union[str, list[str | None]] = Field(
        '', description="Mode for upsampling ('' for default, 'cpsu-*' for pixel shuffle)"
    )
    conv_type: AutoLambda[nn.Module] = Field(
        nn.Conv2d, description="Convolution layer type (default: Conv2d)"
    )
    filter_gen: Optional[Union[str, AutoLambda[nn.Module]]] = Field(
        None, description="Optional filter generator for dynamic convs"
    )

    def __init__(self, num_channels, **kwargs):
        if filter_gen := kwargs.get("filter_gen"):
            kwargs["filter_gen"] = FILTER_TYPES[filter_gen] if isinstance(filter_gen, str) else filter_gen
            kwargs["conv_type"] = Lambda(DynamicConv2d, filter_gen=filter_gen)

        if not isinstance((upsample_mode := kwargs.get("upsample_mode")), list | tuple):
            kwargs["upsample_mode"] = upsample_mode = [upsample_mode]

        super().__init__(num_channels=num_channels, **kwargs)
        layer_start = self.start
        if self.input_allocs is None:
            self.input_allocs = [
                int(round(layer_start * self.num_classes / self.num_channels))
                for _ in range(self.num_channels // self.num_classes)
            ]
        input_allocs = self.input_allocs
        if self.channel_allocs is None:
            self.channel_allocs = [num_channels // len(input_allocs) for _ in range(len(input_allocs))]

        growth_rate = self.growth_rate
        num_layers = self.num_layers
        glue_kernel_size = self.glue_kernel_size
        post_glue_norm = self.post_glue_norm
        post_glue_activ = self.post_glue_activ
        norm_layer = self.norm_layer
        use_checkpoints = self.use_checkpoints
        conv_type = self.conv_type
        channel_div = self.channel_div
        conv_residual = self.residual
        conv_se_param = self.conv_se_param

        self.downs = nn.ModuleList([
            self.unet_down_module(
                channel_alloc, conv_start=alloc, kernel_size=self.down_kernel_size, conv_layer_size=self.down_layer_size,
                num_layers=self.num_layers, conv_block_layers=self.down_block_layers, norm_layer=norm_layer,
                activation=self.down_activ, bilinear=self.bilinear, conv_type=conv_type, growth_rate=growth_rate,
                use_checkpointing=use_checkpoints, channel_div=channel_div, conv_residual=conv_residual,
                conv_se_param=conv_se_param, device=self.device
            ) for channel_alloc, alloc in zip(self.channel_allocs, input_allocs)
        ])

        # Use convs with kernel size = 1 to reshape down_conv outputs concatenated with up conv outputs to expected size
        self.glue_convs = nn.ModuleList()
        self.base_glue = None
        down_channels_outs = [sum([down.out_channel_sizes[l] for down in self.downs]) for l in range(num_layers + 1)]
        expected_up_outs = [
            next_largest_dividend(rec_int_pow(layer_start, growth_rate, num_layers - l), channel_div)
            for l in range(num_layers + 1)
        ]

        def expand_glue_conv(glue_conv, enc_out):
            layers = [glue_conv]
            if post_glue_norm:
                layers.append(post_glue_norm(enc_out))
            if post_glue_activ:
                layers.append(post_glue_activ())
            return layers[0] if len(layers) == 1 else nn.Sequential(*layers,)

        if self.allow_glue:
            for i in range(num_layers):
                down_idx = len(down_channels_outs) - i - 2
                glue_in = down_channels_outs[down_idx] + expected_up_outs[i + 1]
                glue_out = expected_up_outs[i]
                if glue_in != glue_out:
                    glue_conv = OutConv(glue_in, glue_out, kernel_size=glue_kernel_size)                    
                    self.glue_convs.append(expand_glue_conv(glue_conv, glue_out))
                else:
                    self.glue_convs.append(None)
            enc_in = down_channels_outs[-1]
            enc_out = expected_up_outs[0]
            if enc_in != enc_out:
                glue_conv = OutConv(
                    enc_in, enc_out,
                    kernel_size=glue_kernel_size
                )
                self.base_glue = expand_glue_conv(glue_conv, enc_out)
        elif not self.allow_glue and (down_channels_outs[::-1] != expected_up_outs or growth_rate != 2):
            raise Exception("This network will not work without allow_glue = True")

        self.ups = nn.ModuleList()
        for l in range(num_layers):
            d = len(down_channels_outs) - 1
            in_channels = down_channels_outs[d - l]
            out_channels = down_channels_outs[d - l - 1]
            self.ups.append(FlexUp(
                in_channels, out_channels# // factor moved into flexup based on bilinear
                if l < num_layers - 1 else out_channels,
                kernel_size=self.up_kernel_size, conv_layer_size=self.up_layer_size,
                num_conv_layers=self.up_block_layers, bilinear=self.bilinear, norm_layer=norm_layer,
                activ_type=self.up_activ, conv_type=conv_type,
                conv_residual=conv_residual, conv_se_param=conv_se_param,
                upsample_mode=upsample_mode[min(l, len(upsample_mode) - 1)],
                use_checkpointing=use_checkpoints
            ))

        self.outc = OutConv(
            layer_start, self.num_classes, kernel_size=self.out_kernel_size,
            depth=self.out_depth, norm_layer=norm_layer, conv_type=conv_type,
            use_checkpointing=use_checkpoints
        )

    def get_glue_conv(self, idx):
        if idx >= len(self.glue_convs):
            return
        return self.glue_convs[idx]

    def apply_base_glue(self, t):
        if self.base_glue is None:
            return t
        return self.base_glue(t)

    def forward_down(self, *xs,):
        if len(xs) == 1 and len(self.downs) != 1:
            xs = [xs[0][:, i * self.num_classes:(i + 1) * self.num_classes] for i in range(len(self.downs))]
        assert len(xs) == len(self.downs), f"Incorrect number of inputs. Expected {len(self.downs)}, got {len(xs)}"
        channel_outs = [unet_down(x) for unet_down, x in zip(self.downs, xs)]
        prev = [torch.cat(layer_outs, axis=1) for layer_outs in zip(*channel_outs)]
        return prev

    def forward_up(self, down_results):
        tmp2 = down_results[-1]
        tmp1 = down_results[-2]
        out = self.layer_forward(
            self.ups[0],
            self.apply_base_glue(tmp2),
            tmp1,
            self.get_glue_conv(0)
        )
        for i, up_layer in enumerate(self.ups[1:]):
            out = self.layer_forward(
                up_layer, out, down_results[-(i + 3)],
                self.get_glue_conv(i + 1),
                use_checkpoint=self.use_checkpoints
            )
        logits = self.outc(out)
        return logits

    def forward(self, *xs,):
        forward_down = (
            (lambda *inps: checkpoint(self.forward_down, *inps, preserve_rng_state=True, use_reentrant=False))
            if self.use_checkpoints else self.forward_down
        )
        forward_up = (
            (lambda *inps: checkpoint(self.forward_up, *inps, preserve_rng_state=True, use_reentrant=False))
            if self.use_checkpoints else self.forward_up
        )
        down_results = forward_down(*xs,)
        logits = forward_up(down_results)
        return self.out_activ(logits)


class FlexUNet(MultiInpFlexUNet):
    allow_glue: bool = Field(True, description="Allow glue convs to align shapes between down and up paths")
    bilinear: bool = Field(False, description="Whether to use bilinear upsampling")
    use_checkpoints: bool = Field(True, description="Enable checkpointing to save memory")
    input_dropout_prob: float = Field(0.0, description="Dropout probability for input")

    def __init__(self, num_channels, **kwargs):
        super().__init__(
            num_channels=num_channels, input_allocs=[kwargs.get("start", 64)],
            channel_allocs=None, **kwargs
        )
        self.input_dropout = ToggleDropout2d(self.input_dropout_prob)

    def forward_down(self, x):
        x = self.input_dropout(x)
        return self.downs[0](x)


class EncInjectFlexUNet(FlexUNet):
    injector: AutoLambda[nn.Module] = Field(
        Lambda("lambda enc, inj: enc"),
        description="Function to inject vector into encoder output"
    )

    def forward(self, inp, inject=None):
        if is_none(inject):
            inject = inp[:, -self.num_classes:]
            inp = inp[:, :-self.num_classes]
        forward_down = (
            (lambda *inps: checkpoint(self.forward_down, *inps, preserve_rng_state=True, use_reentrant=False))
            if self.use_checkpoints else self.forward_down
        )
        forward_up = (
            (lambda *inps: checkpoint(self.forward_up, *inps, preserve_rng_state=True, use_reentrant=False))
            if self.use_checkpoints else self.forward_up
        )
        down_results = forward_down(inp)
        down_results[-1] = self.injector(down_results[-1], inject)
        logits = forward_up(down_results)
        return self.out_activ(logits)


class UNetEnc(UNetDown):
    flatten_out: bool = Field(True, description="Whether to flatten the encoder output")
    return_shapes: bool = Field(False, description="Whether to return shapes along with activations")
    out_activ: AutoLambda[nn.Module] = Field(
        Lambda("lambda x: x"), description="Activation applied to the encoder output"
    )
    activation: AutoLambda[nn.Module] = Field(
        nn.SiLU, description="Activation function to use in encoder"
    )
    conv_residual: Optional[Union[str, dict, float, int, AutoLambda]] = Field(
        0.1, description="Residual connection strength for convolution blocks"
    )
    bilinear: bool = Field(False, description="Whether to use bilinear upsampling")

    def forward(self, x):        
        out = self.input_layer(x)
        shapes = [out.shape]
        for down_layer in self.downs:
            out = self.layer_forward(
                down_layer, out,
                use_checkpoint=self.use_checkpointing
            )
            shapes.append(out.shape)
        if self.flatten_out:
            out = Flatten()(out)
        out = self.out_activ(out)
        if self.return_shapes:
            return out, shapes
        return out


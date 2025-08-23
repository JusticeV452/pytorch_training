import math
import torch

from typing import Optional, Sequence, Tuple, Union
from pydantic import Field
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm as torch_spectral_norm
from torch.utils.checkpoint import checkpoint_sequential

from .components import Mean, ResConn, ToggleDropout2d, UnFlatten, construct_classifier, construct_conv_classifier
from .convs.multi_kernel import MultiKernelConv
from serialization import AutoLambda, Lambda
from serialization.nn import SerializableModel
from utils import get_group_norm, int_round, next_largest_dividend


class Discriminator(SerializableModel):
    size: int = Field(..., description="Input image size")
    num_channels: int = Field(3, description="Number of image channels")
    layer_depth: int = Field(1, description="Number of convolutional encoding layers")
    classifier_depth: int = Field(1, description="Number of MLP layers after encoding")
    classifier_reduction_factor: int = Field(4, description="Factor to reduce features for classification")
    kernel_size: int = Field(5, description="Convolutional kernel size")
    layer_transform: AutoLambda[nn.Module] = Field(
        Lambda(nn.LeakyReLU, negative_slope=0.2),
        description="Activation used in convolutional layers"
    )
    enc_norm_layer: AutoLambda[nn.Module] = Field(
        Lambda(nn.BatchNorm2d, momentum=0.9),
        description="Normalization layer used in encoder"
    )
    cls_norm_layer: AutoLambda[nn.Module] = Field(
        Lambda(nn.BatchNorm1d, momentum=0.9),
        description="Normalization layer used in classifier"
    )
    num_checkpoint_segments: int = Field(0, exclude=True, description="How many total checkpointed segments to use")
    num_encoder_checkpoints: Optional[int] = Field(
        num_checkpoint_segments, description="Optional override for encoder checkpoint segments"
    )
    num_classifier_checkpoints: Optional[int] = Field(
        num_checkpoint_segments, description="Optional override for classifier checkpoint segments"
    )

    def __init__(self, size, **kwargs):
        super().__init__(size=size, **kwargs)
        self.in_mult = 256
        kernel_size = self.kernel_size
        layer_depth = self.layer_depth
        layer_transform = self.layer_transform

        # Construct encoder
        def conv_layer(in_channels, out_channels, kernel_size=kernel_size, padding=2, stride=2):
            layer = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)]
            if self.enc_norm_layer:
                layer.append(self.enc_norm_layer(out_channels))
            layer.append(layer_transform())
            return layer

        layers = [
            nn.Conv2d(self.num_channels, 32, kernel_size, padding=2, stride=1),
            layer_transform()
        ] + conv_layer(32, 128)
        [layers.extend(conv_layer(128, 128, stride=1)) for _ in range(layer_depth - 1)]

        num_layers = int(round(math.log(size, 2))) - 5
        for l in range(num_layers):
            in_channels = 2 ** (7 + l)
            out_channels = 2 * in_channels
            layers += conv_layer(in_channels, out_channels)
            [layers.extend(conv_layer(out_channels, out_channels, stride=1)) for _ in range(layer_depth - 1)]

        layers += conv_layer(4 * size, 4 * size)
        enc_dim = self.in_mult * size
        layers.append(UnFlatten((enc_dim,)))
        self.encoder = nn.Sequential(*layers,)
        self.classifier = self.construct_classifier(
            enc_dim, self.classifier_depth,
            self.classifier_reduction_factor,
            self.cls_norm_layer, layer_transform
        )
        self.to(self.dtype)

    def construct_classifier(
            self, in_dim, classifier_depth, classifier_reduction_factor,
            cls_norm_layer, layer_transform, spectral_norm=False, **kwargs
        ):
        # Construct classifier
        classif_layers = []
        mult = self.in_mult
        size = self.size
        end_layer_mult = 8

        def linear_layer(in_dim, out_dim):
            layer = [nn.Linear(in_dim, out_dim)]
            if spectral_norm:
                layer[0] = torch_spectral_norm(layer[0])
            if cls_norm_layer:
                layer.append(cls_norm_layer(out_dim))
            layer.append(layer_transform())
            return layer

        for i in range(classifier_depth):
            if mult < end_layer_mult:
                raise Exception(f"Too many layers or too high layer reduction: current_mult = {mult}")
            out_dim = (
                end_layer_mult * size if (i == classifier_depth - 1)
                else in_dim // classifier_reduction_factor
            )
            classif_layers.extend(linear_layer(in_dim, out_dim))
            mult //= classifier_reduction_factor
            in_dim = out_dim

        classif_layers.extend([
            nn.Linear(end_layer_mult * size, 1),
            nn.Sigmoid()
        ])
        return nn.Sequential(*classif_layers)

    def encode(self, x):
        return self.checkpoint_sequential_run(self.encoder, self.num_encoder_checkpoints, x)

    def classify(self, x):
        return self.forward(x)[0]

    def forward(self, x):        
        enc = self.encode(x)
        out = self.checkpoint_sequential_run(
            self.classifier, self.num_classifier_checkpoints, enc
        )
        return out, enc

    def set_dropout_state(self, enabled):
        pass

    def get_dropout_state(self):
        return False


class FlexDiscrim(Discriminator):
    size: Optional[int] = Field(None, description="Input image size")
    num_layers: int = Field(2, description="Number of convolutional layers in each encoder block")
    start: int = Field(32, description="Number of output channels for first layer")
    second_out: int = Field(128, description="Number of output channels for second layer")
    growth_rate: float = Field(2.0, description="Growth factor for channels per layer")
    layer_transform: AutoLambda[nn.Module] = Field(
        nn.LeakyReLU, description="Activation function used in encoder blocks"
    )
    dropout: Union[float, Sequence[float]] = Field(0.0, description="Dropout applied after each encoder block")
    kernel_sizes: Optional[list[Union[int, Tuple[int, int]]]] = Field(
        None, description="Optional list of kernel sizes per encoder layer"
    )
    init_func: Optional[AutoLambda] = Field(
        None, description="Optional weight initialization function"
    )
    enc_norm_layer: AutoLambda[nn.Module] = Field(
        Lambda(get_group_norm, parent_kwargs_={"num_groups": 32}), description="Normalization function for encoder"
    )
    cls_norm_layer: AutoLambda[nn.Module] = Field(
        Lambda(get_group_norm, parent_kwargs_={"num_groups": 8}), description="Normalization function for classifier"
    )
    separate_encs: int = Field(1, description="Number of separate encoders for input splitting")
    enc_transform: Optional[AutoLambda[nn.Module]] = Field(
        layer_transform, description="Optional transform applied to encoder output"
    )
    cls_transform: Optional[AutoLambda[nn.Module]] = Field(
        layer_transform, description="Optional transform applied before classifier"
    )
    cls_type: str = Field("linear", description="Type of classifier head (e.g., 'linear', 'conv')")
    enc_dim_reducer: Optional[AutoLambda[nn.Module]] = Field(
        Mean, description="Module used to reduce enc dimension to (B, 1) when using linear classifier"
    )
    spectral_norm: bool = Field(False, description="Whether to use spectral normalization")
    se_block_gen: Optional[AutoLambda[nn.Module]] = Field(
        None, description="Optional SE block generator to apply after conv blocks"
    )
    conv_type: AutoLambda[nn.Module] = Field(
        nn.Conv2d, description="Base convolutional layer type"
    )
    block_conv_type: AutoLambda[nn.Module] = Field(
        conv_type, description="Convolution type used in conv blocks"
    )
    use_conv_residual: bool = Field(False, description="Whether to use residual connections in conv blocks")
    cls_kwargs: dict = Field({"channel_div": 8}, description="Additional kwargs passed to classifier")
    channel_div: int = Field(32, description="Ensure all channel counts are divisible by this")

    def __init__(self, *args, **kwargs):
        SerializableModel.__init__(self, *args, **kwargs)
        self.dropouts = []
        self.dropouts_enabled = True

        kernel_size = self.kernel_size
        layer_depth = self.layer_depth
        conv_type = self.conv_type
        enc_norm_layer = self.enc_norm_layer
        num_channels = self.num_channels
        cls_type = self.cls_type
        num_layers = self.num_layers
        size = self.size

        if self.kernel_sizes is None:
            assert kernel_size
            self.kernel_sizes = [kernel_size for _ in range(num_layers + 1)]
        if len(self.kernel_sizes) < num_layers + 1:
            for _ in range(num_layers + 1 - len(self.kernel_sizes)):
                self.kernel_sizes.append(self.kernel_sizes[-1])
        kernel_sizes = self.kernel_sizes

        layers = []
        assert (self.enc_dim_reducer or size) and (size or num_layers)
        num_layers = (
            int(round(math.log(size, 2))) - 5 if num_layers is None
            else num_layers
        )

        num_dropouts = ((num_layers + 3) + (num_layers + 1) * (layer_depth - 1))
        dropout_probs = []
        if type(self.dropout) in [int, float]:
            dropout_probs = [self.dropout] * num_dropouts
        if len(dropout_probs) < num_dropouts:
            dropout_probs += [dropout_probs[-1]] * (num_dropouts - len(dropout_probs))
        assert all(0 <= d <= 1 for d in dropout_probs), "Dropout should be in range [0, 1]"
        dropout_idx = 0

        # Construct encoder
        def conv_layer(
                in_channels, out_channels, kernel_size=kernel_sizes[1],
                padding=None, stride=2, cnv_type=conv_type, use_spectral_norm=False
            ):
            nonlocal dropout_idx
            nonlocal layers
            if type(kernel_size) is tuple:
                cnv_type = Lambda(MultiKernelConv, conv_type=cnv_type, spectral_norm=use_spectral_norm)
                padding = "same"
                norm_wrapper = lambda x: x
            else:
                norm_wrapper = torch_spectral_norm if use_spectral_norm else lambda x: x
                padding = kernel_size // 2 if padding is None else padding
            layer = [norm_wrapper(cnv_type(in_channels, out_channels, kernel_size, padding=padding, stride=stride))]

            self.dropouts.append(ToggleDropout2d(dropout_probs[dropout_idx]))
            layer.append(self.dropouts[-1])
            dropout_idx += 1

            if enc_norm_layer:
                layer.append(enc_norm_layer(out_channels))
            layer.append(self.enc_transform())
            return layer

        layers += conv_layer(num_channels // self.separate_encs, self.start, kernel_sizes[0], stride=1)
        layers += conv_layer(self.start, self.second_out, kernel_size=kernel_sizes[0])
        [
            layers.extend(conv_layer(self.second_out, self.second_out, kernel_size=kernel_sizes[0], stride=1))
            for _ in range(layer_depth - 1)
        ]

        in_channels = self.second_out
        for l in range(num_layers):
            out_channels = next_largest_dividend(int_round(self.growth_rate * in_channels), self.channel_div)
            layers += conv_layer(
                in_channels, out_channels, kernel_size=kernel_sizes[l + 1],
                use_spectral_norm=self.spectral_norm and layer_depth == 1
            )
            extend_layers = sum([
                conv_layer(
                    out_channels, out_channels, stride=1,
                    cnv_type=self.block_conv_type,
                    use_spectral_norm=self.spectral_norm and (i == layer_depth - 2)
                ) for i in range(layer_depth - 1)
            ], [])
            layers += [ResConn(extend_layers)] if layer_depth - 1 and self.use_conv_residual else extend_layers
            if self.se_block_gen:
                layers.append(self.se_block_gen(out_channels))
            in_channels = out_channels

        final_num_channels = out_channels
        layers += conv_layer(final_num_channels, final_num_channels, kernel_size=kernel_sizes[-1])
        self.cls_type = cls_type
        if cls_type == "linear" and self.enc_dim_reducer:
            self.in_mult = None
            self.enc_dim = enc_dim = final_num_channels
            layers.extend([
                self.enc_dim_reducer(),
                UnFlatten((enc_dim,))
            ])
            classifier_maker = construct_classifier
        elif cls_type == "linear":
            # Strict: image size must match self.size
            final_pixel_per_channel = (2 ** (int_round(math.log(size, 2)) - (num_layers + 2))) ** 2
            self.enc_dim = enc_dim = final_num_channels * final_pixel_per_channel
            self.in_mult = enc_dim // size
            layers.append(UnFlatten((enc_dim,)))
            classifier_maker = self.construct_classifier
        elif cls_type == "conv":
            self.in_mult = None
            self.enc_dim = enc_dim = final_num_channels
            classifier_maker = construct_conv_classifier
        self.encoder = nn.Sequential(*layers,)

        if self.separate_encs > 1:
            enc_dim *= self.separate_encs

        self.classifier = classifier_maker(
            enc_dim, self.classifier_depth,
            self.classifier_reduction_factor,
            self.cls_norm_layer, self.cls_transform,
            spectral_norm=self.spectral_norm,
            **self.cls_kwargs
        )
        self.to(self.dtype)
        if self.init_func is not None:
            self.apply(self.init_func)

    def set_dropout_state(self, enabled):
        for dropout in self.dropouts:
            dropout.enabled = enabled
        self.dropouts_enabled = enabled

    def get_dropout_state(self):
        return self.dropouts_enabled

    def forward(self, x):
        encs = []
        for i in range(self.separate_encs):
            num_classes = (self.num_channels // self.separate_encs)
            enc_inp = x[:, i * num_classes: (i + 1) * num_classes]
            enc = self.encode(enc_inp)
            encs.append(enc)
        classify = (
            (lambda inp: checkpoint_sequential(self.classifier, self.num_classifier_checkpoints, inp))
            if self.num_classifier_checkpoints else self.classifier
        )
        out = classify(torch.cat(encs, axis=1))
        return out, enc


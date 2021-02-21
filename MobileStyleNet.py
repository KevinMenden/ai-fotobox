import torch.nn as nn

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            nn.Hardswish()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class InvertedResidual(nn.Module):
    """
    Inverted Residual block as defined for MobileNetv2
    """

    def __init__(self, dim_in, dim_hidden, dim_out, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        self.skip_connection = False
        self.activation = nn.Hardswish if use_hs else nn.ReLU
        self.squeeze_excite = SqueezeExcitation(dim_hidden) if use_se else nn.Identity()

        if dim_in == dim_out and stride == 1:
            self.skip_connection = True

        if dim_in == dim_hidden:
            self.conv = nn.Sequential(
                nn.Conv2d(dim_hidden, dim_hidden, kernel_size=kernel_size, stride=stride,
                          padding=(kernel_size - 1) // 2, groups=dim_hidden, bias=False),
                nn.InstanceNorm2d(dim_hidden, affine=True),
                self.activation(),
                self.squeeze_excite,
                nn.Conv2d(dim_hidden, dim_out, 1, 1, 0, bias=False),
                nn.InstanceNorm2d(dim_out, affine=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(dim_in, dim_hidden, 1, 1, 0, bias=False),
                nn.InstanceNorm2d(dim_hidden, affine=True),
                self.activation(),
                nn.Conv2d(dim_hidden, dim_hidden, kernel_size=kernel_size, stride=stride,
                          padding=(kernel_size - 1) // 2, groups=dim_hidden, bias=False),
                nn.InstanceNorm2d(dim_hidden, affine=True),
                self.squeeze_excite,
                self.activation(),
                nn.Conv2d(dim_hidden, dim_out, 1, 1, 0, bias=False),
                nn.InstanceNorm2d(dim_out, affine=True)
            )

    def forward(self, x):
        if self.skip_connection:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SegmentationHead(nn.Module):
    """
    LR-ASPP-like segmentation head for fast segmentation
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/lraspp.py
    """

    def __init__(self, n_channels_in, n_classes, n_skip_features, target_size=224):
        super(SegmentationHead, self).__init__()

        # convolution branch
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels_in, 128, kernel_size=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True)
        )

        # pool branch
        self.pool_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_channels_in, 128, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, n_classes, kernel_size=1),
            nn.InstanceNorm2d(n_classes, affine=True),
            nn.ReLU(inplace=True)
        )

        self.upsample1 = nn.Upsample(size=112, mode="bilinear", align_corners=True)
        self.skip_conv = nn.Conv2d(n_skip_features, n_classes, kernel_size=1, )
        self.upsample2 = nn.Upsample(size=224, mode="bilinear", align_corners=True)
        self.conv_out = nn.Conv2d(n_classes, n_classes, 1)

    def forward(self, x, skip_connection):
        # Convolution branch
        conv_branch = self.conv1(x)

        # Pool branch
        pool = self.pool_branch(x)

        # concat pool and convolution branch
        x = conv_branch * pool
        x = self.upsample1(x)
        x = self.conv2(x)

        # add the skip connection
        skip = self.skip_conv(skip_connection)
        x = x + skip

        # Final upsampling and output
        x = self.upsample2(x)
        x = self.conv_out(x)

        return x


class Mobilenet(nn.Module):
    def __init__(self, architecture, num_classes=91, width_multiplier=1.0):
        super(Mobilenet, self).__init__()
        self.input_channel = _make_divisible(16 * width_multiplier, 8)
        self.architecture = architecture
        self.skip_layer_idx = 0

        # ========================================= #
        # Build the network architecture
        # ========================================= #

        # Input convolution
        self.network_layers = [nn.Sequential(
            nn.Conv2d(3, self.input_channel, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(self.input_channel, affine=True),
            nn.Hardswish(),
        )]

        # Network body of InvertedResidual blocks with SqueezeExcitation blocks
        out_dim_list = []
        for k, t, c, use_se, use_hs, s in self.architecture:
            output_channel = _make_divisible(c * width_multiplier, 8)
            out_dim_list.append(output_channel)
            exp_size = _make_divisible(self.input_channel * t, 8)

            self.network_layers.append(InvertedResidual(
                dim_in=self.input_channel,
                dim_hidden=exp_size,
                dim_out=output_channel,
                kernel_size=k,
                stride=s,
                use_se=use_se,
                use_hs=use_hs
            ))

            self.input_channel = output_channel
        self.network = nn.Sequential(*self.network_layers)

        # Segmentation head
        self.segmentation = SegmentationHead(48, num_classes, 16)

    def forward(self, x):
        skip_connection = None

        for i in range(len(self.network_layers)):
            x = self.network_layers[i](x)
            if i == self.skip_layer_idx:
                skip_connection = x

        x = self.segmentation(x, skip_connection)

        return x


base_architecture = [
    # k, t, c, SE, HS, s
    [3, 1, 16, 1, 0, 2],
    [3, 4.5, 24, 0, 0, 2],
    [3, 3.67, 24, 0, 0, 1],
    [5, 4, 40, 1, 1, 2],
    [5, 6, 40, 1, 1, 1],
    [5, 6, 40, 1, 1, 1],
    [5, 3, 48, 1, 1, 1],
    [5, 3, 48, 1, 1, 1],
]

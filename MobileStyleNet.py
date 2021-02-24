import torch.nn as nn
import torch

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
        new_v = new_v + divisor
    return new_v


class InvertedResidual(nn.Module):
    """
    Inverted Residual block as defined for MobileNetv2
    """

    def __init__(self, channel_in, channel_out, kernel_size, stride):
        super(InvertedResidual, self).__init__()
        self.skip_connection = False
        self.activation = nn.ReLU
        self.reflection_pad = kernel_size // 2
        self.relu = nn.ReLU()

        if channel_in == channel_out and stride == 1:
            self.skip_connection = True

        self.conv = nn.Sequential(
            nn.ReflectionPad2d(self.reflection_pad),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=1),
            nn.InstanceNorm2d(channel_out, affine=True),
            nn.ReLU(),
            nn.ReflectionPad2d(self.reflection_pad),
            nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, stride=stride, groups=channel_in),
            nn.InstanceNorm2d(channel_out, affine=True),
            nn.ReLU(),
            #nn.ReflectionPad2d(self.reflection_pad),
            nn.Conv2d(channel_out, channel_out, kernel_size=1, stride=1),
            nn.InstanceNorm2d(channel_out, affine=True),
        )

    def forward(self, x):
        if self.skip_connection:
            return self.relu(x + self.conv(x))
        else:
            return self.relu(self.conv(x))


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class MobileStyleNet(nn.Module):
    def __init__(self):
        super(MobileStyleNet, self).__init__()

        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)

        self.inv_res1 = InvertedResidual(32, 64, 3, 2)
        self.inv_res2 = InvertedResidual(64, 64, 3, 2)
        self.inv_res3 = InvertedResidual(64, 64, 3, 1)



        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=4)
        self.in4 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        skip_conn = self.relu(self.in1(self.conv1(x)))

        x = self.inv_res1(skip_conn)
        x = self.inv_res2(x)
        x = self.inv_res3(x)


        x = self.relu(self.in4(self.deconv1(x)))
        x = x + skip_conn
        x = self.deconv3(x)

        return x

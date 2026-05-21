from inspect import isfunction
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import QARepVGGBlock, repvgg_model_convert

def get_activation_layer(activation):
    """
    Create an activation layer from a string/function/module.

    Parameters
    ----------
    activation : function, or str, or nn.Module
        Activation function or name of activation function.

    Returns
    -------
    nn.Module
        Activation layer.
    """
    assert activation is not None, "Activation must be provided"
    if isfunction(activation):
        return activation()
    else:
        assert isinstance(activation, nn.Module), "Activation must be a function or nn.Module"
        return activation

class ConvBlock(nn.Module):
    """
    Standard convolution block with Batch normalization and activation.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple
        Size of convolution window.
    stride : int or tuple
        Stride of the convolution.
    padding : int or tuple, or list of 4 ints
        Padding for convolution.
    dilation : int or tuple, default 1
        Dilation rate for convolution.
    groups : int, default 1
        Number of groups in convolution.
    bias : bool, default False
        If True, adds a learnable bias.
    use_bn : bool, default True
        If True, applies BatchNorm.
    bn_eps : float, default 1e-5
        Value added to denominator for numerical stability in BatchNorm.
    activation : function or nn.Module, default nn.ReLU(inplace=True)
        Non-linearity.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=(lambda: nn.ReLU(inplace=True))):
        super(ConvBlock, self).__init__()
        self.activate = (activation is not None)
        self.use_bn = use_bn
        self.use_pad = (isinstance(padding, (list, tuple)) and (len(padding) == 4))

        # ZeroPad2d only if padding set as 4-tuple
        if self.use_pad:
            self.pad = nn.ZeroPad2d(padding=padding)
            padding = 0

        # Main convolution layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        # Optional batch normalization
        if self.use_bn:
            self.bn = nn.BatchNorm2d(
                num_features=out_channels,
                eps=bn_eps
            )
        # Optional activation
        if self.activate:
            self.activ = get_activation_layer(activation)

    def forward(self, x):
        # Optional manual padding
        if self.use_pad:
            x = self.pad(x)
        x = self.conv(x)
        # Optional batch normalization
        if self.use_bn:
            x = self.bn(x)
        # Optional non-linearity
        if self.activate:
            x = self.activ(x)
        return x

def conv1x1_block(
    in_channels,
    out_channels,
    stride=1,
    padding=0,
    groups=1,
    bias=False,
    use_bn=True,
    bn_eps=1e-5,
    activation=(lambda: nn.ReLU(inplace=True))
):
    """
    1x1 version of the standard ConvBlock.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation
    )

def conv3x3_block(
    in_channels,
    out_channels,
    stride=1,
    padding=1,
    dilation=1,
    groups=1,
    bias=False,
    use_bn=True,
    bn_eps=1e-5,
    activation=(lambda: nn.ReLU(inplace=True))
):
    """
    3x3 version of the standard ConvBlock.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation
    )

def dwconv_block(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=1,
    dilation=1,
    bias=False,
    use_bn=True,
    bn_eps=1e-5,
    activation=(lambda: nn.ReLU(inplace=True))
):
    """
    Depthwise version of the standard ConvBlock.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=out_channels,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation
    )

def channel_shuffle(x, groups):
    """
    Channel shuffle operation as in ShuffleNet.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    groups : int
        Number of groups to split channels.

    Returns
    -------
    torch.Tensor
        Output tensor with shuffled channels.
    """
    batch, channels, height, width = x.size()
    # assert channels % groups == 0
    channels_per_group = channels // groups
    x = x.view(batch, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch, channels, height, width)
    return x

def channel_shuffle2(x, groups):
    """
    Alternative channel shuffle operation.
    """
    batch, channels, height, width = x.size()
    # assert channels % groups == 0
    channels_per_group = channels // groups
    x = x.view(batch, channels_per_group, groups, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch, channels, height, width)
    return x

class ChannelShuffle2(nn.Module):
    """
    Module wrapper for channel_shuffle2.
    Stores groups for use in forward pass.
    """
    def __init__(self, channels, groups):
        super(ChannelShuffle2, self).__init__()
        if channels % groups != 0:
            raise ValueError("channels must be divisible by groups")
        self.groups = groups

    def forward(self, x):
        return channel_shuffle2(x, self.groups)

class RepConv3x3(nn.Module):
    """
    RepVGG 3x3 building block.
    """
    def __init__(self, in_channels, out_channels, stride=1, groups=1, use_se=False, deploy=False, identity=False):
        super(RepConv3x3, self).__init__()
        self.repconv3x3 = RepVGGBlock(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            groups=groups,
            stride=stride,
            deploy=deploy,
            use_se=use_se,
            identity=identity
        )
        if deploy:
            self.convert_for_inference()

    def forward(self, x):
        out = self.repconv3x3(x)
        return out

class QARepConv3x3(nn.Module):
    """
    Quantization-aware RepVGG 3x3 building block.
    """
    def __init__(self, in_channels, out_channels, stride=1, groups=1, use_se=False, deploy=False, identity=False):
        super(QARepConv3x3, self).__init__()
        self.repconv3x3 = QARepVGGBlock(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            groups=groups,
            stride=stride,
            deploy=deploy,
            use_se=use_se,
            identity=identity
        )
        if deploy:
            self.convert_for_inference()

    def forward(self, x):
        out = self.repconv3x3(x)
        return out

class GSConv(nn.Module):
    """
    Ghost Squeeze Convolution: using pointwise + depthwise, with channel shuffle.
    """
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, dw_size=9):
        super(GSConv, self).__init__()
        # Half output channels with standard convolution
        self.conv = conv3x3_block(in_channels, out_channels // 2, stride=stride)
        # Depthwise convolution on expanded part
        self.dw_conv = dwconv_block(out_channels // 2, out_channels // 2, kernel_size=dw_size, padding=dw_size // 2)
        # Channel shuffle layer
        self.shuffle_op = ChannelShuffle2(out_channels, 2) 

    def forward(self, x):
        out_conv = self.conv(x)
        out_dw = self.dw_conv(out_conv)
        out = torch.cat([out_conv, out_dw], 1)
        out = self.shuffle_op(out)
        return out

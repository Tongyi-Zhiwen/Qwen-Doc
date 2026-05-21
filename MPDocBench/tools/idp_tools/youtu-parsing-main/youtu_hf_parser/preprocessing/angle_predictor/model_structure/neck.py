from builtins import range
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import conv1x1_block, RepConv3x3, repvgg_model_convert, QARepConv3x3

class REP_FPN_IMPL(nn.Module):
    def __init__(self, input_channels, planes=128, use_se=False, block="REP", identity=False):
        """
        Initialize the REP FPN implementation.

        Args:
            input_channels (list[int]): List of input channels for each feature map.
            planes (int): Number of output channels for intermediate features.
            use_se (bool): Whether to use Squeeze-and-Excitation modules.
            block (str): Type of convolution block to use ("REP" or "QAREP").
            identity (bool): Whether to use identity shortcut in convolutions.
        """
        super(REP_FPN_IMPL, self).__init__()
        
        # Select block type
        if block == "REP":
            block_cls = RepConv3x3
        elif block == "QAREP":
            block_cls = QARepConv3x3
        else:
            raise ValueError(f"Unsupported block type: {block}")

        # Initialize lateral 1x1 conv layers for each input feature map
        self.conv_lat = nn.ModuleList([
            conv1x1_block(ch, planes) for ch in input_channels
        ])

        # Initialize smoothing conv blocks (e.g., RepConv3x3) for top-down pathway
        self.conv_smooth = nn.ModuleList([
            block_cls(planes, planes, stride=1, use_se=use_se, identity=identity)
            for _ in range(1, len(input_channels))
        ])

    def resize_add(self, x, y):
        """
        Resize x to match y by upsampling and add.

        Args:
            x (Tensor): Feature map to be resized and added.
            y (Tensor): Reference feature map.

        Returns:
            Tensor: Sum of upsampled x and y.
        """
        _, _, H, W = y.shape
        x_upsampled = F.interpolate(x, size=(H, W), mode="nearest")
        return x_upsampled + y

    def forward(self, feats):
        """
        Forward pass of the feature pyramid network.

        Args:
            feats (list[Tensor]): List of feature maps in order from low to high resolution
                                  (e.g., [c2, c3, c4, c5]).

        Returns:
            list[Tensor]: List of output features in order from high to low resolution
                          (e.g., [p5, p4, p3, p2]).
        """
        # Apply lateral 1x1 conv to each feature map
        conv_lats = [self.conv_lat[i](feat) for i, feat in enumerate(feats)]
        
        # Reverse to start from highest resolution feature
        conv_lats.reverse()
        conv_out = [conv_lats[0]]

        # Top-down pathway with lateral connections and smoothing
        for i, feat in enumerate(conv_lats[1:]):
            smooth_feat = self.conv_smooth[i](self.resize_add(conv_out[-1], feat))
            conv_out.append(smooth_feat)

        # Return features in order: [high resolution, ..., low resolution]
        return conv_out

class REP_FPN(nn.Module):
    def __init__(self, input_channels, planes=128, use_se=False, block="REP", identity=False):
        """
        Wrapper for the REP_FPN_IMPL.

        Args:
            input_channels (list[int]): List of input channels for each feature map.
            planes (int): Number of output channels for intermediate features.
            use_se (bool): Whether to use SE modules.
            block (str): Type of convolution block ("REP" or "QAREP").
            identity (bool): Whether to use identity shortcuts.
        """
        super(REP_FPN, self).__init__()
        self.rep_fpn = REP_FPN_IMPL(input_channels, planes, use_se, block, identity)
    
    def forward(self, x):
        return self.rep_fpn(x)

    def convert_for_inference(self):
        """
        Convert the model for inference by replacing training-time modules,
        e.g., convert RepVGG blocks to their inference counterparts.
        """
        self.rep_fpn = repvgg_model_convert(self.rep_fpn)

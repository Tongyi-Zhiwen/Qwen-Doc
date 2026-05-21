import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import conv1x1_block, RepConv3x3, repvgg_model_convert, QARepConv3x3, conv3x3_block

class FeatCat(nn.Module):
    """
    Feature aggregation module that resizes input features and concatenates them.
    """
    def __init__(self):
        super(FeatCat, self).__init__()
    
    def resize_scale(self, x, scale):
        """
        Resize feature map by a scale factor using nearest neighbor interpolation.
        """
        return F.interpolate(x, scale_factor=scale, mode="nearest")
    
    def forward(self, fpn_feats):
        """
        Resize features to appropriate scales and concatenate along channel dimension.
        Args:
            fpn_feats (list[Tensor]): List of feature maps ordered from high to low resolution.
        Returns:
            Tensor: Concatenated feature map.
        """
        # Calculate scaling factors for each feature map based on number of features
        out_scale = [2 ** (len(fpn_feats) - 1 - i) for i in range(len(fpn_feats))]
        # Resize each feature map
        for i in range(len(fpn_feats)):
            fpn_feats[i] = self.resize_scale(fpn_feats[i], out_scale[i])
        # Concatenate along channel dimension
        out = torch.cat(fpn_feats, dim=1)
        return out


class REP_Hourglass_IMPL(nn.Module):
    """
    Hierarchical hourglass module with features at different scales, supporting multi-scale fusion.
    """
    def __init__(self, inplanes, inner_planes, use_se=False, 
                 identity=False, skip_connnect=True, expand=True, block="REP"):
        super(REP_Hourglass_IMPL, self).__init__()
        # Determine channel expansion ratios based on expand parameter
        multiply_r = [1, 2, 4, 8] if expand else [1, 1, 1, 1]
        
        # Choose convolution block type
        if block == "REP":
            block_cls = RepConv3x3
        elif block == "QAREP":
            block_cls = QARepConv3x3
        else:
            raise ValueError(f"Unknown block type: {block}")

        # Feature concatenation module
        self.feat_cat = FeatCat()
        # Fuse layer to reduce channels
        self.fuse = conv1x1_block(inplanes, inner_planes)

        # Downsampling path with sequential conv layers
        self.conv3x3_down0 = block_cls(
            inner_planes, 
            inner_planes * multiply_r[0], 
            stride=1, 
            use_se=use_se, 
            identity=identity
        )
        self.conv3x3_down1 = block_cls(
            inner_planes * multiply_r[0], 
            inner_planes * multiply_r[1], 
            stride=1, use_se=use_se, 
            identity=identity
        )
        self.conv3x3_down2 = block_cls(
            inner_planes * multiply_r[1], 
            inner_planes * multiply_r[2], 
            stride=1, 
            use_se=use_se, 
            identity=identity
        )
        self.conv3x3_down3 = block_cls(
            inner_planes * multiply_r[2], 
            inner_planes * multiply_r[3], 
            stride=1, 
            use_se=use_se, 
            identity=identity
        )

        # Upsampling path with sequential conv layers
        self.conv3x3_up0 = block_cls(
            inner_planes * multiply_r[3], 
            inner_planes * multiply_r[2], 
            stride=1, 
            use_se=use_se, 
            identity=identity
        )
        self.conv3x3_up1 = block_cls(
            inner_planes * multiply_r[2], 
            inner_planes * multiply_r[1], 
            stride=1, 
            use_se=use_se, 
            identity=identity
        )
        self.conv3x3_up2 = block_cls(
            inner_planes * multiply_r[1], 
            inner_planes * multiply_r[0], 
            stride=1, 
            use_se=use_se, 
            identity=identity
        )
        self.conv3x3_up3 = block_cls(
            inner_planes * multiply_r[0], 
            inner_planes, 
            stride=1, 
            use_se=use_se, 
            identity=identity
        )

        self.skip_connnect = skip_connnect

    def resize_add(self, x, y, add=True):
        """
        Resize x to match y's size and add it if add=True, else just resize.
        Args:
            x (Tensor): Feature map to resize.
            y (Tensor): Reference feature map.
            add (bool): Whether to add after resizing.
        Returns:
            Tensor: Resized (and possibly added) feature map.
        """
        _, _, H, W = y.shape
        if add:
            return F.interpolate(x, size=(H, W), mode="nearest") + y
        else:
            return F.interpolate(x, size=(H, W), mode="nearest")
        
    def avg_pool(self, x):
        """
        Perform average pooling for downsampling.
        """
        return F.avg_pool2d(x, kernel_size=2, stride=2)

    def forward(self, x):
        """
        Forward pass of the hourglass module.
        Args:
            x (list[Tensor]): List of features from each scale.
        Returns:
            Tensor: Output feature after hierarchical processing.
        """
        # Feature fusion and reduction
        x = self.feat_cat(x)
        x = self.fuse(x)

        # Hierarchical downsampling
        down2 = self.conv3x3_down0(self.avg_pool(x))
        down4 = self.conv3x3_down1(self.avg_pool(down2))
        down8 = self.conv3x3_down2(self.avg_pool(down4))
        down16 = self.conv3x3_down3(self.avg_pool(down8))

        # Hierarchical upsampling and features fusion
        up16 = self.resize_add(self.conv3x3_up0(down16), down8, self.skip_connnect)
        up8 = self.resize_add(self.conv3x3_up1(up16), down4, self.skip_connnect)
        up4 = self.resize_add(self.conv3x3_up2(up8), down2, self.skip_connnect)
        up2 = self.resize_add(self.conv3x3_up3(up4), x)
        return up2


class REP_Hourglass(nn.Module):
    """
    Wrapper class supporting conversion to inference mode.
    """
    def __init__(self, inplanes, inner_planes, use_se=False, skip_connnect=True, expand=True, block="REP"):
        super(REP_Hourglass, self).__init__()
        self.rep_hourglass = REP_Hourglass_IMPL(
            inplanes, inner_planes, use_se, 
            identity=False, skip_connnect=skip_connnect, expand=expand, block=block
        )

    def forward(self, x):
        return self.rep_hourglass(x)

    def convert_for_inference(self):
        """
        Convert model parameters for inference, such as reparameterization.
        """
        self.rep_hourglass = repvgg_model_convert(self.rep_hourglass)


class UP_Head(nn.Module):
    """
    Up-sampling head module for feature resolution enhancement and class prediction.
    """
    def __init__(self, ch, n_classes, up_sample_times=2, reduce_ch=True, act=None):
        super(UP_Head, self).__init__()
        cur_ch = ch
        out_ch = ch // 2

        # Initial convolution layer to reduce channels
        self.conv3x3 = conv3x3_block(cur_ch, out_ch)
        cur_ch = out_ch

        # Sequential upsampling layers
        up_sample_blocks = []
        for i in range(up_sample_times):
            # Optionally reduce channels at each step
            if reduce_ch:
                out_ch = cur_ch // (2 ** (i + 1))
            up_block = nn.Sequential(
                nn.Upsample(scale_factor=2.0, mode='nearest'),
                conv3x3_block(cur_ch, out_ch)
            )
            up_sample_blocks.append(up_block)
            cur_ch = out_ch

        self.up_sample_blocks = nn.Sequential(*up_sample_blocks)

        # Output classification layer
        self.head = nn.Conv2d(cur_ch, n_classes, kernel_size=1)

        # Activation functions dictionary
        act_dict = {
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
        }

        # Choose activation function
        if act is not None:
            self.act = act_dict[act]
        else:
            self.act = nn.Identity()

    def forward(self, x):
        x = self.conv3x3(x)
        x = self.up_sample_blocks(x)
        x = self.head(x)
        x = self.act(x)
        return x

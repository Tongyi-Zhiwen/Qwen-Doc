import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

# Conv2d + BatchNorm2d
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    seq = nn.Sequential()
    seq.add_module(
        'conv',
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
    )
    seq.add_module('bn', nn.BatchNorm2d(out_channels))
    return seq

# Conv2d only
def conv(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    seq = nn.Sequential()
    seq.add_module(
        'conv', 
        nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, padding, 
            groups=groups, 
            bias=True
        )
    )
    return seq

class RepVGGBlock(nn.Module):
    """
    Standard RepVGG block: supports training (with BN branches) and deploy (fused single conv)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super().__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        self.nonlinearity = nn.ReLU(inplace=True)

        self.se = nn.Identity()

        if deploy:
            # In inference: use fused single conv
            self.rbr_reparam = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                         padding, dilation, groups, bias=True, padding_mode=padding_mode)
        else:
            # In training: combine identity BN, 3x3 conv BN, 1x1 conv BN
            self.rbr_identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels, out_channels, 3, stride, 1, groups)
            self.rbr_1x1 = conv_bn(in_channels, out_channels, 1, stride, 0, groups)

    def forward(self, x):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(x)))
       
        id_out = self.rbr_identity(x) if self.rbr_identity is not None else 0
        return self.nonlinearity(self.se(self.rbr_dense(x) + self.rbr_1x1(x) + id_out))

    # For deployment: fuse kernels/bias from conv/bn/identity branches
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            # Conv-BN branch
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            # BN-only (identity branch)
            assert isinstance(branch, nn.BatchNorm2d)
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
            self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

class QARepVGGBlock(RepVGGBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros',
                 deploy=False, use_se=False, identity=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                         groups, padding_mode, deploy, use_se)
        if not deploy:
            self.bn = nn.BatchNorm2d(out_channels)
            self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=False)
            self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 and identity else None
        self._id_tensor = None

    def forward(self, x):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.bn_rep(self.se(self.rbr_reparam(x))))
        id_out = self.rbr_identity(x) if self.rbr_identity is not None else 0
        return self.nonlinearity(self.bn(self.se(self.rbr_dense(x) + self.rbr_1x1(x) + id_out)))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(self.rbr_1x1.weight)
        bias = bias3x3
        if self.rbr_identity is not None:
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
            id_tensor = torch.from_numpy(kernel_value).to(self.rbr_1x1.weight.device)
            kernel += id_tensor
        return kernel, bias

    def _fuse_extra_bn_tensor(self, kernel, bias, branch):
        # Fuse a post-conv BN
        assert isinstance(branch, nn.BatchNorm2d)
        running_mean = branch.running_mean - bias  # Remove previous bias
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.bn_rep = copy.deepcopy(self.bn)
        for para in self.parameters():
            para.detach_()
        del self.rbr_dense
        del self.rbr_1x1
        if hasattr(self, 'rbr_identity'):
            del self.rbr_identity
        if hasattr(self, 'id_tensor'):
            del self.id_tensor
        if hasattr(self, 'alpha'):
            del self.alpha
        if hasattr(self, 'bn'):
            del self.bn
        self.deploy = True

class RepVGG(nn.Module):
    """
    RepVGG backbone
    """
    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None,
                 deploy=False, use_se=False, deep_steam=False, early_feat=False, img_fpn=False):
        super().__init__()
        self.deploy = deploy
        self.override_groups_map = override_groups_map or {}
        self.use_se = use_se
        self.early_feat = early_feat
        self.img_fpn = img_fpn
        assert 0 not in self.override_groups_map
        # Initial filters
        self.in_planes = min(64, int(64 * width_multiplier[0]))
        if deep_steam:
            # Deep steam: use 3 blocks for stage0
            self.stage0 = nn.Sequential(
                QARepVGGBlock(3, self.in_planes // 2, 3, 2, 1, deploy=deploy, use_se=use_se),
                QARepVGGBlock(self.in_planes // 2, self.in_planes // 2, 3, 1, 1, deploy=deploy, use_se=use_se),
                QARepVGGBlock(self.in_planes // 2, self.in_planes, 3, 1, 1, deploy=deploy, use_se=use_se)
            )
        else:
            # Standard first block
            self.stage0 = QARepVGGBlock(3, self.in_planes, 3, 2, 1, deploy=deploy, use_se=use_se)
        self.cur_layer_idx = 1
        # Stage blocks
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2, use_se=use_se)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2, use_se=use_se)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2, use_se=use_se)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2, use_se=use_se)
        # Add image fusion layers if enabled
        if self.img_fpn:
            self.stage2_img_conv = nn.Sequential(
                nn.Upsample(scale_factor=0.25),
                QARepVGGBlock(3, int(64 * width_multiplier[0]), 3, 1, 1, deploy=deploy, use_se=use_se)
            )
            self.stage3_img_conv = nn.Sequential(
                nn.Upsample(scale_factor=0.125),
                QARepVGGBlock(3, int(128 * width_multiplier[1]), 3, 1, 1, deploy=deploy, use_se=use_se)
            )
            self.stage4_img_conv = nn.Sequential(
                nn.Upsample(scale_factor=0.0625),
                QARepVGGBlock(3, int(256 * width_multiplier[2]), 3, 1, 1, deploy=deploy, use_se=use_se)
            )

    def _make_stage(self, planes, num_blocks, stride, use_se):
        # Build a stage as a sequence of blocks
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(QARepVGGBlock(self.in_planes, planes, 3, s, 1, cur_groups, deploy=self.deploy, use_se=use_se))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        c1 = out
        out = self.stage1(out)
        if self.img_fpn:
            out = out + self.stage2_img_conv(x)
        c2 = out
        out = self.stage2(out)
        if self.img_fpn:
            out = out + self.stage3_img_conv(x)
        c3 = out
        out = self.stage3(out)
        if self.img_fpn:
            out = out + self.stage4_img_conv(x)
        c4 = out
        out = self.stage4(out)
        c5 = out
        # Return early features if enabled
        outputs = [c1, c2, c3, c4, c5] if self.early_feat else [c2, c3, c4, c5]
        return outputs

def create_RepVGG_B0_SLIM(deploy=False, use_se=False, deep_steam=False, early_feat=False):
    """
    Factory function for RepVGG-B0-slim
    """
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[0.75, 0.75, 0.75, 0.75],
        override_groups_map=None,
        deploy=deploy,
        use_se=use_se,
        deep_steam=deep_steam,
        early_feat=early_feat
    )

def repvgg_model_convert(model: nn.Module, save_path=None, do_copy=True):
    """
    Convert RepVGG model for deployment (by fusing BNs to convs)
    """
    model = copy.deepcopy(model) if do_copy else model
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model

class RepVGG_Impl(nn.Module):
    """
    RepVGG Wrapper: implements creation, deployment conversion, and forward method
    """
    def __init__(self, type_name, pretrained=False, weights_path=None, deploy=False,
                 use_se=False, deep_steam=False, early_feat=False):
        super().__init__()
        self.repvgg_impl = create_RepVGG_B0_SLIM(
            deploy=False, use_se=use_se, deep_steam=deep_steam, early_feat=early_feat)
        if deploy:
            self.convert_for_inference()

    def forward(self, x):
        return self.repvgg_impl(x)

    def convert_for_inference(self):
        self.repvgg_impl = repvgg_model_convert(self.repvgg_impl)

class QARepVGG_B0_SLIM(RepVGG_Impl):
    """
    Entry point for registry: QARepVGG_B0_SLIM
    """
    def __init__(self, pretrained=False, weights_path=None, deploy=False,
                 use_se=False, deep_steam=False, early_feat=False):
        super().__init__("RepVGG-B0-slim", pretrained, weights_path, deploy,
                         use_se, deep_steam, early_feat)


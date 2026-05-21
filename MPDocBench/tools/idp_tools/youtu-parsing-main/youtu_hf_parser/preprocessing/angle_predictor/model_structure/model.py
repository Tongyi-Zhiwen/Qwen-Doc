import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import QARepVGG_B0_SLIM
from .neck import REP_FPN
from .head import REP_Hourglass, UP_Head

class PTD(nn.Module):
    def __init__(self, backbone, neck=None, feats_fuse=None, heads=None, criterion=None,
                 out_stride=2, **kwargs):
        super(PTD, self).__init__()

        # Initialize backbone, neck, and feature fusion modules
        # self.backbone = QARepVGG_B0_SLIM(**backbone.params)
        # self.neck = REP_FPN(**neck.params)
        # self.feats_fuse = REP_Hourglass(**feats_fuse.params)

        self.backbone = QARepVGG_B0_SLIM(**backbone['params'])
        self.neck = REP_FPN(**neck['params'])
        self.feats_fuse = REP_Hourglass(**feats_fuse['params'])

        # Initialize multiple head modules
        # self.heads = nn.ModuleList([UP_Head(**head.params) for head in heads])
        # self.head_names = [head.name for head in heads]

        self.heads = nn.ModuleList([UP_Head(**head['params']) for head in heads])
        self.head_names = [head['name'] for head in heads]

        # Settings
        self.trace_head_names = kwargs.get("trace_head_names")
        self.out_stride = out_stride
        self.trace_post_process = kwargs.get("trace_post_process")
        self.trace_channel_last = kwargs.get("trace_channel_last")
        self.test = False

    def set_test(self):
        """Switch to inference mode."""
        self.test = True
        # Use inference conversion if available
        for m in [self.backbone, self.neck, self.feats_fuse] + list(self.heads):
            if hasattr(m, "covert_for_inference"):
                m.covert_for_inference()

    def resize_concat(self, x, y):
        """Upsample x to y's size and concatenate along channel dimension."""
        _, _, H, W = y.shape
        return torch.cat([F.upsample(x, size=[H, W], mode="nearest"), y], dim=1)

    def resize_scale(self, x, scale):
        """Upsample x by a given scale factor."""
        return F.upsample(x, scale_factor=scale, mode="nearest")

    def resize_add(self, x, y):
        """Upsample x to y's size and add element-wise."""
        _, _, H, W = y.shape
        return F.upsample(x, size=[H, W], mode="nearest") + y

    def feats_concate(self, feats):
        """Concatenate feature maps after progressive upsampling."""
        out = feats[-1]
        for i in range(len(feats) - 1):
            out = self.resize_concat(feats[i], out)
        return out

    def forward_main(self, x):
        """Core forward pass: feature extraction and fusion."""
        features = self.backbone(x)
        features = self.neck(features)
        out_feat = self.feats_fuse(features)
        return out_feat

    def forward_test(self, x):
        """Inference forward pass."""
        if self.trace_channel_last:
            x = x.contiguous(memory_format=torch.channels_last)
        out_feat = self.forward_main(x)

        outs = {}
        # Only compute specified heads in trace_head_names
        for i, name in enumerate(self.head_names):
            if self.trace_head_names and name in self.trace_head_names:
                outs[name] = self.heads[i](out_feat)

        seg_out = outs.get("kernels")
        if seg_out is None:
            raise ValueError("Missing 'kernels' output in head.")

        # Predict angle vector; use first two channels as default
        vec_out = outs.get("angle_vec", seg_out[:, 0:2, :, :])
        # Compute score map
        score = torch.sigmoid(seg_out[:, 0:1, :, :])

        if self.trace_post_process is not False:
            # Threshold segmentation output
            seg_thresh = torch.sign(seg_out)
            seg_thresh = F.threshold(seg_thresh, 0, 0, inplace=True)
            text_mask = seg_thresh[:, 0, :, :].bool()
            seg_thresh.masked_fill_(~text_mask, 0)
            # Concatenate thresholded seg, angle vec, and score
            out = torch.cat([seg_thresh, vec_out, score], dim=1)
        else:
            out = torch.cat([seg_out, vec_out, score], dim=1)

        if self.trace_channel_last:
            out = out.contiguous(memory_format=torch.contiguous_format)

        return out

    def forward(self, x):
        return self.forward_test(x)

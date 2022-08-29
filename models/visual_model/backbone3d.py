# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
import torch
from torch import nn
from typing import Dict, List

from utils.misc import NestedTensor
from mmcv.runner import auto_fp16
from mmdet3d.models.builder import build_backbone
import importlib



class Joiner3D(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, nest: NestedTensor):
        points = nest.tensors
        ret = self.backbone(points)

        fp_xyz = ret['fp_xyz'][0]
        fp_features = ret['fp_features'][0]
        return fp_xyz, fp_features


def build_backbone3d(args):
    config_lib = importlib.import_module(args.backbone_config)
    config = config_lib.config
    backbone = build_backbone(config)
    model = Joiner3D(backbone)
    # model.num_channels = backbone.num_channels
    model.num_channels = 512
    model.npoints = 512
    return model

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize

import datasets.transforms as T
from .sunrefer2d import SUNREFER2D
from .sunrefer3d import SUNREFER3D

def make_transforms(args, split):
    imsize = args.imsize
    return T.Compose([
        T.RandomResize([imsize]),
        T.ToTensor(),
        T.NormalizeAndPad(size=imsize),
    ])


def build_dataset(split, args):
    if args.dataset ==  'sunrefer2d':
        return SUNREFER2D(
                        split=split,
                        transform=make_transforms(args, split),
                        max_query_len=args.max_query_len)
    if args.dataset ==  'sunrefer3d':
        return SUNREFER3D(
                        split=split,
                        max_query_len=args.max_query_len,
                        vrange=args.vrange,
                        use_rgb=args.use_rgb)
    else:
        raise ValueError(f"No dataset {args.dataset} ")

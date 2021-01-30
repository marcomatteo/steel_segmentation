# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02_dataset.ipynb (unless otherwise specified).

__all__ = ['get_classification_dls', 'classes', 'get_segmentation_dls', 'get_segmentation_dls_from_df']

# Cell
from .core import *
from .data import *

from fastcore.test import *
from fastai.vision.all import *
from PIL import Image
from collections import defaultdict
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np

# Cell
def get_classification_dls(bs, with_tfms: bool = True, size=None):
    """
    Dataloaders from train DataFrame
    """
    b_tfms = [Normalize.from_stats(*imagenet_stats)]

    if with_tfms:
        b_tfms += aug_transforms(
            size=size if size else (256, 1600),
            max_warp=0.,
            flip_vert=True,
            max_rotate=5.,
            max_lighting=0.1)

    dblock = DataBlock(
        blocks=(ImageBlock, MultiCategoryBlock()),
        get_x=ColReader(0, pref=train_path),
        get_y=ColReader(1, label_delim=' '),
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        batch_tfms=b_tfms)

    return dblock.dataloaders(train_multi, bs=bs)

# Cell
classes = [0, 1, 2, 3, 4]

# Cell
def get_segmentation_dls(bs, size, with_btfms=True):
    """Dataloaders from `train_path` folder"""

    b_tfms = [Normalize.from_stats(*imagenet_stats)]

    if with_btfms:
        b_tfms += aug_transforms(
            size=size if size else (256, 1600),
            max_warp=0.,
            flip_vert=True,
            max_rotate=5.,
            max_lighting=0.1)

    def get_labels_from_img(p):
        return labels_path/f'{p.stem}_P.png'

    dblock = DataBlock(
        blocks=(ImageBlock, MaskBlock(codes=classes)),
        get_items=get_image_files,
        get_y=get_labels_from_img,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        batch_tfms=b_tfms)

    return dblock.dataloaders(train_path, bs=bs)

# Cell
def get_segmentation_dls_from_df(train_df, bs, size):
    """Dataloaders from `train` DataFrame"""
    def get_x(df):
        img_name = df["ImageId"]
        return train_path / str(img_name)

    def get_y(df):
        img_name = df["ImageId"].split(".")[0] + "_P.png"
        return labels_path / img_name

    dblock = DataBlock(
        blocks=(ImageBlock, MaskBlock(codes=classes)),
        get_x=get_x,
        get_y=get_y,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        batch_tfms=aug_transforms(size=size))

    return dblock.dataloaders(train_df, bs=bs)
from albumentations.augmentations.geometric.resize import Resize
from fastai.vision.data import ImageBlock
from fastai.vision.core import ToTensor, imagenet_stats
from fastai.data.transforms import IntToFloatTensor, Normalize
from fastai.data.block import DataBlock
from transforms import (
    ReadImagePathFromIndex, ReadRLEs, 
    AlbumentationsTransform, ChannelMask,
    SteelMaskBlock
)
import albumentations as alb

HEIGHT, WIDTH = (224, 1568)

def get_train_aug(): 
    tfm_list = [
      alb.RandomCrop(HEIGHT, WIDTH, p=1.0),
      #alb.Resize(128, 800),
      alb.OneOf(
          [
           alb.VerticalFlip(p=0.5),
           alb.HorizontalFlip(p=0.5),
          ], p=0.6),
      alb.OneOf(
          [
           alb.RandomBrightnessContrast(
               brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
           alb.RandomGridShuffle(grid=(1, 2), p=0.2),
          ], p=0.6),
    ]
    return alb.Compose(tfm_list)

def get_valid_aug(): 
    tfms = [alb.RandomCrop(HEIGHT, WIDTH)]
    return alb.Compose(tfms)

def SteelDataBlock(path, splitter, flatten_mask=False, *args, **kwargs) -> DataBlock:
    """Get the DataBlock for Severstal Dataset.

    Parameters
    ----------
    path : pathlib.Path
        [description]
    splitter : [type]
        [description]
    flatten_mask : bool, optional
        [description], by default False

    Returns
    -------
    DataBlock
        [description]
    """
    get_x = ReadImagePathFromIndex(pref=(path/"train_images"))
    get_y = ReadRLEs()

    train_aug = get_train_aug()
    valid_aug = get_valid_aug()

    item_tfms = [ToTensor(), AlbumentationsTransform(train_aug, valid_aug)]
    batch_tfms=[IntToFloatTensor(), Normalize.from_stats(*imagenet_stats)]
    if not flatten_mask: batch_tfms += [ChannelMask()]

    block = DataBlock(
        blocks = (ImageBlock, SteelMaskBlock),
        get_x = get_x,
        get_y = get_y,
        splitter = splitter,
        item_tfms = item_tfms,
        batch_tfms = batch_tfms,
        *args, **kwargs
    )
    return block
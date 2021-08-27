# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %% [markdown]
# # Segmentation training Unet
# > Training notebook with Tensorboard logs.
# %% [markdown]
# ## Setup

# %%
from fastai.vision.all import *
from fastai.callback.tensorboard import TensorBoardCallback

import warnings
import random
import cv2
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

import torch
import albumentations as alb

import segmentation_models_pytorch as smp

from steel_segmentation.utils import get_train_df
from steel_segmentation.transforms import SteelDataBlock, SteelDataLoaders
from steel_segmentation.losses import MultiClassesSoftBCEDiceLoss, LossEnabler
from steel_segmentation.metrics import ModDiceMulti
from steel_segmentation.optimizer import opt_func


# %%
def seed_everything(seed=69):
    """
    Seeds `random`, `os.environ["PYTHONHASHSEED"]`,
    `numpy`, `torch.cuda` and `torch.backends`.
    """
    #warnings.filterwarnings("ignore")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()
warnings.filterwarnings("ignore")

# %% [markdown]
# Training parameters:

# %%
bs = 16
size = (224,512)
epochs = 30
lr = 3e-4
path = Path("data") # where data dir is

# %% [markdown]
# ## Data loading

# %%
df = get_train_df(path, only_faulty=True, pivot=True)
df.describe(include="all")


# %%
def get_train_aug(height, width): 
    tfm_list = [
        alb.RandomCrop(height, width, p=1.0),
        alb.OneOf(
          [
           alb.VerticalFlip(p=0.5),
           alb.HorizontalFlip(p=0.5),
          ], p=0.5),
        alb.RandomBrightnessContrast(
            brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
    ]
    return alb.Compose(tfm_list)

def get_valid_aug(height, width): 
    tfms = [alb.RandomCrop(height, width, p=1.0)]
    return alb.Compose(tfms)

device = "cuda" if torch.cuda.is_available() else "cpu"
train_aug = get_train_aug(*size)
valid_aug = get_valid_aug(*size)
block = SteelDataBlock(path, train_aug=train_aug, valid_aug=valid_aug)
dls = SteelDataLoaders(block, df, bs=bs, device=device)


# %%
xb, yb = dls.one_batch()
print(xb.shape, yb.shape)

# %% [markdown]
# ## Model

# %%
model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", classes=4, activation=None)
criterion = BCEWithLogitsLossFlat(axis=1, pos_weight=torch.tensor([2.0,2.0,1.0,1.5])) # pos_weight because class imbalance
#opt_func = partial(opt_func, torch_opt=torch.optim.Adam) # no need to use pytorch optim
opt_func = RAdam
model_dir = Path("../models")
metrics = [ModDiceMulti(with_logits=True)]


# %%
learner = Learner(
    dls = dls,
    model = model,
    loss_func = criterion,
    opt_func = opt_func,
    metrics = metrics,
    model_dir = model_dir,
    cbs = [LossEnabler]
)


# %%
learner.summary()


# %%
learner.show_training_loop()

# %% [markdown]
# Logging with the TensorBoardCallback:

# %%
# logging info
log_dir = Path("../logs") / f"unet_resnet_bce_epochs{epochs}_lr{lr}"
log_dir


# %%
train_cbs = [
    TensorBoardCallback(log_dir=log_dir, log_preds=True, trace_model=True, projector=False),
    GradientAccumulation(n_acc=24),
    SaveModelCallback(monitor="valid_loss", fname=log_dir.name, with_opt=True),
    ReduceLROnPlateau(monitor='valid_loss', min_delta=0.15, patience=4),
]


# %%
learner.fit(epochs, lr=lr, cbs=train_cbs)

# %% [markdown]
# 


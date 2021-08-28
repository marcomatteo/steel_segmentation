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
from steel_segmentation.transforms import SteelDataBlock, SteelDataLoaders, get_kfold_splits, KFoldSplitter
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
fine_tuning = True # Set to False if first training
# epochs = 30
epochs = 20 # fine tuning
# lr = 3e-4
lr = 1e-4 # fine tuning
toload = "unet_resnet_bce_epochs30_lr0.0003" # fine tuning
encoder_name = "resnet18"

path = Path(".") 
train_path = path / "data" # where data dir is
model_dir = path / "models"

model = smp.Unet(encoder_name=encoder_name, encoder_weights="imagenet", classes=4, activation=None)
# model.load_state_dict(torch.load(model_dir / toload)["model"]) # fine tuning
# criterion = BCEWithLogitsLossFlat(axis=1, pos_weight=torch.tensor([2.0,2.0,1.0,1.5])) # pos_weight because class imbalance
criterion = MultiClassesSoftBCEDiceLoss(bce_pos_weights=torch.tensor([2.0,2.0,1.0,1.5])) # fine tuning
opt_func = RAdam
metrics = [ModDiceMulti(with_logits=True)]

# %% [markdown]
# ## Data loading

# %%
# df = get_train_df(train_path, only_faulty=True, pivot=True)
df = get_train_df(train_path, only_faulty=False, pivot=True)

# %%
# Stratified K-fold for finetuning
splits = get_kfold_splits(df, nsplits=2)

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
valid_aug = get_valid_aug(height=256, width=1600) # validation on full size

# %%
if fine_tuning:
    dataloaders = []
    for idx, _ in enumerate(splits):
        splitter = KFoldSplitter(splits, idx)
        block = SteelDataBlock(train_path, train_aug=train_aug, valid_aug=valid_aug, splitter=splitter)
        dls = SteelDataLoaders(block, df, bs=bs, device=device)
        dls.valid.bs = bs // 2 # full size is more heavy on GPU memory
        print(f"Training elements: {dls.train.items.shape}")
        print(f"Validation elements: {dls.valid.items.shape}")
        xb, yb = dls.train.one_batch()
        print(f"Training batch shapes (x,y): {xb.shape}, {yb.shape}")
        xb, yb = dls.valid.one_batch()
        print(f"Validation batch shapes (x,y): {xb.shape}, {yb.shape}")
        dataloaders.append(dls)
else:
    block = SteelDataBlock(train_path, train_aug=train_aug, valid_aug=valid_aug)
    dls = SteelDataLoaders(block, df, bs=bs, device=device)
    dls.valid.bs = bs // 2 # full size is more heavy on GPU memory
    print(f"Training elements: {dls.train.items.shape}")
    print(f"Validation elements: {dls.valid.items.shape}")
    xb, yb = dls.train.one_batch()
    print(f"Training batch shapes (x,y): {xb.shape}, {yb.shape}")
    xb, yb = dls.valid.one_batch()
    print(f"Validation batch shapes (x,y): {xb.shape}, {yb.shape}")

# %% [markdown]
# ## Model
if fine_tuning:
    for idx, _ in enumerate(splits):
        dls = dataloaders[idx]
        learner = Learner(
            dls = dls,
            model = model,
            loss_func = criterion,
            opt_func = opt_func,
            metrics = metrics,
            model_dir = model_dir,
            cbs = [LossEnabler]
        )
        learner.load(toload)
        print(learner.summary()) # debug
        learner.show_training_loop() # debug
        log_dir = path / "logs" / f"unet_{encoder_name}-bce_dice-epochs_{epochs}-lr_{lr}-{idx}_fold"
        train_cbs = [
            TensorBoardCallback(log_dir=log_dir, log_preds=True, trace_model=True, projector=False),
            GradientAccumulation(n_acc=24),
            SaveModelCallback(monitor="valid_loss", fname=log_dir.name, with_opt=True),
            ReduceLROnPlateau(monitor='valid_loss', patience=4),
        ]
        learner.fit(epochs, lr=lr, cbs=train_cbs)
else:
   
    learner = Learner(
        dls = dls,
        model = model,
        loss_func = criterion,
        opt_func = opt_func,
        metrics = metrics,
        model_dir = model_dir,
        cbs = [LossEnabler]
    )
    print(learner.summary()) # debug
    learner.show_training_loop() # debug
    log_dir = path / "logs" / f"unet_resnet_bce_epochs{epochs}_lr{lr}"

    train_cbs = [
        TensorBoardCallback(log_dir=log_dir, log_preds=True, trace_model=True, projector=False),
        GradientAccumulation(n_acc=24),
        SaveModelCallback(monitor="valid_loss", fname=log_dir.name, with_opt=True),
        ReduceLROnPlateau(monitor='valid_loss', patience=4),
    ]

    learner.fit(epochs, lr=lr, cbs=train_cbs)

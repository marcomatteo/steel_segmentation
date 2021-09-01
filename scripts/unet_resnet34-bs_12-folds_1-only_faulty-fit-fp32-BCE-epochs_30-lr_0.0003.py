
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %% [markdown]
# # Segmentation training

# %% Load libraries
from fastai.vision.all import *
from fastai.callback.tensorboard import TensorBoardCallback

import torch
import albumentations as alb

import segmentation_models_pytorch as smp

from steel_segmentation.utils import get_train_df, seed_everything
from steel_segmentation.transforms import SteelDataBlock, SteelDataLoaders, get_kfold_splits, KFoldSplitter
from steel_segmentation.losses import MultiClassesSoftBCEDiceLoss, LossEnabler
from steel_segmentation.metrics import ModDiceMulti
from steel_segmentation.optimizer import opt_func

seed_everything()

# %% Augmentations
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

# %% Parameters
bs = 12
size = (224,512)
kfolds = 1
fine_tuning_weights = None # Set to None if first training else the .pth model weights
arch = "unet" # unet or fpn
encoder_name = "resnet34" # "efficientnet-b3"
loss = "BCE" # bce or dice (MultiClassesSoftBCEDiceLoss)
epochs = 30
lr = 3e-4
only_faulty = True # only defected images or all the train_images
one_cycle = False # fit_one_cycle if True, fit otherwise
mixed_prec = False # fp16() learner

# %% Load configuration
path = Path(".") # "." if script, "../" if jupyter nb
train_path = path / "data" # where data dir is
model_dir = path / "models"
log_dir = path / "logs" 
tensorboard_run_name = f"{arch}_{encoder_name}-" \
    + f"size_({size[0]},{size[1]})-" \
    + f"bs_{bs}-folds_{kfolds}-" + ("only_faulty" if only_faulty else "all_imgs") \
    + "-" + ("fine_tuning-" if not (fine_tuning_weights is None) else "") \
    + ("fit_one_cycle" if one_cycle else "fit") + "-" \
    + ("fp16" if mixed_prec else "fp32") + "-" \
    + f"{loss}-epochs_{epochs}-lr_{lr}"

df = get_train_df(train_path, only_faulty=only_faulty, pivot=True)

train_aug = get_train_aug(*size)
valid_aug = get_valid_aug(height=256, width=1600) # validation on full size

device = "cuda" if torch.cuda.is_available() else "cpu"

if arch == "unet":
    model = smp.Unet(encoder_name=encoder_name, encoder_weights="imagenet", classes=4, activation=None)
elif arch == "fpn":
    model = smp.FPN(encoder_name=encoder_name, encoder_weights="imagenet", classes=4, activation=None)

# pos_weight because class imbalance
if loss == "BCE": 
    criterion = BCEWithLogitsLossFlat(axis=1, pos_weight=torch.tensor([2.0,2.0,1.0,1.5])) 
else:
    criterion = MultiClassesSoftBCEDiceLoss(bce_pos_weights=torch.tensor([2.0,2.0,1.0,1.5]))
opt_func = RAdam
metrics = [ModDiceMulti(with_logits=True)]

# %% Debug dls
def print_dls(dls: DataLoaders):
    print(f"Training elements: {dls.train.items.shape}")
    print(f"Validation elements: {dls.valid.items.shape}")
    xb, yb = dls.train.one_batch()
    print(f"Training batch shapes (x,y): {xb.shape}, {yb.shape}")
    xb, yb = dls.valid.one_batch()
    print(f"Validation batch shapes (x,y): {xb.shape}, {yb.shape}")

def get_dls(splitter=None):
    block = SteelDataBlock(train_path, train_aug=train_aug, valid_aug=valid_aug, splitter=splitter)
    dls = SteelDataLoaders(block, df, bs=bs, device=device)
    dls.valid.bs = bs // 2 # full size is more heavy on GPU memory
    print_dls(dls)
    return dls

# %% Train func
def train(tensorboard_log:Path, splitter=None):
    print(f"\nTraining {tensorboard_log.name}\n")
    dls = get_dls(splitter)
    learner = Learner(
        dls = dls,
        model = model,
        loss_func = criterion,
        opt_func = opt_func,
        metrics = metrics,
        model_dir = model_dir,
        cbs = [LossEnabler]
    )
    if mixed_prec:
        learner = learner.to_fp16()
    if (not (fine_tuning_weights is None)) and isinstance(fine_tuning_weights, str):
        assert (model_dir / (fine_tuning_weights + ".pth")).is_file()
        learner.load(fine_tuning_weights)
    print(learner.summary()) # debug
    learner.show_training_loop() # debug
    train_cbs = [
        TensorBoardCallback(log_dir=tensorboard_log, log_preds=True, 
                            trace_model=(False if mixed_prec else True), projector=False),
        GradientAccumulation(n_acc=24),
        SaveModelCallback(monitor="valid_loss", fname=tensorboard_log.name, with_opt=True),
        ReduceLROnPlateau(monitor='valid_loss', patience=3),
    ]
    if one_cycle:
        learner.fit_one_cycle(epochs, lr_max=lr, cbs=train_cbs)
    else:
        learner.fit(epochs, lr=lr, cbs=train_cbs)

# %% Main
def main():
    if kfolds > 1:
        # Stratified K-fold strategy
        splits = get_kfold_splits(df, nsplits=kfolds)
        for idx, _ in enumerate(splits):
            splitter = KFoldSplitter(splits, idx)
            tensorboard_log = log_dir / (tensorboard_run_name + f"-{idx}_fold")
            train(tensorboard_log, splitter=splitter)
            torch.cuda.empty_cache()
    elif kfolds == 1:
        tensorboard_log = log_dir / tensorboard_run_name
        train(tensorboard_log)

if __name__ == "__main__":
    main()

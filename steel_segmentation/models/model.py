# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/06_models.model.ipynb (unless otherwise specified).

__all__ = ['models_dir', 'class_metrics', 'seg_metrics', 'seed', 'Trainer']

# Cell
from ..core import *
from ..data import *
from ..preprocessing import *
from .dls import *
from .metrics import *
from .unet import Unet

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from fastai.vision.all import *
    import fastai
from fastcore.foundation import *

# Cell
models_dir = path.parent / "models"

# Cell
class_metrics = [accuracy_multi, PrecisionMulti(), RecallMulti()]

# Cell
seg_metrics = [ModDiceMulti(), dice_kaggle]

# Cell
import os
import pdb
import time
import warnings
import random
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

warnings.filterwarnings("ignore")
seed = 69
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# Cell
class Trainer:
    '''This class takes care of training and validation of our model'''

    def __init__(self, model, save_path,
                 num_epochs=20, lr=5e-4,
                 bs=16, num_workers=6):
        self.num_workers = num_workers
        self.batch_size = {"train": bs, "val": bs//2}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = lr
        self.num_epochs = num_epochs
        self.net = model
        self.save_path = save_path

        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True)
        self.net = self.net.to(self.device)
        cudnn.benchmark = True

        self.dataloaders = {
            phase: get_kaggle_train_dls(
                data_folder=path,
                phase=phase,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }

        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}

    def forward(self, images, targets):
        """
        Forward pass:
            load to GPU the imgs and masks,
            calculate predictions,
            calculate loss

        Returns:
            loss and predictions
        """
        images = images.to(self.device)
        masks = targets.to(self.device)
        preds = self.net(images)
        loss = self.loss_fn(preds, masks)
        return loss, preds

    def iterate(self, epoch, phase):
        """
        Iterate throught each batch in training or validatio phase.
        """
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")

        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]

        running_loss = 0.0
        total_batches = len(dataloader)
#         tk0 = tqdm(dataloader, total=total_batches)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader): # replace `dataloader` with `tk0` for tqdm
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1 ) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)
#             tk0.set_postfix(loss=(running_loss / ((itr + 1))))

        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou = epoch_log(phase, epoch, epoch_loss, meter, start)

        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)

        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        """
        Training loop for each epochs.
        """
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            with torch.no_grad():
                val_loss = self.iterate(epoch, "val")
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, self.save_path)
            print()
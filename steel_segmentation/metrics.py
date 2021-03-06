# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/05_metrics.ipynb (unless otherwise specified).

__all__ = ['ModDiceMulti', 'KaggleDice', 'single_dice_coef', 'single_dice_coef_channel', 'KaggleDiceCoefMulti',
           'FastKaggleCoefDiceMulti', 'KaggleDiceCoef', 'FastKaggleDiceCoef']

# Cell
from .metadata import *
from .masks import *
from .datasets import *
from .dataloaders import *

import fastai
from fastai.vision.all import *
from fastcore.foundation import *

import torch
import torch.nn.functional as F

from collections import defaultdict

import segmentation_models_pytorch as smp

# Cell
class ModDiceMulti(Metric):
    "Averaged Dice metric (Macro F1) for multiclass target in segmentation"

    def __init__(self, axis=1, with_logits=False):
        self.axis = axis
        self.with_logits = with_logits

    def reset(self): self.inter, self.union =  {}, {}

    def accumulate(self, learn):
        if self.with_logits:
            logit = learn.pred
            prob = torch.sigmoid(logit)
            pred = (prob > 0.5).float().argmax(dim=self.axis)
        else:
            pred = learn.pred.argmax(dim=self.axis)

        y = learn.yb[0]
        # Added to deal with 4-channels masks
        if pred.shape != y.shape:
            y = y.argmax(dim=self.axis)

        pred, targ = flatten_check(pred, y)
        for c in range(learn.pred.shape[self.axis]):
            p = torch.where(pred == c, 1, 0)
            t = torch.where(targ == c, 1, 0)
            p, t = TensorBase(p), TensorBase(t) # may be redundant (old fastai bug)
            c_inter = (p*t).float().sum().item()
            c_union = (p+t).float().sum().item()
            if c in self.inter:
                self.inter[c] += c_inter
                self.union[c] += c_union
            else:
                self.inter[c] = c_inter
                self.union[c] = c_union

    @property
    def value(self):
        binary_dice_scores = np.array([])
        for c in self.inter:
            binary_dice_scores = np.append(
                binary_dice_scores,
                2.*self.inter[c]/self.union[c] if self.union[c] > 0 else np.nan)
        self.binary_dice_scores = binary_dice_scores
        return np.nanmean(binary_dice_scores)

# Cell
class KaggleDice(Metric):
    """
    Multi-class Dice used in Severstal comp,
    is 1 when prediction and mask are empty
    """
    def __init__(self, axis=1, with_logits=False, eps=1e-9):
        self.axis = axis
        self.eps = eps
        self.with_logits = with_logits

    def reset(self): self.inter, self.union = defaultdict(list), defaultdict(list)

    def accumulate(self, learn):
        if self.with_logits:
            logit = learn.pred
            prob = torch.sigmoid(logit)
            pred = (prob > 0.5).float().argmax(dim=self.axis)
        else:
            pred = learn.pred.argmax(dim=self.axis)

        y = learn.yb[0]
        if pred.shape != y.shape:
            y = y.argmax(dim=self.axis)

        n, c = y.shape[0], pred.shape[self.axis]

        preds, targs = flatten_check(pred, y)
        for i in range(0, c):
            p = torch.where(preds == i, 1, 0)
            t = torch.where(targs == i, 1, 0)

            p, t = TensorBase(p), TensorBase(t)

            c_inter = (p*t).sum(-1).float()#.item()
            c_union = (p+t).sum(-1).float()#.item()

            self.inter[i].append(c_inter)
            self.union[i].append(c_union)

    @property
    def value(self):
        binary_dice_scores = np.array([])
        for c in range(len(self.inter)):
            inter = torch.stack(self.inter[c])
            union = torch.stack(self.union[c])

            val = 2.*(inter+self.eps)/(union+self.eps)
            cond = union == 0
            val[cond] = 1

            binary_dice_scores = np.append(binary_dice_scores, val.cpu().numpy())

        self.binary_dice_scores = binary_dice_scores
        return np.nanmean(binary_dice_scores)
        #return (binary_dice_scores).reshape(-1, 4).mean(0).mean()

# Cell
def single_dice_coef(y_true, y_pred, smooth=1):
    """Binary segmentation function."""
    y_true_f = np.ndarray.flatten(y_true)
    y_pred_f = np.ndarray.flatten(y_pred)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def single_dice_coef_channel(y_true, y_pred, smooth=1):
    """Multichannel segmentation function."""
    ch1 = single_dice_coef(y_true[:,0,:,:], y_pred[:,0,:,:],smooth)
    ch2 = single_dice_coef(y_true[:,1,:,:], y_pred[:,1,:,:],smooth)
    ch3 = single_dice_coef(y_true[:,2,:,:], y_pred[:,2,:,:],smooth)
    ch4 = single_dice_coef(y_true[:,3,:,:], y_pred[:,3,:,:],smooth)
    res = (ch1+ch2+ch3+ch4)/4
    return res

# Cell
KaggleDiceCoefMulti = AccumMetric(single_dice_coef_channel, to_np=True, flatten=False, thresh=0.5)

# Cell
FastKaggleCoefDiceMulti = AccumMetric(single_dice_coef_channel, to_np=True, flatten=False)

# Cell
KaggleDiceCoef = AccumMetric(single_dice_coef, to_np=True, flatten=False, thresh=0.5)

# Cell
FastKaggleDiceCoef = AccumMetric(single_dice_coef, to_np=True, flatten=False)
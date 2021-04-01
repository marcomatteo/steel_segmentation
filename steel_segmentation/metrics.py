# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/05_metrics.ipynb (unless otherwise specified).

__all__ = ['ModDiceMulti', 'KaggleDice', 'dice_kaggle', 'fastai_dice_metrics', 'iou', 'iou_binary_metric', 'iou_metric',
           'compute_ious', 'compute_iou_batch', 'dice', 'metric', 'predict', 'Meter', 'epoch_log']

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

import segmentation_models_pytorch as smp

# Cell
class ModDiceMulti(Metric):
    "Averaged Dice metric (Macro F1) for multiclass target in segmentation"

    def __init__(self, axis=1): self.axis = axis
    def reset(self): self.inter, self.union = {}, {}

    def accumulate(self, learn):
        pred = learn.pred.argmax(dim=self.axis)
        y = learn.yb[0]

        if pred.shape != y.shape:
            y = y.argmax(dim=self.axis)

        pred, targ = flatten_check(pred, y)
        for c in range(learn.pred.shape[self.axis]):
            p = torch.where(pred == c, 1, 0)
            t = torch.where(targ == c, 1, 0)
            p, t = TensorBase(p), TensorBase(t)
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
                binary_dice_scores, 2.*self.inter[c]/self.union[c] if self.union[c] > 0 else np.nan)
        return np.nanmean(binary_dice_scores)

# Cell
class KaggleDice(Metric):
    def __init__(self, axis=1, eps=1e-9): self.axis, self.eps = axis, eps
    def reset(self): self.inter, self.union = {}, {}

    def accumulate(self, learn):
        y = learn.yb[0]
        preds = learn.pred

        if preds.shape == y.shape:
            y = y.argmax(dim=self.axis)

        n, c = y.shape[0], preds.shape[self.axis]

        preds = preds.argmax(dim=self.axis).view(n, -1)
        targs = y.view(n, -1)

#         pred, targ = flatten_check(preds, targs)
        for i in range(0, c):
            p = torch.where(preds == i, 1, 0)
            t = torch.where(targs == i, 1, 0)

            p, t = TensorBase(p), TensorBase(t)

            c_inter = (p*t).sum(-1).float()#.item()
            c_union = (p+t).sum(-1).float()#.item()
            if i in self.inter:
                self.inter[i] = torch.cat([self.inter[i], c_inter], dim=0)
                self.union[i] = torch.cat([self.union[i], c_union], dim=0)
            else:
                self.inter[i] = c_inter
                self.union[i] = c_union

#             print(f"Iter n.{i}\nintersect: {self.inter[i]}\nunion: {self.union[i]}", end="\n\n")

    @property
    def value(self):
        binary_dice_scores = np.array([])
        for c in range(len(self.inter)):
            cond = self.union[c] == 0
            val = 2.*(self.inter[c]+self.eps)/(self.union[c]+self.eps)
            val[cond] = 1
            binary_dice_scores = np.append(binary_dice_scores, val)
        return np.nanmean(binary_dice_scores)

# Cell
def dice_kaggle(
    preds: Tensor,
    targs: Tensor,
    iou: bool = False,
    eps: float = 1e-8
):
    """
    The metric of the competition,
    if there's no defect in `targs` and no defects in `preds`: dice=1.
    """
    n, c = targs.shape[0], preds.shape[1]
    preds = preds.argmax(dim=1).view(n, -1)
    targs = targs.view(n, -1)

    intersect_list, union_list = [], []
    for i in range(c):
        inp, trgs = TensorBase(preds), TensorBase(targs)

        inter = ((inp == i) & (trgs == i)).sum(-1).float()
        un = ((inp == i).sum(-1) + (trgs == i).sum(-1))

        intersect_list.append(inter)
        union_list.append(un)

    intersect = torch.stack(intersect_list)
    union = torch.stack(union_list)

    if not iou:
        dice = ((2.0 * intersect + eps) / (union + eps))
        return dice.mean()
    else:
        int_over_union = ((intersect + eps) / (union - intersect + eps)).mean()

# Cell
fastai_dice_metrics = [ModDiceMulti()]

# Cell
def iou(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
    threshold: float = None,
    activation: str = "Sigmoid"
):
    """
    https://github.com/catalyst-team/catalyst/blob/master/catalyst/dl/utils/criterion/iou.py
    Args:
        outputs (torch.Tensor): A list of predicted elements
        targets (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold (float): threshold for outputs binarization
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ["none", "Sigmoid", "Softmax2d"]
    Returns:
        float: IoU (Jaccard) score
    """
    outputs = F.sigmoid(outputs)

    if threshold is not None:
        outputs = (outputs > threshold).float()

    intersection = torch.sum(targets * outputs)
    union = torch.sum(targets) + torch.sum(outputs)
    iou = intersection / (union - intersection + eps)

    return iou

# Cell
def iou_binary_metric(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    iou = mean(ious)    # mean accross images if per_image
    return 100 * iou

# Cell
def iou_metric(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []
        for i in range(C):
            if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                label, pred = TensorBase(label), TensorBase(pred)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [np.mean(iou) for iou in zip(*ious)] # mean accross images if per_image
    return 100 * np.array(ious)

# Cell
def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    """computes IoU for one ground truth mask and predicted mask"""
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
        else:
            ious.append(1)
    return ious if ious else [0]

# Cell
def compute_iou_batch(outputs, labels, classes=None):
    """computes mean iou for a batch of ground truth masks and predicted masks"""
    ious = []
    outputs, labels = outputs.cpu(), labels.cpu()
    preds = np.copy(outputs) # copy is imp
    labels = np.array(labels) # tensor to np
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou

# Cell
def dice(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
    threshold: float = None
):
    """
    Computes the binary dice metric
    Args:
        outputs (list):  A list of predicted elements
        targets (list): A list of elements that are to be predicted
        eps (float): epsilon
        threshold (float): threshold for outputs binarization
    Returns:
        double:  Dice score
    """
    outputs = F.sigmoid(outputs)

    if threshold is not None:
        outputs = (outputs > threshold).float()

    intersection = torch.sum(targets * outputs)
    union = torch.sum(targets) + torch.sum(outputs)
    dice = 2 * intersection / (union + eps)

    return dice

# Cell
def metric(probability, truth, threshold=0.5, reduction='none'):
    """
    Calculates dice of positive and negative images seperately
    `probability` and `truth` must be `torch.Tensors`.
    """
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert(probability.shape == truth.shape)

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice, dice_neg, dice_pos, num_neg, num_pos

# Cell
def predict(X, threshold):
    """X is sigmoid output of the model"""
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds

# Cell
class Meter:
    """A meter to keep track of iou and dice scores throughout an epoch"""
    def __init__(self, phase, epoch):
        self.base_threshold = 0.5 # <<<<<<<<<<< here's the threshold
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice, dice_neg, dice_pos, _, _ = metric(probs, targets, self.base_threshold)

        self.base_dice_scores.extend(dice.tolist())
        self.dice_pos_scores.extend(dice_pos.tolist())
        self.dice_neg_scores.extend(dice_neg.tolist())

        preds = predict(probs, self.base_threshold)

        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

    def get_metrics(self):
        """
        Calc the mean of dices metrics (dice, dice_neg, dice_pos)
        and IoU mean.

        Returns:
            `dices` as list of means `[dice, dice_neg, dice_pos]`,
            `iou` as mean of IoUs
        """
        dice     = np.nanmean(self.base_dice_scores)
        dice_neg = np.nanmean(self.dice_neg_scores)
        dice_pos = np.nanmean(self.dice_pos_scores)

        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.iou_scores)

        return dices, iou

# Cell
def epoch_log(phase, epoch, epoch_loss, meter, start):
    """logging the metrics at the end of an epoch"""
    dices, iou = meter.get_metrics()
    dice, dice_neg, dice_pos = dices
    print(f"Loss: {epoch_loss:.4f} | IoU: {iou:.4f} | dice: {dice:.4f} | dice_neg: {dice_neg:.4f} | dice_pos: {dice_pos:.4f}")
    return dice, iou
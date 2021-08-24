__all__ = ['SoftDiceLoss', 'MultiClassesSoftBCEDiceLoss']

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.torch_core import TensorBase

#reference: https://github.com/asanakoy/kaggle_carvana_segmentation/blob/master/asanakoy/losses.py
class SoftDiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        probs = F.sigmoid(logits)
        num = labels.size(0)
        m1 = probs.view(num, -1)
        m2 = labels.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
        score = 1 - score.sum() / num
        return score

#reference: https://github.com/zdaiot/Kaggle-Steel-Defect-Detection
class WeightedSoftDiceLoss(nn.Module):

    def __init__(self, size_average=True, weight=[0.2, 0.8]):
        super().__init__()
        self.size_average = size_average
        self.weight = torch.FloatTensor(weight)

    def forward(self, logit_pixel, truth_pixel):
        batch_size = len(logit_pixel)
        logit = logit_pixel.view(batch_size, -1)
        truth = truth_pixel.view(batch_size, -1)
        assert(logit.shape == truth.shape)

        loss = self.soft_dice_criterion(logit, truth)

        if self.size_average:
            loss = loss.mean()
        return loss

    def soft_dice_criterion(self, logit, truth):
        batch_size = len(logit)
        probability = torch.sigmoid(logit)

        p = probability.view(batch_size, -1)
        t = truth.view(batch_size, -1)

        w = truth.detach()
        self.weight = self.weight.type_as(logit)
        w = w * (self.weight[1] - self.weight[0]) + self.weight[0]

        p = w * (p*2 - 1)  #convert to [0,1] --> [-1, 1]
        t = w * (t*2 - 1)

        intersection = (p * t).sum(-1)
        union =  (p * p).sum(-1) + (t * t).sum(-1)
        dice  = 1 - 2 * intersection/union

        loss = dice
        return loss

#reference: https://github.com/zdaiot/Kaggle-Steel-Defect-Detection
class SoftBCEDiceLoss(nn.Module):

    def __init__(self, size_average=True, weight=[0.2, 0.8]):
        super().__init__()
        self.size_average = 'mean' if size_average else 'none'
        self.weight = weight
        self.bce_loss = nn.BCEWithLogitsLoss(
            reduction=self.size_average, 
            pos_weight=torch.tensor(self.weight[1])
        )
        self.softdiceloss = WeightedSoftDiceLoss(
            size_average=self.size_average, 
            weight=weight
        )

    def forward(self, input, target):
        input, target = TensorBase(input).float(), TensorBase(target).float()
        soft_bce_loss = self.bce_loss(input, target)
        soft_dice_loss = self.softdiceloss(input, target)
        loss = 0.7 * soft_bce_loss + 0.3 * soft_dice_loss

        return loss

#reference: https://github.com/zdaiot/Kaggle-Steel-Defect-Detection
class MultiClassesSoftBCEDiceLoss(nn.Module):

    def __init__(self, classes_num=4, size_average=True, weight=[0.2, 0.8]):
        super().__init__()
        self.classes_num = classes_num
        self.size_average = size_average
        self.weight = weight
        self.soft_bce_dice_loss = SoftBCEDiceLoss(size_average=self.size_average, weight=self.weight)

    def forward(self, input, target):
        """
        Args:
            input: tensor, [batch_size, classes_num, height, width]
            target: tensor, [batch_size, classes_num, height, width]
        """
        loss = 0
        for class_index in range(self.classes_num):
            input_single_class = input[:, class_index, :, :]
            target_singlt_class = target[:, class_index, :, :]
            single_class_loss = self.soft_bce_dice_loss(input_single_class, target_singlt_class)
            loss += single_class_loss

        loss /= self.classes_num

        return loss
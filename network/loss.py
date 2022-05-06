import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch.nn as nn


class DiceLoss(_Loss):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.axis = (2, 3, 4)
        self.smooth = 1e-7

    def forward(self, input, target):
        return 1. - self.dice_score(input, target)

    def dice_score(self, input, target):
        numerator = torch.sum(input * target, dim=self.axis)
        dice = 2. * (numerator + self.smooth) / (
                torch.sum(input, dim=self.axis) + torch.sum(target, dim=self.axis) + self.smooth)
        return torch.mean(dice)


class GeneralizedDiceLoss(DiceLoss):
    def __init__(self, n_classes=5, use_background=False):
        super(GeneralizedDiceLoss, self).__init__()
        self.n_classes = n_classes
        self.use_background = use_background

    def dice_score(self, y_pred, y_true):
        y_true = torch.squeeze(y_true, dim=1)
        one_hot_y_true = F.one_hot(y_true.type(torch.long), num_classes=5)
        one_hot_y_true = torch.permute(one_hot_y_true, [0, 4, 1, 2, 3])
        intersection = torch.sum(one_hot_y_true * y_pred, dim=self.axis)
        ground_o = torch.sum(one_hot_y_true, dim=self.axis)
        pred_o = torch.sum(y_pred, dim=self.axis)
        denominator = ground_o + pred_o
        w = torch.reciprocal(torch.square(ground_o.type(torch.float)))
        w_max = torch.max(torch.masked_select(w, torch.isfinite(w)))
        w = torch.where(torch.isinf(w), torch.full(w.shape, w_max, device='cuda:0'), w)
        generalized_dice = (2.0 * intersection * w + self.smooth) / (denominator * w + self.smooth)
        generalized_dice_by_label = torch.mean(generalized_dice, dim=0)
        if not self.use_background:
            return torch.mean(generalized_dice_by_label[1:])
        return torch.mean(generalized_dice_by_label)


class KDLoss(DiceLoss):
    def __init__(self):
        super(KDLoss, self).__init__()
        self.alpha = 0.5

    def forward(self, input, target):
        hard_labels = target[0]
        teacher_labels = target[1]
        return (1. - self.alpha) * (1. - self.dice_score(torch.sigmoid(input), hard_labels)) + self.alpha * (
            F.binary_cross_entropy_with_logits(input, teacher_labels))


class GeneralizedKDLoss(GeneralizedDiceLoss):
    def __init__(self):
        super(GeneralizedKDLoss, self).__init__()
        self.alpha = 0.5
        self.cross = nn.CrossEntropyLoss()

    def forward(self, input, target):
        hard_labels = target[0]
        teacher_labels = target[1]

        return (1. - self.alpha) * (1. - self.dice_score(F.softmax(input, dim=-1), hard_labels)) + self.alpha * -(
                teacher_labels * torch.log(F.softmax(input, dim=-1))).sum(dim=1).mean()

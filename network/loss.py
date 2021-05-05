import torch
from torch.nn.modules.loss import _Loss


class DiceLoss(_Loss):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.axis = (2,3,4)
        self.smooth = 1e-7

    def forward(self, input, target):
        return 1. - self.dice_score(input, target)

    def dice_score(self, input, target):
        numerator = torch.sum(input * target, axis=self.axis)
        dice = 2. * (numerator + self.smooth) / (
                torch.sum(input, axis=self.axis) + torch.sum(target, axis=self.axis) + self.smooth)
        return torch.mean(dice)


class KDLoss(DiceLoss):
    def __init__(self):
        super(KDLoss, self).__init__()
        self.kld = torch.nn.KLDivLoss(reduction='batchmean')

    def forward(self, input, target):
        hard_labels = target[0]
        teacher_labels = target[1]
        return 1. - self.dice_score(input, hard_labels)  # + self.kld(input, teacher_labels)

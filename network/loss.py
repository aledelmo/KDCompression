import torch
from torch.nn.modules.loss import _Loss


class DiceLoss(_Loss):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.axis = (2, 3, 4)
        self.smooth = 1e-7

    def forward(self, input, target):
        return 1. - self.dice_score(input, target)

    def dice_score(self, input, target):
        numerator = torch.sum(input * target, axis=self.axis) + self.smooth
        ground_o = torch.sum(target, dim=self.axis)
        pred_o = torch.sum(input, dim=self.axis)
        denominator = ground_o + pred_o + self.smooth
        dice = 2 * numerator / denominator
        return torch.mean(dice)

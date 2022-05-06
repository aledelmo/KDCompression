import torch
import torch.nn.functional as F


class DiceScore:
    def __init__(self):
        self._dice_scores = []
        self.axis = (2, 3, 4)
        self.smooth = 1e-7

    def reset(self):
        self._dice_scores = []

    def update(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        numerator = torch.sum(y_true * y_pred, dim=self.axis)
        dice = 2. * (numerator + self.smooth) / (
                torch.sum(y_true, dim=self.axis) + torch.sum(y_pred, dim=self.axis) + self.smooth)
        self._dice_scores.append(torch.mean(dice))

    def compute(self):
        return torch.mean(torch.FloatTensor(self._dice_scores))


class DiceScoreMultiClass(DiceScore):
    def __init__(self, n_classes=5, use_background=False):
        super(DiceScoreMultiClass, self).__init__()
        self.n_classes = n_classes
        self.use_background = use_background
        self.c = [[] for _ in range(n_classes)]

    def update(self, y_pred, y_true, sample_weight=None):
        y_true = torch.squeeze(y_true, dim=1)
        one_hot_y_true = F.one_hot(y_true.type(torch.long), num_classes=5)
        one_hot_y_true = torch.permute(one_hot_y_true, [0, 4, 1, 2, 3])

        numerator = torch.sum(one_hot_y_true * y_pred, dim=self.axis)

        dice = 2. * (numerator + self.smooth) / (
                torch.sum(one_hot_y_true, dim=self.axis) + torch.sum(y_pred, dim=self.axis) + self.smooth)

        dice_by_label = torch.mean(dice, dim=0)

        if not self.use_background:
            self._dice_scores.append(torch.mean(dice_by_label[1:]))
        else:
            self._dice_scores.append(torch.mean(dice_by_label))


class Mean:
    def __init__(self):
        self._values = []

    def reset(self):
        self._values = []

    def update(self, value):
        self._values.append(value)

    def compute(self):
        return torch.mean(torch.FloatTensor(self._values))

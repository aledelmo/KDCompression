import torch


class DiceScore:
    def __init__(self):
        self._dice_scores = []
        self.axis = (1,2,3)
        self.smooth = 1e-7

    def reset(self):
        self._dice_scores = []

    def update(self, y_pred, y_true):
        numerator = torch.sum(y_true * y_pred, axis=self.axis)
        dice = 2. * (numerator + self.smooth) / (
                torch.sum(y_true, axis=self.axis) + torch.sum(y_pred, axis=self.axis) + self.smooth)
        self._dice_scores.append(torch.mean(dice))

    def compute(self):
        return torch.mean(torch.FloatTensor(self._dice_scores))


class Mean:
    def __init__(self):
        self._values = []

    def reset(self):
        self._values = []

    def update(self, value):
        self._values.append(value)

    def compute(self):
        return torch.mean(torch.FloatTensor(self._values))

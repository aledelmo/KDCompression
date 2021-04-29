import torch
import numpy as np
from scipy.ndimage import find_objects


class Clip(object):
    def __init__(self, t1=5, t2=95):
        self.t1 = t1
        self.t2 = t2

    def __call__(self, sample):
        image, label, teacher = sample['image'], sample['label'], sample['teacher']
        image = np.clip(image, np.percentile(image, self.t1), np.percentile(image, self.t2))
        return {"image": image, "label": label, "teacher": teacher}


class Normalize(object):
    def __call__(self, sample):
        image, label, teacher = sample['image'], sample['label'], sample['teacher']

        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            image = (image - mean) / std
        else:
            image *= 0.

        return {"image": image, "label": label, "teacher": teacher}


class ToTensor(object):
    def __call__(self, sample):
        image, label, teacher = sample['image'], sample['label'], sample['teacher']
        image = np.expand_dims(image.transpose((2, 0, 1)), axis=0)
        label = np.expand_dims(label.transpose((2, 0, 1)), axis=0)
        teacher = np.expand_dims(teacher.transpose((2, 0, 1)), axis=0)
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label),
                'teacher': torch.from_numpy(teacher)}

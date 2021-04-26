import numpy as np
from scipy.ndimage import find_objects


class Clip(object):
    def __init__(self, t1=5, t2=95):
        self.t1 = t1
        self.t2 = t2

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = np.clip(image, np.percentile(image, self.t1), np.percentile(image, self.t2))
        return {"image": image, "label": label}


class Normalize(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            image = (image - mean) / std
        else:
            image *= 0.

        return {"image": image, "label": label}


class CutToBBox(object):
    def __init__(self, patch_size=(96, 48, 96)):
        self.patch_size = patch_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        loc = find_objects(label > 0)[0]
        image = image[loc]
        label = label[loc]

        pad_x = np.maximum(0, (self.patch_size[0] - image.shape[0]) // 2 + 1)
        pad_y = np.maximum(0, (self.patch_size[1] - image.shape[1]) // 2 + 1)
        pad_z = np.maximum(0, (self.patch_size[2] - image.shape[2]) // 2 + 1)

        padding = [[pad_x, pad_x], [pad_y, pad_y], [pad_z, pad_z]]
        image = np.pad(image, padding, "constant")
        label = np.pad(label, padding, "constant")
        return {"image": image, "label": label}

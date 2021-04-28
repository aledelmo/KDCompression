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


class CutToBBox(object):
    def __init__(self, patch_size=(96, 48, 96)):
        self.patch_size = patch_size

    def __call__(self, sample):
        image, label, teacher = sample['image'], sample['label'], sample['teacher']

        loc = find_objects(label > 0)[0]
        image = image[loc]
        label = label[loc]
        teacher = teacher[loc]

        pad_x = np.maximum(0, (self.patch_size[0] - image.shape[0]) // 2 + 1)
        pad_y = np.maximum(0, (self.patch_size[1] - image.shape[1]) // 2 + 1)
        pad_z = np.maximum(0, (self.patch_size[2] - image.shape[2]) // 2 + 1)

        padding = [[pad_x, pad_x], [pad_y, pad_y], [pad_z, pad_z]]
        image = np.pad(image, padding, "constant")
        label = np.pad(label, padding, "constant")
        teacher = np.pad(teacher, padding, "constant")
        return {"image": image, "label": label, "teacher": teacher}


class CropRandomPatch(object):
    def __init__(self, patch_size=(96, 48, 96)):
        self.patch_size = patch_size

    def __call__(self, sample):
        image, label, teacher = sample['image'], sample['label'], sample['teacher']

        dim_x, dim_y, dim_z = [p // 2 for p in self.patch_size]
        padding = [[dim_x, dim_x], [dim_y, dim_y], [dim_z, dim_z]]
        image = np.pad(image, padding, "constant")
        label = np.pad(label, padding, "constant")
        teacher = np.pad(teacher, padding, "constant")
        idx = np.array(np.where(label != 0)).T
        np.random.shuffle(idx)
        idx = idx[0]
        image = image[idx[0] - dim_x:idx[0] + dim_x, idx[1] - dim_y:idx[1] + dim_y, idx[2] - dim_z:idx[2] + dim_z]
        label = label[idx[0] - dim_x:idx[0] + dim_x, idx[1] - dim_y:idx[1] + dim_y, idx[2] - dim_z:idx[2] + dim_z]
        teacher = teacher[idx[0] - dim_x:idx[0] + dim_x, idx[1] - dim_y:idx[1] + dim_y, idx[2] - dim_z:idx[2] + dim_z]
        return {"image": image, "label": label, "teacher": teacher}


class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = image.transpose((2, 0, 1))
        label = label.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}

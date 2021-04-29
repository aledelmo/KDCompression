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
        self.dim_x_l, self.dim_x_h = (np.floor(patch_size[0] / 2), np.ceil(patch_size[0] / 2))
        self.dim_y_l, self.dim_y_h = (np.floor(patch_size[1] / 2), np.ceil(patch_size[1] / 2))
        self.dim_z_l, self.dim_z_h = (np.floor(patch_size[2] / 2), np.ceil(patch_size[2] / 2))

    def __call__(self, sample):
        image, label, teacher = sample['image'], sample['label'], sample['teacher']

        idx = np.random.shuffle(np.where(label > 0))[0]

        current_shape = label.shape
        if idx[0] - self.dim_x_l < 0:
            pad_value_left = abs(idx[0] - self.dim_x_l)
        else:
            pad_value_left = 0
        if idx[0] + self.dim_x_h > current_shape[0]:
            pad_value_right = abs(idx[0] + self.dim_x_h - current_shape[0])
        else:
            pad_value_right = 0
        if idx[1] - self.dim_y_l < 0:
            pad_value_bottom = abs(idx[1] - self.dim_y_l)
        else:
            pad_value_bottom = 0
        if idx[1] + self.dim_y_h > current_shape[1]:
            pad_value_top = abs(idx[1] + self.dim_y_h - current_shape[1])
        else:
            pad_value_top = 0
        if idx[2] - self.dim_z_l < 0:
            pad_value_back = abs(idx[2] - self.dim_z_l)
        else:
            pad_value_back = 0
        if idx[2] + self.dim_z_h > current_shape[2]:
            pad_value_forward = abs(idx[2] + self.dim_z_h - current_shape[2])
        else:
            pad_value_forward = 0

        padding = [[pad_value_left, pad_value_right],
                   [pad_value_bottom, pad_value_top],
                   [pad_value_back, pad_value_forward], [0, 0]]

        if np.any(np.reshape(padding, [-1]) != 0):
            image = np.pad(image, padding, "CONSTANT")
            label = np.pad(label, padding, "CONSTANT")
            teacher = np.pad(teacher, padding, "CONSTANT")

        idx = idx + (pad_value_left, pad_value_bottom, pad_value_back, 0)

        image = image[idx[0] - self.dim_x_l:idx[0] + self.dim_x_h, idx[1] - self.dim_y_l:idx[1] + self.dim_y_h,
                idx[2] - self.dim_z_l:idx[2] + self.dim_z_h, :]
        label = label[idx[0] - self.dim_x_l:idx[0] + self.dim_x_h, idx[1] - self.dim_y_l:idx[1] + self.dim_y_h,
                idx[2] - self.dim_z_l:idx[2] + self.dim_z_h, :]
        teacher = teacher[idx[0] - self.dim_x_l:idx[0] + self.dim_x_h, idx[1] - self.dim_y_l:idx[1] + self.dim_y_h,
                  idx[2] - self.dim_z_l:idx[2] + self.dim_z_h, :]

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

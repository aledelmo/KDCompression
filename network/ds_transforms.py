import torch
import numpy as np
import torch.nn.functional as F


class ToTensor(object):
    def __init__(self, hdf5=False):
        self.hdf5 = hdf5

    def __call__(self, sample):
        if self.hdf5:
            return torch.from_numpy(sample['image'][()]), torch.from_numpy(sample['label'][()]),\
                   torch.from_numpy(sample['teacher'][()])
        else:
            return torch.from_numpy(sample['image']), torch.from_numpy(sample['label']), \
                   torch.from_numpy(sample['teacher'])


patch_size = (96, 48, 96)
dim_x_l, dim_x_h = (np.floor(patch_size[0] / 2).astype(np.int32), np.ceil(patch_size[0] / 2).astype(np.int32))
dim_y_l, dim_y_h = (np.floor(patch_size[1] / 2).astype(np.int32), np.ceil(patch_size[1] / 2).astype(np.int32))
dim_z_l, dim_z_h = (np.floor(patch_size[2] / 2).astype(np.int32), np.ceil(patch_size[2] / 2).astype(np.int32))


def collate_random_crop(batch, device):
    cropped_image = []
    cropped_label = []
    cropped_teacher = []
    for b in batch:
        current_x, current_y, current_teacher = b[0].to(device=device, non_blocking=True),\
                                                b[1].to(device=device, non_blocking=True),\
                                                b[2].to(device=device, non_blocking=True),

        idx = torch.where(current_y > 0)
        idx = torch.stack(idx, dim=1)
        idx = torch.flatten(idx[torch.randint(low=0, high=idx.shape[0], size=(1,))])

        current_shape = current_y.shape
        if idx[2] - dim_x_l < 0:
            pad_value_left = abs(idx[2] - dim_x_l)
        else:
            pad_value_left = 0
        if idx[2] + dim_x_h > current_shape[2]:
            pad_value_right = abs(idx[2] + dim_x_h - current_shape[2])
        else:
            pad_value_right = 0

        if idx[3] - dim_y_l < 0:
            pad_value_bottom = abs(idx[3] - dim_y_l)
        else:
            pad_value_bottom = 0
        if idx[3] + dim_y_h > current_shape[3]:
            pad_value_top = abs(idx[3] + dim_y_h - current_shape[3])
        else:
            pad_value_top = 0

        if idx[1] - dim_z_l < 0:
            pad_value_back = abs(idx[1] - dim_z_l)
        else:
            pad_value_back = 0
        if idx[1] + dim_z_h > current_shape[1]:
            pad_value_forward = abs(idx[1] + dim_z_h - current_shape[1])
        else:
            pad_value_forward = 0

        padding = [pad_value_bottom, pad_value_top,
                   pad_value_left, pad_value_right,
                   pad_value_back, pad_value_forward, 0, 0]

        if sum(padding) > 0:
            current_x = F.pad(current_x, padding, "constant")
            current_y = F.pad(current_y, padding, "constant")
            current_teacher = F.pad(current_teacher, padding, "constant")

            idx = idx + torch.tensor([0, pad_value_back, pad_value_left, pad_value_bottom], device=device)

        current_x = current_x[:, idx[1] - dim_z_l:idx[1] + dim_z_h, idx[2] - dim_x_l:idx[2] + dim_x_h,
                              idx[3] - dim_y_l:idx[3] + dim_y_h]
        current_y = current_y[:, idx[1] - dim_z_l:idx[1] + dim_z_h, idx[2] - dim_x_l:idx[2] + dim_x_h,
                              idx[3] - dim_y_l:idx[3] + dim_y_h]
        current_teacher = current_teacher[:, idx[1] - dim_z_l:idx[1] + dim_z_h, idx[2] - dim_x_l:idx[2] + dim_x_h,
                                          idx[3] - dim_y_l:idx[3] + dim_y_h]

        cropped_image.append(current_x)
        cropped_label.append(current_y)
        cropped_teacher.append(current_teacher)

    return torch.stack(cropped_image), torch.stack(cropped_label), torch.stack(cropped_teacher)

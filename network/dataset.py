import glob
import os.path
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, Subset

from sklearn.model_selection import train_test_split as split


class MRIDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.imgs = sorted(glob.glob(os.path.join(img_dir, '*[tT]2*.nii.gz')))
        self.masks = sorted(glob.glob(os.path.join(img_dir, '*[mM]ask*.nii.gz')))
        self.labels = sorted(glob.glob(os.path.join(img_dir, '*[sS]egmentation*.nii.gz')))
        self.teacher_labels = sorted(glob.glob(os.path.join(img_dir, '*[tT]eacher*.nii.gz')))
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.load_nii(self.imgs[idx])
        mask = self.load_nii(self.masks[idx])
        label = self.load_nii(self.labels[idx])
        teacher = self.load_nii(self.teacher_labels[idx])

        img *= mask
        label *= mask

        sample = {"image": img, "label": label, "teacher": teacher}

        if self.transform:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def load_nii(f_name):
        img = nib.load(f_name)
        canonical_img = nib.as_closest_canonical(img)
        return canonical_img.get_fdata()


def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = split(np.arange(len(dataset)), test_size=val_split)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

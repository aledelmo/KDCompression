import h5py
import numpy as np
from torch.utils.data import Dataset, Subset

from sklearn.model_selection import train_test_split as split


class MRIDataset(Dataset):
    def __init__(self, ds_path, transform=None):
        self.ds = h5py.File(ds_path, 'r')
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img = self.ds["sample_{}".format(idx)]['img']
        label = self.ds["sample_{}".format(idx)]['label']
        teacher = self.ds["sample_{}".format(idx)]['teacher']

        sample = {"image": np.array(img), "label": np.array(label), "teacher": np.array(teacher)}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def close(self):
        self.ds.close()


def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = split(np.arange(len(dataset)), test_size=val_split)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

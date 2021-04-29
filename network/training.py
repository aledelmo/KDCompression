import os.path
from tqdm import tqdm
from time import sleep
import numpy as np

from .metrics import Mean

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


class Training:
    def __init__(self, model, optimizer, loss_fn, metric_fn, train_ds, test_ds, patch_size, log_dir):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.patch_size = patch_size
        self.log_dir = log_dir

        self.train_loss = Mean()
        self.test_loss = Mean()
        self.summary_writer_train = SummaryWriter(os.path.join(log_dir, 'Train'))
        self.summary_writer_test = SummaryWriter(os.path.join(log_dir, 'Test'))

        self.scaler = GradScaler()

        self.dim_x_l, self.dim_x_h = (np.floor(patch_size[0] / 2).astype(np.int32),
                                      np.ceil(patch_size[0] / 2).astype(np.int32))
        self.dim_y_l, self.dim_y_h = (np.floor(patch_size[1] / 2).astype(np.int32),
                                      np.ceil(patch_size[1] / 2).astype(np.int32))
        self.dim_z_l, self.dim_z_h = (np.floor(patch_size[2] / 2).astype(np.int32),
                                      np.ceil(patch_size[2] / 2).astype(np.int32))

    def train_model(self, epochs, device):
        print('Training model...')
        sleep(0.01)
        with tqdm(range(epochs), desc="Training", unit="epoch") as pbar:
            for epoch in pbar:
                sleep(0.01)
                self._train_epoch(device)

                self.summary_writer_train.add_scalar("Loss", self.train_loss.compute().item(), epoch)
                self.summary_writer_test.add_scalar("Loss", self.test_loss.compute().item(), epoch)
                self.summary_writer_test.add_scalar("Dice Score", self.metric_fn.compute().item(), epoch)

                postfix = {"Train Loss": self.train_loss.compute().item(),
                           "Test Loss": self.test_loss.compute().item(),
                           "Dice Score": self.metric_fn.compute().item()}
                pbar.set_postfix(**postfix)

                self.train_loss.reset()
                self.test_loss.reset()
                self.metric_fn.reset()

        self.summary_writer_train.close()
        self.summary_writer_test.close()

    def _preprocess(self, current_x, current_y, current_teacher):
        idx = torch.where(current_y >= 0)
        idx = torch.stack(list(idx), dim=1)
        idx = idx[torch.randint(low=0, high=idx.shape[0], size=(1,), device='cuda')][0]

        current_shape = current_y.shape
        if idx[3] - self.dim_x_l < 0:
            pad_value_left = abs(idx[3] - self.dim_x_l)
        else:
            pad_value_left = 0
        if idx[3] + self.dim_x_h > current_shape[3]:
            pad_value_right = abs(idx[3] + self.dim_x_h - current_shape[3])
        else:
            pad_value_right = 0

        if idx[4] - self.dim_y_l < 0:
            pad_value_bottom = abs(idx[4] - self.dim_y_l)
        else:
            pad_value_bottom = 0
        if idx[4] + self.dim_y_h > current_shape[4]:
            pad_value_top = abs(idx[4] + self.dim_y_h - current_shape[4])
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

        padding = (pad_value_bottom, pad_value_top,pad_value_left, pad_value_right,pad_value_back, pad_value_forward,0,0,0,0)

        image = F.pad(current_x, padding, "constant")
        label = F.pad(current_y, padding, "constant")
        teacher = F.pad(current_teacher, padding, "constant")

        idx = idx + torch.tensor([0, 0, pad_value_back, pad_value_left, pad_value_bottom]).to(device='cuda')

        image = image[:, :,  idx[2] - self.dim_z_l:idx[2] + self.dim_z_h, idx[3] - self.dim_x_l:idx[3] + self.dim_x_h,
                      idx[4] - self.dim_y_l:idx[4] + self.dim_y_h]
        label = label[:, :, idx[2] - self.dim_z_l:idx[2] + self.dim_z_h, idx[3] - self.dim_x_l:idx[3] + self.dim_x_h,
                      idx[4] - self.dim_y_l:idx[4] + self.dim_y_h]
        teacher = teacher[:, :,  idx[2] - self.dim_z_l:idx[2] + self.dim_z_h, idx[3] - self.dim_x_l:idx[3] + self.dim_x_h,
                          idx[4] - self.dim_y_l:idx[4] + self.dim_y_h]

        return image, label, teacher

    def _train_epoch(self, device):
        self.model.train()
        for sample in self.train_ds:
            x_train, y_train, y_teacher_train = sample['image'].to(device), sample["label"].to(device), sample["teacher"].to(device)
            x_train, y_train, y_teacher_train = self._preprocess( x_train, y_train, y_teacher_train)
            self._train_step(x_train, y_train, y_teacher_train)
        self.model.eval()
        for sample in self.test_ds:
            x_test, y_test, y_teacher_test = sample['image'].to(device), sample["label"].to(device), sample["teacher"].to(device)
            x_test, y_test, y_teacher_test = self._preprocess(x_test, y_test, y_teacher_test)
            self._test_step(x_test, y_test, y_teacher_test)

    def _train_step(self, x_train, y_train, y_teacher_train):
        self.optimizer.zero_grad(set_to_none=True)
        with autocast():
            y_pred = self.model(x_train)
            loss = self.loss_fn(y_pred, [y_train, y_teacher_train])
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.train_loss.update(loss)

    @torch.no_grad()
    def _test_step(self, x_test, y_test, y_teacher_test):
        y_pred = self.model(x_test)
        loss = self.loss_fn(y_pred, [y_test, y_teacher_test])
        self.test_loss.update(loss)
        self.metric_fn.update(y_pred, y_test)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

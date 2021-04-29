import os.path
from tqdm import tqdm
from time import sleep

from .metrics import Mean

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


class Training:
    def __init__(self, model, optimizer, loss_fn, metric_fn, train_ds, test_ds, log_dir):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.log_dir = log_dir

        self.train_loss = Mean()
        self.test_loss = Mean()
        self.summary_writer_train = SummaryWriter(os.path.join(log_dir, 'Train'))
        self.summary_writer_test = SummaryWriter(os.path.join(log_dir, 'Test'))

        self.scaler = GradScaler()

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

    def _train_epoch(self, device):
        self.model.train()
        for sample in self.train_ds:
            x_train, y_train, y_teacher_train = sample['image'], sample["label"], sample["teacher"]
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            y_teacher_train = y_teacher_train.to(device)
            self._train_step(x_train, y_train, y_teacher_train)
        self.model.eval()
        for sample in self.test_ds:
            x_test, y_test, y_teacher_test = sample['image'], sample["label"], sample["teacher"]
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            y_teacher_test = y_teacher_test.to(device)
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

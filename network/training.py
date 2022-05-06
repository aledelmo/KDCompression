import os.path
from tqdm import tqdm
from time import sleep

from .metrics import Mean

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter


class Training:
    def __init__(self, model, optimizer, loss_fn, metric_fn, train_ds, test_ds, patch_size, log_dir, ckpt_dir):
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

        self.ckpt_dir = os.path.join(os.getcwd(), ckpt_dir)

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        self.scaler = GradScaler()

        self.best_model_state = None

    def train_model(self, epochs, patch_per_image):
        best_metric = 0.

        print('Training model...')
        with tqdm(range(epochs), desc="Training", unit="epoch") as pbar:
            for epoch in pbar:
                sleep(0.01)
                self._train_epoch(patch_per_image)

                self.summary_writer_train.add_scalar("Loss", self.train_loss.compute().item(), epoch)
                self.summary_writer_test.add_scalar("Loss", self.test_loss.compute().item(), epoch)
                self.summary_writer_test.add_scalar("Dice Score", self.metric_fn.compute().item(), epoch)

                postfix = {"Train Loss": self.train_loss.compute().item(),
                           "Test Loss": self.test_loss.compute().item(),
                           "Dice Score": self.metric_fn.compute().item()}
                pbar.set_postfix(**postfix)

                if self.metric_fn.compute().item() > best_metric:

                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }, os.path.join(self.ckpt_dir, 'ckpt-{}'.format(epoch)))
                    best_metric = self.metric_fn.compute().item()
                    self.best_model_state = self.model.state_dict()

                self.train_loss.reset()
                self.test_loss.reset()
                self.metric_fn.reset()

        self.summary_writer_train.close()
        self.summary_writer_test.close()

    def _train_epoch(self, patch_per_image):
        self.model.train()
        for _ in range(patch_per_image):
            for sample in self.train_ds:
                self._train_step(*sample)
        self.model.eval()
        for _ in range(patch_per_image):
            for sample in self.test_ds:
                self._test_step(*sample)

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
        y_pred = F.softmax(self.model(x_test), dim=-1)
        loss = self.loss_fn(y_pred, [y_test, y_teacher_test])
        self.test_loss.update(loss)
        self.metric_fn.update(y_pred, y_test)

    def save_model(self, path):
        torch.save(self.best_model_state, path)

    def restore_from_checkpoint(self):
        ckpts = sorted(os.listdir(self.ckpt_dir))
        if ckpts:
            ckpt = ckpts[-1]
            checkpoint = torch.load(ckpt)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Restored from {}".format(ckpt))
        else:
            print("Initializing from scratch.")

import torch
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from igs import Ft, IFt


class BraTSClassifierBinary(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.net = torchvision.models.resnet18()
        self.net.conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.net.fc = torch.nn.Linear(in_features=512, out_features=2, bias=True)
        self.register_buffer('sampling', torch.ones(1, 4, 240, 1))
        self.use_sampling = False

    @staticmethod
    def _to_pred(logits):
        return logits.argmax(1)

    def predict(self, batch):
        logits = self(batch)
        return self._to_pred(logits)

    def forward(self, batch):
        if self.use_sampling:
            ks = Ft(batch['image'] * batch['std'] + batch['mean']) * self.sampling
            images = (IFt(ks).abs() - batch['mean']) / (batch['std'] + 1e-11)
            return self.net(images)
        else:
            return self.net(batch['image'])

    def training_step(self, batch, batch_idx):
        y = self(batch)
        loss = F.cross_entropy(y, batch['target'])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        y = self(batch)
        loss = F.cross_entropy(y, batch['target'])
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()))
        return optimizer


class BraTSClassifierHuber(BraTSClassifierBinary):

    def __init__(self):
        super().__init__()
        self.net.fc = torch.nn.Linear(in_features=512, out_features=1, bias=True)

    @staticmethod
    def _to_pred(logits):
        return (logits >= 0).float().flatten()

    def training_step(self, batch, batch_idx):
        y = self(batch)
        target = batch['target']
        target[target == 0] = -1
        loss = F.soft_margin_loss(y.flatten(), target)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        y = self(batch)
        target = batch['target']
        target[target == 0] = -1
        loss = F.soft_margin_loss(y.flatten(), target)
        self.log('val_loss', loss)
        return loss

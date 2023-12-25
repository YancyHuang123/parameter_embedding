from typing import List, Optional
import torch.nn as nn
import torch
from torch.optim import SGD, Adam
from Wrapper.WrapperModule import WrapperModule
from torchmetrics.classification import MulticlassAccuracy
import torch.nn.functional as F


class ResnetExperiment(WrapperModule):
    def __init__(self, max_lr, epochs, trainset_len):
        super(ResnetExperiment, self).__init__()
        self.hostnet = None
        self.opt = Adam(self.resnet.parameters())
        self.loss_fun = nn.CrossEntropyLoss()
        self.train_acc = MulticlassAccuracy(num_classes=10)
        self.distribution_ignore = ['loss_fun', 'train_acc']
        self.X_random=torch.randn()
        

    def training_step(self, batch, batch_idx):
        X, Y = batch
        out = self.resnet(X)
        loss = self.loss_fun(out, Y)
        self.train_acc(out, Y)
        loss.backward()
        
        self.opt.step()

        self.log_dict({'loss': loss}, on_epoch=True, on_step=False)

    def on_epoch_end(self, training_results, val_results):
        self.log_dict({'training_acc': self.train_acc.compute()},
                      on_step=True, on_epoch=False)  # add auto epoch adding feature
        self.train_acc.reset()

    def test_step(self, batch, batch_idx):
        pass

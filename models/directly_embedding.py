from typing import List, Optional
import torch.nn as nn
import torch
from torch.optim import SGD, Adam
from Wrapper.WrapperModule import WrapperModule
from torchmetrics.classification import MulticlassAccuracy
import torch.nn.functional as F


class DirectlyEmbedding(WrapperModule):
    def __init__(self, parameters, code: torch.Tensor):
        super(DirectlyEmbedding, self).__init__()

        # the flatten mean parameters
        
        self.p=parameters
        # self.w = nn.Parameter(w, requires_grad=True)

        w=torch.mean(self.p, dim=3).reshape(-1)
        self.w_init = w.clone().detach()

        self.opt = Adam([parameters], lr=0.001)
        self.train_acc = MulticlassAccuracy(num_classes=10)
        self.distribution_ignore = ['train_acc']
        self.code = code  # the embedding code
        self.code_len = code.shape[0]

        self.X_random = torch.randn((self.code_len, self.w_init.shape[0]))

    def loss_fun(self, x, y, new, old, l):
        penalty = l*F.binary_cross_entropy(x, y)
        norm = torch.linalg.norm(new-old)/2.0
        loss = norm+penalty
        loss = penalty
        return loss, penalty, norm

    def decode(self, X, w):
        prob = self.get_prob(X, w)
        return torch.where(prob > 0.5, 1, 0)

    def get_prob(self, X, w):
        mm = torch.mm(self.X_random, w.reshape((w.shape[0], 1)))
        return F.sigmoid(mm).flatten()

    def training_step(self, batch, batch_idx):
        w = torch.mean(self.p, dim=3).reshape(-1)
        prob = self.get_prob(self.X_random, w)
        loss, penalty, norm = self.loss_fun(
            prob, self.code, w, self.w_init, l=10)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        self.log_dict({'loss': loss, 'penalty': penalty, 'norm': norm},
                      on_epoch=False, on_step=True)

    def test_step(self, batch, batch_idx):
        w = torch.mean(self.p, dim=3).reshape(-1)
        decode = self.decode(self.X_random, w)
        self.train_acc(decode, self.code)
        self.log_dict({'decode_acc': self.train_acc.compute()})
        self.train_acc.reset()
        print(decode)

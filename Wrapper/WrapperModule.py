from abc import abstractmethod
from typing import List, Dict, Union, Optional
import torch
import torch.nn as nn

from Wrapper.WrapperLogger import WrapperLogger


class WrapperModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.current_epoch = 0
        self.current_step = 0
        self.device = 'cpu'
        # save the names of attributes that you don't want to distribute to mutiple devices
        self.distribution_ignore = []
        self.cuda_ignore = []  # save the attributes that you don't want to move to cuda

    def save(self, save_folder):
        torch.save(self.state_dict(), f'{save_folder}/model.pt')

    def load(self, path):
        state_dict = torch.load(path)
        # make the state_dict compatible for nn.DataParallel
        state_dict = {k.replace('module.', ''): v for k,
                      v in state_dict.items()}
        self.load_state_dict(state_dict)

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def on_epoch_end(self, training_results: Optional[List] = None, val_results: Optional[List] = None):
        pass

    @abstractmethod
    def on_validation_end(self, results: Optional[List] = None):
        pass

    @abstractmethod
    def on_training_end(self, results: Optional[List] = None):
        pass

    @abstractmethod
    def test_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def on_test_end(self, results):
        pass

    @abstractmethod
    def configure_optimizers(self):
        pass

    @abstractmethod
    def configure_losses(self):
        pass

    def log_dict(self, dict: Dict, on_step=False, on_epoch=True, prog_bar=True):
        if on_epoch:
            self.logger.add_epoch_log(dict)
        if on_step:
            self.logger.add_log(dict, self.current_epoch, self.current_step)

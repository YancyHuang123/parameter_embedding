from Wrapper.WrapperDataModule import WrapperDataModule
from .dataset import get_dataset
from torch.utils.data import Dataset, DataLoader


class Loader(WrapperDataModule):
    def __init__(self, batch_size, worker=4):
        self.batch_size = batch_size
        self.worker = worker
        self.trainset, self.testset = get_dataset()

    def train_dataloader(self):
        train_loader = DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.worker,
                                  shuffle=True, pin_memory=True)
        return train_loader

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        test_loader = DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.worker,
                                 shuffle=False, pin_memory=True)
        return test_loader

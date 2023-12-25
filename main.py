from data_preprocess.cifar_loader import Loader
from Wrapper.WrapperTrainer import WrapperTrainer as Trainer
from models.resnet import ResnetExperiment

batch_size = 128

if __name__ == '__main__':
    max_epochs = 200
    data_loader = Loader(batch_size=batch_size)
    train_loader = data_loader.train_dataloader()
    test_loader = data_loader.test_dataloader()
    model = ResnetExperiment()

    train = Trainer(max_epochs, accelerator='gpu')

    train.fit(model, train_loader)

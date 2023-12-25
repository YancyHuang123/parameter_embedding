from data_preprocess.cifar_loader import Loader
from Wrapper.WrapperTrainer import WrapperTrainer as Trainer
from models.resnet import ResnetExperiment, ResNet34
from models.directly_embedding import DirectlyEmbedding
import torch

batch_size = 128

if __name__ == '__main__':
    max_epochs = 1
    data_loader = Loader(batch_size=batch_size)
    train_loader = data_loader.train_dataloader()
    test_loader = data_loader.test_dataloader()
    model = ResnetExperiment()
    model.load('./lite_logs/2023-12-21_20-17-28/model.pt')
    model.resnet.train()
    parameters = model.resnet.layer1[0].conv1.weight
    a=parameters.clone().detach()
    
    # trainer.fit(model, train_loader,test_loader)
    print(parameters.grad)

    embedding = DirectlyEmbedding(parameters, torch.ones((512*8,)))
    trainer_embedding = Trainer(max_epochs, accelerator='cpu')
    trainer_embedding.fit(embedding, [[torch.tensor(1)] for i in range(200)])

    #model.resnet.layer1[0].conv1.weight=embedding.w

    print(torch.equal(model.resnet.layer1[0].conv1.weight,a))

    trainer_embedding.test(embedding, [[torch.tensor(1)] for i in range(2)])

    trainer_model = Trainer(max_epochs, accelerator='gpu',
                            saving_folder=trainer_embedding.save_folder)
    trainer_model.test(model, test_loader)
    
    
    # trainer_model.test(model,test_loader)
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
transform_test = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)

def get_dataset():
    # load trainset and testset
    trainset = torchvision.datasets.CIFAR10(
        root="./datas", train=True, download=True, transform=transform_train
    )

    testset = torchvision.datasets.CIFAR10(
        root="./datas", train=False, download=True, transform=transform_test
    )

    return trainset, testset

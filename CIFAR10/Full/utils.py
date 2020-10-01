import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch

def read_vision_dataset(path, batch_size=128, num_workers=4, dataset='CIFAR10', transform=None):
    '''
    Read dataset available in torchvision
    
    Arguments:
        dataset : string
            The name of dataset, it should be available in torchvision
        transform_train : torchvision.transforms
            train image transformation
            if not given, the transformation for CIFAR10 is used
        transform_test : torchvision.transforms
            train image transformation
            if not given, the transformation for CIFAR10 is used
    Return: 
        trainloader, testloader
    '''
    if not transform and dataset=='CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    trainset = getattr(datasets,dataset)(root=path, train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testset = getattr(datasets,dataset)(root=path, train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return trainloader, testloader

def random(size, device='cpu'):
    dist = torch.distributions.normal.Normal(0, 1)
    samples = dist.rsample((size, 3, 32, 32))
    samples = samples / samples.reshape(size, -1).norm(dim=1)[:, None, None, None]
    return samples.to(device)
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST

from .utils import *


def get_MNIST_data_loaders(train_batch_size, val_batch_size, train_hold_ratio=0.5):
    dataset_name = 'MNIST'
    class_count = 10
    
    train_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
    
    test_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
    
    train_dataset = datasets.FashionMNIST('./data', download=True, train=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    
    test_dataset = datasets.FashionMNIST('./data', download=True, train=False, transform=test_transform)
    
    test_indices = np.arange(len(test_dataset))
    divide_point = int(len(test_dataset) * train_hold_ratio)
    
    validation_dataset = Subset(train_dataset, indices=np.random.permutation(test_indices)[:divide_point])
    validation_loader = DataLoader(validation_dataset, batch_size=val_batch_size, shuffle=True)
    
    holdout_dataset = Subset(train_dataset, indices=np.random.permutation(test_indices)[divide_point:])
    holdout_loader = DataLoader(holdout_dataset, batch_size=val_batch_size, shuffle=True)
    
    return train_loader, validation_loader, holdout_loader, dataset_name, class_count


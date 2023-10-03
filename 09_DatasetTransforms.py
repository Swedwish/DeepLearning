import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Dataset

dataset = torchvision.datasets.MNIST(
    root='./data', transform=torchvision.transforms.ToTensor,
)

class WineDataset(Dataset):

    def __init__(self, transform=None) -> None:
        #data loading
        xy = np.loadtxt('./data/wine.csv', delimiter=',', dtype=np.float32, skiprows= 1)
        self.y = xy[:,[0]]
        self.x = xy[:,1:]
        self.n_samples = xy.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index],self.y[index]   #dataset[0]...

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples
    

class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs),torch.from_numpy(targets)
    

class MulTransform:
    def __init__(self, factor) -> None:
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target
    
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
dataset = WineDataset(composed)


#dataset = WineDataset(transform=ToTensor())

first_data = dataset[0]
features, labels = first_data
print(type(features),type(labels))
print(f'{features}\n {WineDataset()[0][0]}')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

testDataLocation = 'C:/Users/vlads/Desktop/DeepLearning/data/DigitRecognition/test.csv'

class LinearModel2(nn.Module):
    def __init__(self, n_inputfeatures, n_hidden, n_classes) -> None:
        super(LinearModel2,self).__init__()
        self.linear1 = nn.Linear(n_inputfeatures, n_hidden)
        self.activFunc = nn.ReLU()
        self.linear2 = nn.Linear(n_hidden, n_classes)

    def forward(self, x):

        out = self.linear1(x)
        out = self.activFunc(out)
        out = self.linear2(out)
        return out

class DigitTestData(Dataset):
    def __init__(self, dataLocation) -> None:
        self.xy = np.loadtxt(dataLocation,skiprows=1,delimiter=',', dtype=np.float32)
        self.n_samples = self.xy.shape[0]

    def __getitem__(self, index):
        return self.xy[index]
    
    def __len__(self):
        return self.n_samples

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
testDataLocation = 'C:/Users/vlads/Desktop/DeepLearning/data/DigitRecognition/test.csv'
PATH = 'C:/Users/vlads/Desktop/DeepLearning/12/model.pth'
model = torch.load(PATH)
model.eval()


testData = DigitTestData(dataLocation=testDataLocation)
n_images = len(testData)
batch_size = n_images
testDataLoader = DataLoader(dataset= testData, batch_size=batch_size)
dataIter = iter(testDataLoader)
sample = next(dataIter)

print(len(sample), len(testData))
sample = sample.to(device)
out = model(sample)
_, predictions = torch.max(out,1)
predictions = predictions.to('cpu').numpy()
a = np.arange(1, n_images + 1)
a = np.reshape(a,(-1,1))
predictions = np.reshape(predictions,(-1,1))
result = np.hstack((a,predictions),dtype=np.int64)
np.savetxt("C:/Users/vlads/Desktop/DeepLearning/12/result.csv", result, delimiter=",", fmt='%d')


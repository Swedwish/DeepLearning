#MNIST
#DataLoader, Transformation
#Multilayer net, activation
#Loss and optim
#Training loop
#Model evaluation
#GPU support
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

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



class DigitData(Dataset):
    def __init__(self, dataLocation) -> None:
        self.xy = np.loadtxt(dataLocation,skiprows=1,delimiter=',', dtype=np.float32)
        self.n_samples = self.xy.shape[0]

    def __getitem__(self, index):
        x = self.xy[index,1:]
        y = self.xy[index,[0]]
        return x,y
    
    def __len__(self):
        return self.n_samples


#trainDataLocation = './data/DigitRecognition/train.csv'
trainDataLocation = 'C:/Users/vlads/Desktop/DeepLearning/data/DigitRecognition/train.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
digitData = DigitData(trainDataLocation)
trainData,testData = train_test_split(digitData,test_size=0.2, random_state=1234)
n_inputfeatures = 28*28
n_classes = 10
batch_size = 100
n_hidden = 100
#print(digitData.xy.shape)
train_loader = DataLoader(dataset=trainData,batch_size= batch_size, shuffle=True)
test_loader = DataLoader(dataset=testData,batch_size= batch_size, shuffle=False)

example = iter(train_loader)
samples,labels = next(example)
#print(np.array(testData).shape,np.array(trainData).shape, digitData.xy.shape)



#scaler = MinMaxScaler()
#digitData.x = scaler.fit_transform(digitData.x)
#scaler.fit()





model = LinearModel2(n_inputfeatures=n_inputfeatures, n_hidden= n_inputfeatures, n_classes= n_classes).to(device)
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
n_epochs = 5
n_batches = len(trainData)/batch_size

for epoch in range(n_epochs):
    for i,(images,labels) in enumerate(train_loader):
        images = images.to(device)
        y_pred = model(images)
        labels = labels.view(batch_size)
        labels = labels.type(torch.LongTensor).to(device)
        #print(y_pred.shape, labels.shape)
        loss = criterion(y_pred, labels)
    #   -backward pass: gradients
        loss.backward()
        
    #   -update weight
        optimizer.step()
        optimizer.zero_grad()
        if (i+1)%100 == 0:
            print(f'epoch:{epoch+1}/{n_epochs}, step = {i+1}, loss = {loss.item():.4f}')
    
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images,labels in test_loader:
        images = images.to(device)
        labels = labels.view(batch_size).to(device)
        outputs = model(images)

        _, predictions = torch.max(outputs,1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
        #print(n_samples,n_correct, predictions.shape, labels.shape)
    acc = n_correct/n_samples
    print(f'total accuracy = {acc}')

PATH = 'C:/Users/vlads/Desktop/DeepLearning/12/model.pth'
torch.save(model, PATH)
'''
    example = iter(test_loader)
    samples,labels = next(example)
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.imshow(samples[i].view(28,28), cmap = 'gray')
    plt.show()
    samples = samples.to(device)
    out = model(samples)
    _, predictions = torch.max(out,1)
    for i in range(15):
        print(f'predicted:{predictions[i]}, real:{labels[i][0]}')
'''
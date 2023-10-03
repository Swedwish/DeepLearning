import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis= 0)

x = np.array([2.0, 1.0, 0.1])

print(softmax(x))
x = torch.tensor([2.0,1.0,0.1], dtype=torch.float32)

outputs = torch.softmax(x,dim=0)
print(outputs)

def cross_entropy(actual, predicted):
    loss = -np.sum(actual*np.log(predicted))
    return loss

Y = np.array([1.0,0.0,0.0])
Y_good_pred = np.array([0.7,0.2,0.1])
Y_bad_pred = Y_good_pred[::-1]
print(cross_entropy(Y,Y_good_pred)) #loss
print(cross_entropy(Y,Y_bad_pred))  #loss
#big loss is bad

loss = nn.CrossEntropyLoss()
Y = torch.tensor([0]) #size nsamples*nclasses = 1x3
Y_good_pred = torch.tensor([[2.0,1.0,0.1]])
Y_bad_pred = torch.tensor([[0.5,2.0,0.3]])

l1 = loss(Y_good_pred, Y)
l2 = loss(Y_bad_pred, Y)

print(l1.item())
print(l2.item())
#1) Design model (input,output size, forward pass)
#2) Construct loss and optimiser 
#3) Training loop:
#   -forward pass: compute prediction and loss
#   -backward pass: gradients
#   -update weight

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
#0) prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples=100,n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape
print(n_samples, n_features)
#1) model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

#2) loss and optimiser
learning_rate = 0.05
criterion = nn.MSELoss() 
optimiser = torch.optim.SGD(model.parameters(),lr=learning_rate)
#3) training looop
n_epochs = 100
for epoch in range(n_epochs):
#   -forward pass: compute prediction and loss
    y_predicted = model(X)
    loss = criterion(y_predicted,y)
#   -backward pass: gradients
    loss.backward()
#   -update weight
    optimiser.step()

    optimiser.zero_grad() 

    if (epoch+1)%1 == 0:
        print(f'epoch:{epoch+1}, loss = {loss.item():.4f}')

predicted = model(X).detach().numpy()
plt.plot(X_numpy,y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()
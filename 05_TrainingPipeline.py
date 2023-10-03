import torch
import torch.nn as nn

# f = w * x

# f = 2 * x

x = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

x_test = torch.tensor([5], dtype= torch.float32)
n_samples, n_features = x.shape

#model prediction
input_size = n_features
output_size = n_features
#model = nn.Linear(input_size, output_size)
class LinearRegression(nn.Module):

    def __init__(self,input_dim, output_dim) -> None:
        super(LinearRegression, self).__init__()
        #define layers
        
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)


model = LinearRegression(input_size, output_size)

#loss = Mean Square Error == MSE
loss = nn.MSELoss()

# gradient
# MSE = 1/N * (w*x - y)**2
# dJ/dx = 1/N * 2x * (wx-y)*w


print(f'Prediction before training: f(5) = {model(x_test).item():.3f}')

#Training

learning_rate = 0.01
n_iters = 100
print(f'params:{model.parameters}')
optimiser = torch.optim.SGD(model.parameters(),lr = learning_rate)

for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = model(x)

    #loss 
    l = loss(y,y_pred)

    #gradients
    l.backward()

    #update weights
    optimiser.step()    
    #zero gradients
    optimiser.zero_grad()

    if epoch%10==0:
        [w,b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')


print(f'prediction after training: f(5) = {model(x_test).item():.3f}')


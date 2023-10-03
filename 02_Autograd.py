import torch

x = torch.rand(3, requires_grad=True)
print(x)
y = x + 2       #addBackward
print(y)
z = y*y*2
print(z)        #Multbackward
#z = z.mean()   #Meanbackward
print(z)

v = torch.tensor([0.1, 1.0, 0.001], dtype= torch.float32)       #J*v, where J is Jacobian
z.backward(v)                                                   #no need for an argument if Z is scalar


#To stop tracking grad:
x.requires_grad_(False)
y = x.detach()         #y does not requre grad
with torch.no_grad():
    y = x + 2          #y does not requre grad





#example

weights = torch.ones(4,requires_grad=True)
for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()




#optimiser
weights = torch.ones(4,requires_grad=True)
optimizer = torch.optim.SGD(weights, lr = 0.01 )
optimizer.step()
optimizer.zero_grad()


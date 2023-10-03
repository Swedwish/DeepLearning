import torch
import numpy as np

x = torch.rand(2,2)
y = torch.rand(2,2)

z = x+y    #z = torch.add(x,y)
x.add_(y) #== x+=y
x[1][1].item()    #get value
x.view(4) #to 1 dimention
x.view(-1,1)  #to 4x1 size
print(z)


#NUMPY
a = torch.ones(5)
b = a.numpy()
print(a)
print(b)
b[0]+=1
print(a)
print(b) #same memory location

a = np.ones(5)
b = torch.from_numpy(a)  #same memory problem




#CUDA

if torch.cuda.is_available():
    device = torch.device("cuda")
    d = torch.ones(5, device = device)
    e = torch.ones(5, device= device)
    f = torch.tensor(5)
    f.to(device=device)
    f = d + e
    print("\nGPU HERE")
    print(f)




#GRADIENT
x = torch.ones(5,requires_grad=True)
print(x)

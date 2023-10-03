import numpy as np

# f = w * x

# f = 2 * x

x = np.array([1,2,3,4], dtype=np.float32)
y = np.array([2,4,6,8], dtype=np.float32)

w = 0.0

#model prediction
def forwardPass(x):
    return w*x

#loss = Mean Square Error == MSE
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

# gradient
# MSE = 1/N * (w*x - y)**2
# dJ/dx = 1/N * 2x * (wx-y)*w
def gradient(x,y,y_predicted):
    return np.dot(2*x,y_predicted - y).mean()

print(f'Prediction before training: f(5) = {forwardPass(5):.3f}')

#Training

learning_rate = 0.01
n_iters = 20

for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = forwardPass(x)

    #loss 
    l = loss(y,y_pred)

    #gradients
    dw = gradient(x,y,y_pred)

    #update weights
    w -= learning_rate* dw
    
    if epoch%2==0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')


print(f'prediction after training: f(5) = {forwardPass(5):.3f}')


#ACTIVATION FUNCTIONS

import torch
import torch.nn as nn

nn.Sigmoid()            #0 to 1 binary classification
nn.ReLU()               #0, x<=0; x, x>0
nn.LeakyReLU()          #ax, x<=0, a<<1; x, x> 0
nn.Softmax()            #0 to 1 multichoise classification
nn.Tanh()               #???
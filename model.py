import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import grad

class ImageGradient(nn.Module):
    def __init__(self, pos=0):
        k = np.repeat([np.repeat([[-0.5,0,0.5]],3,axis=0)],3,axis=0)
        t = list(range(3))
        t[pos] = 0
        t[0] = pos
        k = np.transpose(k, t)

        self.kernel: torch.Tensor = torch.from_numpy(k)
        self.kernel.requires_grad = False


    def forward(self, x: torch.Tensor):
        x.requires_grad = False
        return F.conv3d(x, self.kernel)

class Hamiltonian(nn.Module):
    def __init__(self, df):

        self.gradx = ImageGradient(pos=0)
        self.grady = ImageGradient(pos=1)

        self.conv1 = nn.Conv3d(df * 6, df, 1)
        self.conv2 = nn.Conv3d(df, df, 1)
        self.conv3 = nn.Conv3d(df, 1, 1)

    def forward(self, p: torch.Tensor, q: torch.Tensor):

        dpx = self.gradx(p)
        dpy = self.grady(p)
        dqx = self.gradx(p)
        dqy = self.grady(p)

        # Compute hamiltonian

        input = torch.cat([p, dpx, dpy, q, dqx, dqy], dim=1)
        input = nn.ReLU()(self.conv1(input))
        input = nn.ReLU()(self.conv2(input))
        input = nn.ReLU()(self.conv3(input))

        return input

class HamiltonianLoss(nn.Module):

    def __init__(self):
        self.gradx = ImageGradient(pos=0)
        self.grady = ImageGradient(pos=1)
        self.gradt = ImageGradient(pos=2)

    def forward(self, ham, p, dpx, dpy, q, dqx, dqy):
        varp = grad(ham, p, create_graph=True) - self.gradx(grad(ham, dpx, create_graph=True)) - self.grady(grad(ham, dpy, create_graph=True))
        varq = grad(ham, q, create_graph=True) - self.gradx(grad(ham, dqx, create_graph=True)) - self.grady(grad(ham, dqy, create_graph=True))

        l_q = self.gradt(q) - varp
        l_p = self.gradt(p) + varq

        batch_size = l_q.size(0)

        l_q = (l_q * l_q).view(batch_size, -1).sum()
        l_p = (l_p * l_p).view(batch_size, -1).sum()

        return l_q + l_p




class HamFieldModel(nn.Module):
    def __init__(self):

        pass
    
    def forward(self, x):
        return x
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import grad

class ImageGradient(nn.Module):
    def __init__(self, in_channels, pos):
        super(ImageGradient, self).__init__()

        self.in_channels = in_channels
        k = np.zeros((self.in_channels, self.in_channels, 3,3,3), dtype=np.float32)
        arr = [-0.5,0,0.5]
        if pos == 0:
            k[:, :,1,1] = arr
        elif pos == 1:
            k[:, 1,:,1] = arr
        elif pos == 2:
            k[:, 1,1,:] = arr
        self.kernel: torch.Tensor = torch.from_numpy(k)
        self.kernel.requires_grad = False


    def forward(self, x: torch.Tensor):
        return F.conv3d(x, self.kernel, padding=1)

class Hamiltonian(nn.Module):
    def __init__(self, df):
        super(Hamiltonian, self).__init__()
        self.gradx = ImageGradient(df, pos=1)
        self.grady = ImageGradient(df, pos=2)

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

        return input, p, dpx, dpy, q, dqx, dqy

class HamiltonianLoss(nn.Module):

    def __init__(self, df):
        super(HamiltonianLoss, self).__init__()
        self.gradx = ImageGradient(df, pos=1)
        self.grady = ImageGradient(df, pos=2)
        self.gradt = ImageGradient(df, pos=0)

    def forward(self, ham, p, dpx, dpy, q, dqx, dqy):
        batch_size = ham.size(0)

        hsum = ham.view(batch_size, -1).sum()

        varp = grad(hsum, p, create_graph=True)[0] - self.gradx(grad(hsum, dpx, create_graph=True)[0]) - self.grady(grad(hsum, dpy, create_graph=True)[0])
        varq = grad(hsum, q, create_graph=True)[0] - self.gradx(grad(hsum, dqx, create_graph=True)[0]) - self.grady(grad(hsum, dqy, create_graph=True)[0])

        l_q = self.gradt(q) - varp
        l_p = self.gradt(p) + varq


        l_q = (l_q * l_q).view(batch_size, -1).sum()
        l_p = (l_p * l_p).view(batch_size, -1).sum()

        l = l_q + l_p

        return l

class Encoder(nn.Module):
    def __init__(self, df):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv3d(3, 16, 3)
        self.conv2 = nn.Conv3d(16, 16, 3)
        self.conv3 = nn.Conv3d(16, 2*df, 3)
        self.df = df
        

    def forward(self, images):
        x = nn.ReLU()(self.conv1(images))
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv3(x))

        p, q = torch.split(x, self.df, dim=1)
        
        return p, q

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, p, q):
        pass

class HamFieldModel(nn.Module):
    def __init__(self, df):
        super(HamFieldModel, self).__init__()
        self.encoder = Encoder(df)
        self.ham = Hamiltonian(df)
        self.decoder = Decoder(df)

    def forward(self, images):
        return self.ham(*self.encoder(images))
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import grad

class ImageGradient(nn.Module):
    def __init__(self, in_channels, pos):
        super(ImageGradient, self).__init__()

        self.in_channels = in_channels
        arr = [-0.5,0,0.5]
        # arr = [1./280,-4./105,1./5,-4./5,0.,4./5,-1./5,4./105,-1./280]
        self.order = len(arr)
        if pos == 0:
            k = np.zeros((self.in_channels, self.in_channels, self.order, 1, 1), dtype=np.float32)
            k[:, :, :, 0, 0] = arr
            self.padding = ((self.order - 1)//2, 0, 0)
        elif pos == 1:
            k = np.zeros((self.in_channels, self.in_channels, 1, self.order, 1), dtype=np.float32)
            k[:, :, 0, :, 0] = arr
            self.padding = (0, (self.order - 1)//2, 0)
        elif pos == 2:
            k = np.zeros((self.in_channels, self.in_channels, 1, 1, self.order), dtype=np.float32)
            k[:, :, 0, 0, :] = arr
            self.padding = (0, 0, (self.order - 1)//2)

        self.kernel = nn.Parameter(torch.from_numpy(k), requires_grad=False)


    def forward(self, x: torch.Tensor):
        return F.conv3d(x, self.kernel, padding=self.padding)

class HamPart(nn.Module):
    def __init__(self, df):
        super(HamPart, self).__init__()

        self.conv1 = nn.Conv3d(df * 3, df * 6, 1, padding=1)
        self.conv2 = nn.Conv3d(df * 6, df * 6, 1, padding=1)
        self.conv3 = nn.Conv3d(df * 6, df * 3, 1, padding=1)
        self.conv4 = nn.Conv3d(df * 3, 1, 1, padding=1)

        self.dropout1 = nn.Dropout3d()
        self.dropout2 = nn.Dropout3d()
        self.dropout3 = nn.Dropout3d()
        self.gradx = ImageGradient(df, pos=1)
        self.grady = ImageGradient(df, pos=2)

    def forward(self, x: torch.Tensor):

        dxx = self.gradx(x)
        dxy = self.grady(x)
        input = torch.cat([x, dxx, dxy], dim=1)
        # input = self.dropout1(nn.ReLU()(self.conv1(input)))
        # input = self.dropout2(nn.ReLU()(self.conv2(input)))
        # input = self.dropout3(nn.ReLU()(self.conv3(input)))
        input = (nn.ReLU()(self.conv1(input)))
        input = (nn.ReLU()(self.conv2(input)))
        input = (nn.ReLU()(self.conv3(input)))
        input = nn.ReLU()(self.conv4(input))

        return input, x, dxx, dxy


class Hamiltonian(nn.Module):
    def __init__(self, df):
        super(Hamiltonian, self).__init__()

        self.kin = HamPart(df)
        self.pot = HamPart(df)

    def forward(self, p: torch.Tensor, q: torch.Tensor):

        kinetic, p, dpx, dpy = self.kin(p)
        potential, q, dqx, dqy = self.pot(q)

        input = kinetic + potential

        return input, p, dpx, dpy, q, dqx, dqy

def variational_derivatives(gradx, grady, ham, p, dpx, dpy):
    batch_size = ham.size(0)
    hsum = ham.view(batch_size, -1).sum()

    varp = grad(hsum, p, create_graph=True)[0] - gradx(grad(hsum, dpx, create_graph=True)[0]) - grady(grad(hsum, dpy, create_graph=True)[0])

    return varp

class Integrator(nn.Module):
    def __init__(self, df):
        super(Integrator, self).__init__()
        self.gradx = ImageGradient(df, pos=1)
        self.grady = ImageGradient(df, pos=2)


    def forward(self, ham, p, dpx, dpy, q, dqx, dqy, dt):
        varp, varq = variational_derivatives(self.gradx, self.grady, ham, p, dpx, dpy, q, dqx, dqy)

        # Euler integration
        pp = p - dt * varq
        qp = q + dt * varp

        return pp, qp

class SymIntegratorP(nn.Module):
    def __init__(self, df, use_p):
        super(SymIntegratorP, self).__init__()
        self.gradx = ImageGradient(df, pos=1)
        self.grady = ImageGradient(df, pos=2)
        self.use_p = use_p

    def forward(self, ham, p, dpx, dpy, q, dqx, dqy, dt):
        # Euler integration
        if self.use_p:
            var_deriv = variational_derivatives(self.gradx, self.grady, ham, p, dpx, dpy)

            return p - dt * var_deriv, q
        else:
            var_deriv = variational_derivatives(self.gradx, self.grady, ham, q, dqx, dqy)

            return p, q + dt * var_deriv


class HamiltonianLoss(nn.Module):

    def __init__(self, df):
        super(HamiltonianLoss, self).__init__()
        self.gradx = ImageGradient(df, pos=1)
        self.grady = ImageGradient(df, pos=2)
        self.gradt = ImageGradient(df, pos=0)

    def forward(self, ham, p, dpx, dpy, q, dqx, dqy):
        batch_size = ham.size(0)
        varp, varq = variational_derivatives(self.gradx, self.grady, ham, p, dpx, dpy, q, dqx, dqy)

        l_q = self.gradt(q) - varp
        l_p = self.gradt(p) + varq

        # Collect subset window
        pad = (self.gradx.order - 1) //2

        # Actual padding is twice the padding of a single gradient kernel
        # pad *= 2

        chan = l_q.size(1)
        l_q = l_q[:, :, pad:-pad+1, pad:-pad+1, pad:-pad+1]
        l_p = l_q[:, :, pad:-pad+1, pad:-pad+1, pad:-pad+1]

        l_q = (l_q * l_q).view(batch_size, -1).sum()
        l_p = (l_p * l_p).view(batch_size, -1).sum()

        l = l_q + l_p

        return l

class Encoder(nn.Module):
    def __init__(self, df):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv3d(3, 4, 3)
        self.bn1 = nn.BatchNorm3d(4)
        self.dropout1 = nn.Dropout3d()
        self.conv2 = nn.Conv3d(4, 8, 3)
        self.bn2 = nn.BatchNorm3d(8)
        self.dropout2 = nn.Dropout3d()
        self.conv3 = nn.Conv3d(8, 2*df, 3)
        self.df = df
        

    def forward(self, images):
        x = self.dropout1(nn.ReLU()(self.bn1(self.conv1(images))))
        x = self.dropout2(nn.ReLU()(self.bn2(self.conv2(x))))
        x = nn.ReLU()(self.conv3(x))

        p, q = torch.split(x, self.df, dim=1)
        
        return p, q

class Decoder(nn.Module):
    def __init__(self, df):
        super(Decoder, self).__init__()
        self.df = df
        self.conv1 = nn.ConvTranspose3d(df*2, 4, 3)
        self.bn1 = nn.BatchNorm3d(4)
        self.dropout1 = nn.Dropout3d()
        self.conv2 = nn.ConvTranspose3d(4, 4, 3)
        self.bn2 = nn.BatchNorm3d(4)
        self.dropout2 = nn.Dropout3d()
        self.conv3 = nn.ConvTranspose3d(4, 3, 3)

    def forward(self, p, q):
        input = torch.cat([p,q], dim=1)
        input = self.dropout1(nn.ReLU()(self.bn1(self.conv1(input))))
        input = self.dropout2(nn.ReLU()(self.bn2(self.conv2(input))))
        input = nn.Sigmoid()(self.conv3(input))

        return input

class HamFieldModel(nn.Module):
    def __init__(self, df):
        super(HamFieldModel, self).__init__()
        self.encoder = Encoder(df)
        self.ham = Hamiltonian(df)
        self.decoder = Decoder(df)

    def forward(self, images):
        p, q = self.encoder(images)
        return self.ham(p, q), self.decoder(p, q)


class PredictiveHamModel(nn.Module):

    def __init__(self, df, hamiltonian_model: Hamiltonian):
        super(PredictiveHamModel, self).__init__()
        self.ham = hamiltonian_model
        self.integrator = Integrator(df)

    def forward(self, p, q, dt):
        out = self.ham(p, q)
        n = self.integrator(*out, dt)
        return n
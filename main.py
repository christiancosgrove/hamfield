import pong
from model import HamFieldModel, HamiltonianLoss
import numpy as np
from skvideo.io import vwrite
from skimage.io import imsave
import torch
from tqdm import tqdm
from torch.optim import Adam
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

class PongDataset(Dataset):
    def __init__(self, num_balls, num_frames, resolution):
        system = pong.PingPong(num_balls)

        f = system.frames(num_frames, resolution, 0.01)

        vwrite('out/out2.gif', f.transpose((1, 2, 3, 0)))

        # Subsample windows
        data = []
        for i in range(0, num_frames - window_size):
            data.append(np.expand_dims(f[:, i:i+window_size, :, :], 0))
        self.dat = np.concatenate(data, axis=0).astype(np.float32) / 255.0
        print('generated data with shape ', self.dat.shape)

    def __len__(self):
        return self.dat.shape[0]

    def __getitem__(self, i):
        return self.dat[i]

window_size = 7

def try_cuda(module):
    if torch.cuda.is_available():
        return module.cuda()
    # else:
        # print('Warning: CUDA not enabled')
    return module

def train():

    epochs = 100
    df = 4

    model = try_cuda(HamFieldModel(df))
    print('generating data')
    
    dataset = PongDataset(3, 256, 32)
    mb_size = 8
    loader = DataLoader(dataset, mb_size, shuffle=True, num_workers=4)

    
    hloss = try_cuda(HamiltonianLoss(df))

    optimizer = Adam(model.parameters(), lr=5e-3)

    for e in tqdm(range(epochs)):
        for i, batch in tqdm(enumerate(loader)):
            batch = try_cuda(batch)
            optimizer.zero_grad()

            output, decoded = model(batch)
            ham_loss = hloss(*output)
            decoder_loss = nn.MSELoss()(decoded, batch)
            loss = 1e-1*ham_loss + decoder_loss
            # loss = decoder_loss
            loss.backward()
            optimizer.step()

            def disp_tensor(t):
                im = t.cpu().detach().numpy()
                im = im[0, :, 0, :, :]
                im = np.transpose(im, (1, 2, 0))
                im *= 255
                im = im.astype(np.uint8)
                return im

        if e % 2 == 0:
            imsave(f'out/out{i}.png', disp_tensor(decoded))
            imsave(f'out/in{i}.png', disp_tensor(batch))



if __name__ == '__main__':
    train()
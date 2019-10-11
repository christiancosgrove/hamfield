import pong
from model import HamFieldModel, HamiltonianLoss
import numpy as np
from skvideo.io import vwrite
from skimage.io import imsave
import torch
from tqdm import tqdm
from torch.optim import Adam
from torch import nn

window_size = 7

def gen_data():
    system = pong.PingPong(8)

    num_frames = 16
    f = system.frames(num_frames, 122, 0.01)
    # Subsample windows
    data = []
    for i in range(0, num_frames - window_size):
        data.append(np.expand_dims(f[:, i:i+window_size, :, :], 0))
    return np.concatenate(data, axis=0).astype(np.float32) / 255.0

def train():

    iters = 100

    df = 4

    model = HamFieldModel(df)
    print('generating data')
    data = gen_data()
    print('generated data with shape ', data.shape)
    
    mb_size = 4

    optimizer = Adam(model.parameters())
    for i in tqdm(range(iters)):
        sample = np.random.choice(data.shape[0], size=mb_size)
        batch = torch.tensor(data[sample])
        optimizer.zero_grad()

        output, decoded = model(batch)
        ham_loss = HamiltonianLoss(df)(*output)
        decoder_loss = nn.MSELoss()(decoded, batch)
        loss = ham_loss + decoder_loss
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            im = decoded.cpu().detach().numpy()
            im = im[0, :, 0, :, :]
            im = np.transpose(im, (1, 2, 0))
            im *= 255
            im = im.astype(np.uint8)
            imsave(f'out/test{i}.png', im)



if __name__ == '__main__':
    train()
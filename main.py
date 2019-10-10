import pong
from model import HamFieldModel, HamiltonianLoss
import numpy as np
from skvideo.io import vwrite

window_size = 5

def gen_data():
    system = pong.PingPong(8)

    num_frames = 512


    f = system.frames(num_frames, 512, 0.01)

    # Subsample windows
    data = []
    for i in range(0, num_frames - window_size):
        data.append(np.expand_dims(f[i:i+window_size], 0))
    # print(data)
    return np.concatenate(data)

def train():
    model = HamFieldModel(8)
    data = gen_data()
    
    mb_size = 16

    while True:
        sample = np.random.choice(data.shape[0], size=mb_size)
        batch = data[sample]
        print(batch.shape)


if __name__ == '__main__':
    train()
import pong
from model import HamFieldModel

from skvideo.io import vwrite

def train():
    model = HamFieldModel(8)
    system = pong.PingPong(8)

    f = system.frames(512, 512, 0.01)

    vwrite('out/pong.gif', f)


if __name__ == '__main__':
    train()
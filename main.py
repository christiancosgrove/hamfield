import pong
from model import HamFieldModel

from skvideo.io import vwrite

def train():
    # model = HamFieldModel(8)
    system = pong.PingPong(5)

    f = system.frames(100)

    vwrite('pong.mp4', f)


if __name__ == '__main__':
    train()
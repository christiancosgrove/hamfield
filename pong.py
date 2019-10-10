import numpy as np
import skimage

class PhysicsSystem(object):
    def step(self, timestamp):
        raise NotImplementedError

    def render(self) -> np.array:
        raise NotImplementedError

    def frames(self, count):
        frames = []
        for _ in range(count):
            self.step()
            frames.append(self.render())
        return np.array(frames)


class PingPong(PhysicsSystem):
    def __init__(self, num_balls, ball_size=0.05):
        pos = np.random.uniform(0, 1, size=(2, num_balls))
        vel = np.random.uniform(-1, 1, size=(2, num_balls))
        super()


    def step(self, timestamp):
        pass
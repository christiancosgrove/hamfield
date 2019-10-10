import numpy as np
from skimage.draw import circle

class PhysicsSystem(object):
    def step(self, timestep):
        raise NotImplementedError

    def render(self, resolution: int) -> np.array:
        raise NotImplementedError

    def frames(self, count: int, resolution: int, timestep=np.float) -> np.array:
        frames = []
        for _ in range(count):
            self.step(timestep)
            frames.append(self.render(resolution))
        return np.array(frames)


class PingPong(PhysicsSystem):
    def __init__(self, num_balls, ball_size=0.05):
        self.pos = np.random.uniform(0, 1, size=(2, num_balls))
        self.vel = np.random.uniform(-1, 1, size=(2, num_balls))

        self.num_balls = num_balls
        super()


    def step(self, timestep):
        self.pos += self.vel * timestep

    def render(self, resolution: int) -> np.array:
        frame = np.ones((resolution, resolution), dtype=np.uint8) * 255

        for i in range(self.num_balls):
            circ = circle(resolution * self.pos[0,i], resolution * self.pos[1,i], 10, shape=frame.shape)
            frame[circ] = 0
        return frame


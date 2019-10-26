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
            frames.append(np.expand_dims(self.render(resolution), axis=1))
        c = np.concatenate(frames, axis=1)
        return c


class PingPong(PhysicsSystem):
    def __init__(self, num_balls, ball_size=0.2):

        self.pos = np.random.uniform(0, 1, size=(2, num_balls))
        self.vel = np.random.uniform(-1, 1, size=(2, num_balls))

        self.num_balls = num_balls
        self.radius = ball_size

        self.fix()

    def fix(self):
        # Check collisions
        for i in range(self.num_balls):
            for j in range(i):
                if j == i:
                    continue
                diff = self.pos[:, i] - self.pos[:, j]
                if ((diff)**2).sum() < (2*self.radius) ** 2:
                    self.pos[:, i] += 2*1.01*self.radius * diff / np.linalg.norm(diff) - diff

    def step(self, timestep):
        # Check collisions
        for i in range(self.num_balls):
            for j in range(i):
                if j == i:
                    break
                diff = self.pos[:, i] - self.pos[:, j]
                if ((diff)**2).sum() < (2*self.radius) ** 2:
                    # self.pos[:, i] += diff/2
                    # self.pos[:, j] -= diff/2

                    normdiff = diff/np.linalg.norm(diff)

                    self.vel[:, i] += -2*np.dot(self.vel[:, i], normdiff) * normdiff 
                    self.vel[:, j] += -2*np.dot(self.vel[:, j], normdiff) * normdiff

        # Collisions with walls
        for i in range(self.num_balls):
            if self.pos[0, i] < self.radius:
                self.pos[0, i] = self.radius
                self.vel[0, i] *= -1
            if 1 - self.pos[0, i] < self.radius:
                self.pos[0, i] = 1 - self.radius
                self.vel[0, i] *= -1
        for i in range(self.num_balls):
            if self.pos[1, i] < self.radius:
                self.pos[1, i] = self.radius
                self.vel[1, i] *= -1
            if 1 - self.pos[1, i] < self.radius:
                self.pos[1, i] = 1 - self.radius
                self.vel[1, i] *= -1
        
        self.pos += self.vel * timestep
        self.fix()

    def render(self, resolution: int) -> np.array:
        frame = np.ones((resolution, resolution, 3), dtype=np.uint8) * 255

        for i in range(self.num_balls):
            rr, cc = circle(resolution * self.pos[0,i], resolution * self.pos[1,i], self.radius * resolution, shape=frame.shape)
            frame[rr, cc, :] = 0
        return np.transpose(frame, (2, 0, 1))


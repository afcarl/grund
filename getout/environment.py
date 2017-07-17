import numpy as np

from ..abstract import EnvironmentBase


class Entity:

    def __init__(self, coords, color=1):
        self.coords = coords
        self.color = color

    def move(self, vector):
        self.coords += np.array(vector)

    def touches(self, other):
        return np.all(self.coords == other.coords)


class GetOut(EnvironmentBase):

    def __init__(self, size):
        self.exit = None
        self.player = None
        self.size = np.array(size)
        self.canvas = np.zeros(size)
        self.actions = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1), (0, 1),
                        (1, -1), (1, 0), (1, 1)]
        self.steps = 0

    def neurons_required(self):
        return tuple(self.size), len(self.actions)

    def escaping(self, player: Entity):
        return np.any(player.coords < 0) or np.any(player.coords >= self.size-1)

    def draw(self):
        self.canvas = np.zeros(self.size)
        self.canvas[tuple(self.player.coords)] = self.player.color
        self.canvas[tuple(self.exit.coords)] = self.exit.color

    def reset(self):
        self.steps = 0

        xcoords = (np.random.uniform(size=2) * self.size).astype(int)
        rndax = int(np.random.uniform() < 0.5)
        xcoords[rndax] = np.random.choice([0, (self.size-1)[rndax]])
        pcoords = self.size // 2

        self.exit = Entity(xcoords, color=-1)
        self.player = Entity(pcoords, color=1)

        self.draw()
        return self.canvas

    def step(self, action):
        self.steps += 1
        self.player.move(self.actions[action])
        self.draw()
        reward = 0.
        done = False
        if self.escaping(self.player):
            reward = -1.
            done = True
        if self.player.touches(self.exit):
            reward = -1.
            done = True
        if self.steps > np.prod(self.size) * 2:
            reward = -1.
            done = True
        return self.canvas, reward, done

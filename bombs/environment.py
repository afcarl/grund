from collections import defaultdict

import numpy as np

from grund.bombs.classes import Field, Flame, Bomb, Wall, Player
from grund.util.abstract import EnvironmentBase


class Environment(EnvironmentBase):

    def __init__(self, size=5):
        self.size = size
        self.map = defaultdict(list)
        self._build_map()

    def asarray(self):
        ar = np.zeros((self.size, self.size), dtype=int)
        for field in self.map.values():
            ar[tuple(field[0].position)] = field[0].sign
        return ar

    def _build_map(self):
        for i in range(1, self.size, 2):
            for j in range(1, self.size, 2):
                Wall.create_and_add(self, (i, j))
        Player.create_and_add(self, (0, 0), sign=1)
        Player.create_and_add(self, (self.size-1, self.size-1), sign=2)

    def at(self, position):
        position = tuple(position)
        return self.map.get(position, Field(self, position))

    @property
    def neurons_required(self):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass


"""
TODO:
Somehow handle the fact that multiple entities may occupy the same field.
In reality this might only happen with Player + Bomb on Bomb laying. 
"""


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    env = Environment()
    plt.imshow(env.asarray())
    plt.show()

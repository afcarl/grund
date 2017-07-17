import numpy as np


class Snake:

    def __init__(self, coords, color):
        self.coords = coords
        self.color = color
        self.body = []
        self.direction = 1
        self.vector = {1: np.array([1, 0]),
                       2: np.array([0, 1]),
                       3: np.array([-1, 0]),
                       4: np.array([0, -1]),
                       0: np.array([0, 0])}
        self.opposite = {1: 3, 3: 1, 2: 4, 4: 2, 0: None}

    def move(self, direction):
        if direction == self.direction or \
                self.opposite[direction] == self.direction or \
                not direction:
            direction = self.direction
        else:
            self.direction = direction
        self.body.append(tuple(self.coords))
        newcoords = self.coords + self.vector[direction]
        self.coords = newcoords

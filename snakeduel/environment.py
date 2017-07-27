import numpy as np

from ..abstract import EnvironmentBase
from .classes import Snake


class SnakeDuel(EnvironmentBase):

    def __init__(self, size=(400, 300)):
        self.size = np.array(size)
        self.canvas = np.zeros(size, dtype=int)
        self.snakes = []
        self.actions = (0, 1, 2, 3)

    def escaping(self, snake: Snake):
        return np.any(snake.coords < 0) or np.any(snake.coords >= self.size)

    def suicide(self, snake: Snake):
        return tuple(snake.coords) in snake.body

    def draw(self, snake):
        self.canvas[tuple(snake.coords)] = snake.color

    def step(self, action):
        s = self.snakes[0]  # type: Snake
        s.move(action)
        if self.escaping(s) or self.suicide(s):
            return self.canvas, -1., 1
        self.draw(s)
        return self.canvas, 0., 0

    def reset(self):
        mid = self.size // 2
        coords1 = mid.copy()
        coords1[0] //= 2
        coords2 = mid.copy()
        coords2[0] += mid[0] // 2
        self.canvas = np.zeros(self.size)
        self.snakes = [Snake(coords1, 1)]
        list(map(self.draw, self.snakes))
        return self.canvas

    @property
    def neurons_required(self):
        return self.size, len(self.actions)
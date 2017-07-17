import numpy as np
import pygame


ENEMYSPEED = 5


class _EntityBase:

    def __init__(self, game, color, size, coords=None):
        self.game = game
        self.coords = np.array([-50, -50])
        self.color = color
        self.size = size
        if coords is not None:
            self.teleport(coords)

    def adjust_coordinates(self):
        self.coords[self.coords < self.size] = self.size
        toobig = self.coords > (self.game.size - self.size)
        self.coords[toobig] = self.game.size[toobig] - self.size

    def draw(self):
        pygame.draw.circle(self.game.screen, self.color, self.coords, self.size)
        return self

    def move(self, dvec):
        self.coords += dvec.astype(int)
        self.adjust_coordinates()

    def distance(self, other):
        return np.linalg.norm(self.coords - other.coords)

    def touches(self, other):
        return self.distance(other) <= (self.size + other.size)

    def escaping(self):
        r = self.size
        P = self.coords
        return np.any(P <= r) or np.any(P >= self.game.size - r)

    def teleport(self, destination=None):
        if destination is None:
            destination = (np.random.uniform(0.05, 0.95, 2) * self.game.size).astype(int)
        self.coords = destination
        self.adjust_coordinates()


class EnemyBall(_EntityBase):

    def __init__(self, game, color, size):
        self._move_generator = self._automove()
        super().__init__(game, color, size)
        self.hori = np.random.uniform(size=1)[0] < 0.5
        self.teleport()
        while self.distance(game.player) < game.meandist / 2:
            self.teleport()

    def _automove(self):
        d = np.array([ENEMYSPEED]*2)
        d[int(self.hori)] = 0
        if np.random.randn() < 0:
            d *= -1
        while 1:
            if self.escaping():
                d *= -1
            yield d

    def move(self, dvec=None):
        super().move(next(self._move_generator))


class Square(_EntityBase):

    def __init__(self, game, color, size):
        super().__init__(game, color, size*2)
        self.teleport()

    def draw(self):
        adjust = self.size // 2
        pygame.draw.rect(
            self.game.screen, self.color,
            pygame.Rect(self.coords-adjust, [self.size]*2)
        )
        return self


class PlayerBall(_EntityBase):

    def __init__(self, game, color, size):
        super().__init__(game, color, size, game.size // 2)

    def dead(self):
        return any(self.touches(other) for other in self.game.enemies)

import numpy as np

from ..util.movement import directions


class Field:

    sign = 0

    def __init__(self, environment, position=None, sign=None):
        self.sign = sign if sign is not None else self.__class__.sign
        self.position = position if position is not None else np.array([-1, -1])
        self.environment = environment

    @classmethod
    def create_and_add(cls, environment, position, sign=None):
        assert environment.at(position).sign == Field.sign
        entity = cls(environment, np.array(position), sign if sign is not None else cls.sign)
        environment.map[tuple(position)] = entity

    @property
    def coords(self):
        return tuple(self.position)


class Wall(Field):
    sign = 5


class Flame(Field):

    sign = 4

    def __init__(self, environment, position, timeleft=2):
        super().__init__(environment, position)
        self.timeleft = timeleft

    def tick(self):
        self.timeleft -= 1


class Bomb(Field):

    sign = 3

    def __init__(self, environment, position, timeleft=4, radius=3):
        super().__init__(environment, position)
        self.timeleft = timeleft
        self.radius = radius
        self.triggered = False

    def tick(self):
        self.timeleft -= 1
        self.triggered = not bool(self.timeleft)

    def expode(self):
        explosion = []
        for d in directions:
            for step in range(1, self.radius):
                coord = self.position + d * step
                field = self.environment.grund[coord[0], coord[1]]
                if field == Bomb.sign:
                    self.environment
                explosion.append(Flame(self.environment, coord))
                if field == Bomb.sign:
                    pass


class Player(Field):

    def __init__(self, environment, position, sign):
        super().__init__(environment, position, sign)
        self.alive = True

    def move(self, direction):
        tposition = self.position + directions[direction]
        tfield = self.environment.at(tposition)
        if tfield.sign in blockers:
            return
        self.position = tposition


blockers = {Bomb.sign, Wall.sign}

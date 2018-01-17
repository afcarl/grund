from ..util.movement import vectors


class Snake:

    def __init__(self, coords, color):
        self.coords = coords
        self.color = color
        self.body = []
        self.direction = 1
        self.opposite = {1: 3, 3: 1, 2: 4, 4: 2, 0: None}

    def move(self, direction):
        if direction == self.direction or \
                self.opposite[direction] == self.direction or \
                not direction:
            direction = self.direction
        else:
            self.direction = direction
        self.body.append(tuple(self.coords))
        newcoords = self.coords + vectors[direction]
        self.coords = newcoords

    @property
    def suicide(self):
        return tuple(self.coords) in self.body

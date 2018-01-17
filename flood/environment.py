import numpy as np

from util.abstract import EnvironmentBase


class Flood(EnvironmentBase):

    def __init__(self, size=(10, 10), ncolors=4):
        self.ncolors = ncolors
        self.size = size
        self.playground = None
        self.watchlist = None
        self.flooded = None
        self.moves = 0
        self.points = 0

    def update_watchlist(self, action):
        pass

    def watch_point(self, xy, action):
        watchme = set()
        for neighbour in self.get_neighbours_of(xy):
            if self.playground[xy] == action:
                watchme.add()

    def step(self, action):
        if action > self.ncolors:
            raise ValueError("Invalid action! Action space spans {} -- {}"
                             .format(0, self.ncolors))
        current = self.playground[0, 0]
        if current == action:
            return self.playground, self.reward, self.done
        newcells = set()
        for xy in self.flooded.union(self.watchlist):
            self.playground[xy] = action
        watchme = set()
        for xy in self.watchlist:
            nbh = self.get_neighbours_of(xy)
            if all(self.playground[coord] == current for coord in nbh):
                self.flooded.add(xy)
                self.watchlist.remove(xy)
                continue
            newcells.update(
                {coord for coord in self.get_neighbours_of(xy)
                 if self.playground[coord] == action}
            )
        for xy in self.watchlist:
            self.playground[xy] = action
        return self.playground, self.reward, self.done

    @property
    def done(self):
        return len(self.watchlist) == self.playground.size

    @property
    def reward(self):
        return len(self.watchlist)

    def reset(self):
        self.playground = np.random.randint(1, self.ncolors+1, self.size, dtype=int)
        self.moves = 0
        self.points = 0
        self.watchlist = {(0, 0)}
        self.flooded = set()
        return self.playground

    def get_neighbours_of(self, xy):
        x, y = xy
        if x == 0:
            if y == 0:
                return [(x+1, y), (x, y+1)]
            elif y == self.size[1]:
                return [(x+1, y), (x, y-1)]
            else:
                return [(x+1, y), (x, y-1), (x, y+1)]
        if x == self.size[0]:
            if y == 0:
                return [(x-1, y), (x, y+1)]
            elif y == self.size[1]:
                return [(x-1, y), (x, y-1)]
            else:
                return [(x-1, y), (x, )]
        return [(x+1, y), (x, y+1),
                (x-1, y), (x, y-1)]


def _event_stream(plt):
    pressed = None
    while 1:
        if not pressed:
            plt.pause(0.5)
            continue
        yield pressed
        pressed = None


def flood_mainloop(agent, game: Flood):
    from matplotlib import pyplot as plt
    plt.ion()
    fig, ax = plt.subplots()
    screen = ax.imshow(game.reset(), cmap="hot", vmin=1, vmax=game.ncolors)
    fig.canvas.mpl_connect("key_press_event", _keypress_handler)

    state = game.reset()
    screen.set_data(state)
    plt.pause(0.1)

    events = _event_stream()
    for action in events:
        state, reward, done = game.step(action)
        screen.set_data(state)
        plt.pause(0.1)
        if done:
            break

from ._ball import *
from ._util import (
    calc_meand, downsample
)
from util.abstract import EnvironmentBase

BLUE = (0, 0, 200)
DARK_GREY = (50, 50, 50)
LIGHT_GREY = (100, 100, 100)


class ReSkiv(EnvironmentBase):

    def __init__(self, fps, screensize, state="statistics",
                 playersize=10, enemysize=5, squaresize=10,
                 playercolor=DARK_GREY, enemycolor=BLUE, squarecolor=LIGHT_GREY,
                 downsmpl=True, headless=False):

        pygame.init()
        self.ballargs = {
            "player": (self, playercolor, playersize),
            "enemy": (self, enemycolor, enemysize),
            "square": (self, squarecolor, squaresize)
        }
        self.size = np.array(screensize, dtype=int)
        self.meandist = calc_meand(self.size)
        self.maxdist = np.linalg.norm(self.size)
        if headless:
            pygame.display.set_mode((1, 1), 0, 32)
            self.screen = pygame.Surface(self.size)
        else:
            self.screen = pygame.display.set_mode(self.size)
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.player = None
        self.square = None
        self.enemies = None
        self.points = 0.
        self.nopoints = 0
        self.steps_taken = 0
        self.episodes = 0
        self.actions = [(-1, -1), (-1, 0), (-1, 1),
                        ( 0, -1), ( 0, 0), ( 0, 1),
                        (+1, -1), (+1, 0), (+1, 1)]
        self.labels = np.eye(len(self.actions))
        if state not in ("statistics", "pixels", "proximity"):
            raise RuntimeError("State should be one of: [statistics, pixels, proximity]")
        self._state_str = state
        self.state = {"pixels": self.pixels,
                      "statistics": self.statistics,
                      "proximity": self.proximity}[state]
        prxs = ((playersize*4*2)**2) // (4 if downsmpl else 1)
        self.data_shape = {
            "pixels": [1] + list(self.size // (4 if downsmpl else 1)),
            "statistics": (5,),
            "proximity": (prxs,)}[state]
        self.downsample = downsmpl

    def sample_action(self, prob):
        arg = np.random.choice(np.arange(len(self.actions)), size=1, p=prob)[0]
        actions = self.actions[arg]
        labels = self.labels[arg]
        return actions, labels

    def pixels(self):
        state = pygame.surfarray.array3d(self.screen)
        if self.downsample:
            state = downsample(state)
        return state

    def statistics(self):
        pcoords = self.player.coords / self.size
        scoords = self.square.coords / self.size
        enemies = min(self.player.distance(enemy) / self.maxdist
                      for enemy in self.enemies)
        return np.array(list(pcoords) + list(scoords) + [enemies])

    def proximity(self):
        proxysz = self.ballargs["player"][-1]*4
        pcoords = self.player.coords
        snw = np.maximum(proxysz - pcoords, 0)
        sse = np.maximum(pcoords + proxysz - self.size, 0)
        px0, py0 = pcoords - (proxysz - snw)
        px1, py1 = pcoords + (proxysz - sse)
        proxy = pygame.surfarray.array3d(self.screen)[px0:px1, py0:py1, 2].astype(float)
        proxy /= 255.
        state = np.pad(proxy, ((snw[0], sse[0]), (snw[1], sse[1])),
                       mode="constant", constant_values=-1.)
        return state[::4, ::4] if self.downsample else state

    def reset(self):
        self.player = PlayerBall(*self.ballargs["player"]).draw()
        self.square = Square(*self.ballargs["square"]).draw()
        self.enemies = [EnemyBall(*self.ballargs["enemy"]).draw() for _ in range(1)]
        self.points = 0.
        return self.step(np.array([0, 0]))[0]

    def score(self):
        scbool = self.player.touches(self.square)
        if scbool:
            self.square = Square(*self.ballargs["square"])
            self.enemies.append(EnemyBall(*self.ballargs["enemy"]))
            self.points += 5.

        return scbool

    def step(self, dvec):

        self.screen.fill((0, 0, 0))
        # steep_hills(self)
        # hills = pygame.pixelcopy.make_surface(steep_hills(self))
        # self.screen.blit(hills, (0, 0))

        self.square.draw()
        self.player.move(dvec)
        self.player.draw()

        reward, done = self.reward_function()

        for e in self.enemies:
            e.move()
            e.draw()
        return self.state(), reward, done

    def progression(self, tock):
        self.clock.tick(tock)
        pygame.display.flip()

    @staticmethod
    def check_quit():
        if any(e.type == pygame.QUIT for e in pygame.event.get()):
            return True
        return False

    def reward_function(self):
        done = 0
        rwd = 0.
        if self.score():
            rwd = 9.
        if self.player.dead():
            done = 1
            rwd = -2.
        if self.player.escaping():
            rwd = -2.
        return rwd, done


class NoEnemy(ReSkiv):

    def score(self):
        scbool = self.player.touches(self.square)
        if scbool:
            self.square = Square(*self.ballargs["square"])
            self.points += 5
        return scbool

    def reset(self):
        self.player = PlayerBall(*self.ballargs["player"]).draw()
        self.square = Square(*self.ballargs["square"]).draw()
        self.enemies = []
        self.points = 0.
        return self.step(np.array([0, 0]))[0]

    def reward_function(self):
        rwd = 0.
        if self.score():
            rwd = 1.
        if self.player.escaping():
            rwd = -1.
        return rwd, 0


class NoSquare(ReSkiv):

    def score(self):
        scbool = self.steps_taken % 200 == 0 and self.steps_taken > 0
        if scbool:
            self.enemies.append(EnemyBall(*self.ballargs["enemy"]))
            self.points += 5
        return scbool

    def reset(self):
        self.player = PlayerBall(*self.ballargs["player"]).draw()
        self.enemies = [EnemyBall(*self.ballargs["enemy"]) for _ in range(1)]
        self.points = 0.
        return self.step(np.array([0, 0]))[0]

    def reward_function(self):
        if self.score():
            return 1, 0
        if self.player.dead():
            return -1, 1
        return 0, 0

    square = type("", (object,), {"draw": lambda: None})()

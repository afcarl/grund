from scipy import ndimage
from matplotlib import pyplot as plt


class Rendered:

    def __init__(self, environment):
        self.zoom = 50
        self.environment = environment

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.screen = self.ax.imshow(self.resample(environment.reset()), cmap="hot",
                                     vmin=1, vmax=environment.ncolors)
        self.fig.canvas.mpl_connect("key_press_event", self._keypress_handler)
        self.pressed = None

    def _keypress_handler(self, event):
        try:
            kp = int(event.key)
        except Exception as E:
            print("Caught:", E)
            self.pressed = None
        else:
            if kp in (1, 2, 3, 4):
                print("KP is", kp)
                self.pressed = kp
            else:
                self.pressed = None

    def resample(self, matrix):
        # return ndimage.zoom(matrix, self.zoom, order=0)
        return matrix

    def display(self, matrix):
        self.screen.set_data(self.resample(matrix))
        plt.pause(0.1)

    def event_stream(self):
        while 1:
            if not self.pressed:
                plt.pause(0.5)
                continue
            print("Keypress:", self.pressed)
            yield self.pressed
            self.pressed = None

    def play(self):
        state = self.environment.reset()
        r = 1
        self.display(state)
        events = self.event_stream()
        for action in events:
            print("ROUND", r, "ACTION:", action)
            state, reward, done = self.environment.step(action)
            self.display(state)
            r += 1
            if done:
                break

    def teardown(self):
        plt.ioff()


if __name__ == '__main__':
    from grund.flood import Flood
    Rendered(Flood((10, 10))).play()

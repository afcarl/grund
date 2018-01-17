import numpy as np

vectors = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]
)
directions = dict(zip([None, "left", "up", "right", "down"], vectors))
U, D, L, R = directions["up"], directions["down"], directions["left"], directions["right"]

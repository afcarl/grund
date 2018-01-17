import numpy as np
from matplotlib import pyplot as plt

from brainforge import BackpropNetwork
from brainforge.layers import DenseLayer, Flatten
from brainforge.reinforcement import DQN


def simulation(game, render=False):
    inshape, outshape = game.neurons_required
    ann = BackpropNetwork(input_shape=np.prod(inshape), layerstack=[
        Flatten(),
        DenseLayer(300, activation="tanh"),
        DenseLayer(outshape, activation="softmax")
    ], cost="xent", optimizer="rmsprop")
    agent = DQN(ann, outshape)

    if render:
        plt.ion()
        obj = plt.imshow(game.reset(), vmin=-1, vmax=1, cmap="hot")

    episode = 1

    while 1:
        print()
        print(f"Episode {episode}")
        canvas = game.reset()
        if render:
            obj.set_data(canvas)
        step = 0
        done = 0
        reward = None
        while not done:
            action = agent.sample(canvas, reward)
            canvas, reward, done = game.step(action)
            if render:
                obj.set_data(canvas)
            step += 1
            # print(f"\rStep: {step}", end="")
            if render:
                plt.pause(0.1)
        print(f" Accumulating! Steps taken: {step}, {'died' if reward < 0 else 'alive'}")
        agent.accumulate(canvas, reward)
        if episode % 10 == 0:
            print("Updating!")
        episode += 1


if __name__ == '__main__':
    from grund import snakeduel
    environment = snakeduel.SnakeDuel((40, 40))
    simulation(environment, render=False)
    print()

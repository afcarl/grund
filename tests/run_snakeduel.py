import numpy as np
from matplotlib import pyplot as plt

from brainforge import Network
from brainforge.layers import DenseLayer, Flatten
from brainforge.reinforcement import DQN


def simulation(game):
    inshape, outshape = game.neurons_required()
    ann = Network(np.prod(inshape), layers=[
        Flatten(),
        DenseLayer(300, activation="tanh"),
        DenseLayer(outshape, activation="softmax")])
    ann.finalize("xent", "adam")
    agent = DQN(ann, outshape)

    plt.ion()
    obj = plt.imshow(game.reset(), vmin=-1, vmax=1, cmap="hot")

    episode = 1

    while 1:
        print()
        print(f"Episode {episode}")
        canvas = game.reset()
        obj.set_data(canvas)
        step = 0
        done = 0
        reward = None
        while not done:
            action = agent.sample(canvas, reward)
            canvas, reward, done = game.step(action)
            obj.set_data(canvas)
            step += 1
            print(f"\rStep: {step}", end="")
            plt.pause(0.1)
        print(f" Accumulating! Steps taken: {step}, {'died' if reward < 0 else 'alive'}")
        agent.accumulate(reward)
        if episode % 10 == 0:
            print("Updating!")
            agent.update()
        episode += 1

if __name__ == '__main__':
    from grund import getout
    environment = getout.GetOut((50, 50))
    simulation(environment)
    print()

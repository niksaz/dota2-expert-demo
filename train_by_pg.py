#!/usr/bin/env python3
import pickle
import time

from scipy import signal

from policy_gradient import PGAgent, ReplayBuffer
from dotaenv import DotaEnvironment
import numpy as np
from policy_gradient import replay_util
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger('DotaRL')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/policy_gradient{time}.log'.format(time=time.strftime("%Y%m%d-%H%M%S")))
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def create_dota_agent():
    return PGAgent(environment=DotaEnvironment,
                   episodes=3000,
                   batch_size=200,
                   eps=0.99,
                   discount=0.3,
                   eps_update=0.999)


def main():
    agent = create_dota_agent()
    # agent.train_on_replay(epochs=100000, batch_size=1000)
    agent.train()
    # plot()


def plot():
    with open('saved_rewards', 'rb') as output_file:
        reward = pickle.load(output_file)

    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x) + np.random.random(100) * 0.8

    def smooth(y, box_pts):
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    rew = map(lambda x: 1e3 if x > 1e8 else x, reward)
    reward = np.array(list(rew))
    arr = []
    for i in range(len(reward) // 200):
        arr.append(np.mean(reward[i * 200: (i + 1) * 200]))
    arr = np.array(arr)
    plt.figure(dpi=300)
    plt.plot(smooth(arr, 3))
    plt.title('Average reward per episode')
    plt.xlabel('Episode')
    plt.ylabel('Discounted reward')
    plt.savefig('graph.png')
    plt.show()


if __name__ == '__main__':
    main()

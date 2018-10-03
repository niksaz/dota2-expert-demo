import pickle
import numpy as np

import matplotlib.pyplot as plt


def plot():
    with open('saved_rewards', 'rb') as output_file:
        reward = pickle.load(output_file)

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
    plot()

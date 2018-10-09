import pickle

import matplotlib.pyplot as plt


def plot_saved_rewards():
    with open('saved_rewards.pkl', 'rb') as output_file:
        reward = pickle.load(output_file)

    non_zero_rewards = []
    for r in reward:
        if r != 0:
            non_zero_rewards.append(r)
    print('Number of non-zero rewards is', len(non_zero_rewards))

    plt.plot(non_zero_rewards)
    plt.title('Reward per episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('graph.png')
    plt.show()


if __name__ == '__main__':
    plot_saved_rewards()

import pickle
import argparse

import matplotlib.pyplot as plt


def plot_saved_rewards(rewards_file):
    with open(rewards_file, 'rb') as output_file:
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


def main():
    parser = argparse.ArgumentParser(description='Plot the episode rewards.')
    parser.add_argument('rewards_file', type=str,
                        help='a path to the rewards file')
    args = parser.parse_args()
    plot_saved_rewards(args.rewards_file)


if __name__ == '__main__':
    main()

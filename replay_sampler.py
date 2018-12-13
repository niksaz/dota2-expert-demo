# Author: Mikita Sazanovich

from dotaenv import DotaEnvironment
from dotaenv.codes import ATTACK_CREEP
import pickle
import argparse
import os


def record(filename):
    env = DotaEnvironment()

    state = env.reset()
    states = [state]
    done = False
    while not done:
        state, reward, done = env.execute(action=ATTACK_CREEP)
        states.append(state)

    with open(filename, 'wb') as output_file:
        pickle.dump(states, output_file)


def print_out(filename):
    with open(filename, 'rb') as input_file:
        states = pickle.load(input_file)

    for state in states:
        print(state)
    print(len(states))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('replay_name')
    parser.add_argument('--record', action='store_true', help='Records your actions in the game')
    parser.add_argument('--print', action='store_true', help='Print the recorded actions')
    args = parser.parse_args()

    filename = os.path.join('replays/', args.replay_name)

    if args.record:
        record(filename)
    if args.print:
        print_out(filename)


if __name__ == '__main__':
    main()

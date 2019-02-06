# Author: Mikita Sazanovich

from dotaenv import DotaEnvironment
from dotaenv.codes import ATTACK_CREEP
import pickle
import argparse
import os
import numpy as np


def record(filename):
    env = DotaEnvironment()

    env.reset()
    state_action_pairs = []
    done = False
    while not done:
        pairs = env.step(action=ATTACK_CREEP)
        for _, (state, _, done, info) in pairs:
            state_action_pairs.append((state, info))
    print('Frames recorded:', len(state_action_pairs))

    filtered = []
    last_state = None
    for state, info in state_action_pairs:
        if last_state is not None and np.linalg.norm(last_state - state) == 0:
            continue
        last_state = state
        filtered.append((state, info))
    print('After filtering:', len(filtered))

    with open(filename, 'wb') as output_file:
        pickle.dump(filtered, output_file)


def print_out(filename):
    with open(filename, 'rb') as input_file:
        state_action_pairs = pickle.load(input_file)

    for state, info in state_action_pairs:
        print(state, info)
    print(len(state_action_pairs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('replay_name')
    parser.add_argument('--record', action='store_true', help='Records your actions in the game')
    parser.add_argument('--print', action='store_true', help='Print the recorded actions')
    args = parser.parse_args()

    replays_folder = 'replays-action'
    if not os.path.exists(replays_folder):
        os.makedirs(replays_folder)
    filename = os.path.join(replays_folder, args.replay_name)

    if args.record:
        record(filename)
    if args.print:
        print_out(filename)


if __name__ == '__main__':
    main()

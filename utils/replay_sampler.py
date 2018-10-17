# Author: Mikita Sazanovich

from dotaenv import DotaEnvironment
import pickle
import math
import numpy as np

filename = 'replays/5.pickle'


def record():
    GOAL = np.array([-1543.998535, -1407.998291])
    env = DotaEnvironment()

    state = env.reset()
    states = [state]
    done = False
    while not done:
        next_state, reward, done = env.execute(action=12)
        states.append(next_state)
        next_pos = np.array(next_state[:2])
        if np.linalg.norm(next_pos - GOAL) < 500:
            done = True

    with open(filename, 'wb') as output_file:
        pickle.dump(states, output_file)


def print_out():
    with open(filename, 'rb') as input_file:
        states = pickle.load(input_file)

    cnt = 0
    last_state = np.array([0, 0, 0])
    for state in states:
        if np.all(state == last_state):
            continue
        cnt += 1
        diff = np.array(state[:2]) - np.array(last_state[:2])
        print(state, 'diff is', diff)
        angle_pi = math.atan2(diff[1], diff[0])
        if angle_pi < 0:
            angle_pi += 2 * math.pi
        print(angle_pi / math.pi * 180)
        last_state = state
    print('Overall cnt is', cnt)

record()
print_out()

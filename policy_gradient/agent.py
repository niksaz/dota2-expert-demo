import logging
import pickle
import random

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from policy_gradient.analyze_model import print_network_weights
from policy_gradient.network import Network
from policy_gradient.replay_buffer import ReplayBuffer
from dotaenv.codes import STATE_DIM, ACTIONS_TOTAL
from deepq.reward_shaper import StateReplayRewardShaper
from deepq.state_preprocessor import StatePreprocessor

logger = logging.getLogger('DotaRL.PGAgent')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

input_shape = STATE_DIM
output_shape = ACTIONS_TOTAL


class PGAgent:
    """
    Policy gradient agent.
    """
    __slots__ = ('env',
                 'replay_buffer',
                 'network',
                 'episodes',
                 'discount',
                 'batch_size',
                 'eps_update',
                 'eps',
                 'total_rewards')

    def __init__(self, environment, episodes=100, batch_size=100, eps=0.7,
                 discount=0.99, eps_update=0.99, restore=False):
        self.replay_buffer = ReplayBuffer()
        self.network = Network(input_shape=input_shape,
                               output_shape=output_shape,
                               restore=restore)
        self.env = environment()
        self.episodes = episodes
        self.batch_size = batch_size
        self.eps = eps
        self.discount = discount
        self.eps_update = eps_update
        if restore:
            self.replay_buffer.load_data()
            with open('saved_rewards.pkl', 'rb') as input_file:
                self.total_rewards = pickle.load(input_file)
        else:
            self.total_rewards = []

    def show_performance(self):
        for episode in range(self.episodes):
            # sample data
            states, actions, rewards, next_states, terminal = \
                self.sample_episode(batch_size=self.batch_size, eps=0.00)
            rewards = np.array(rewards, dtype='float32')

            temp = 'Finished episode {ep} with total reward {rew}. eps={eps}'
            logger.debug(temp.format(ep=episode, rew=np.sum(rewards),
                                     eps=self.eps))

    def train(self):
        reward_shaper = StateReplayRewardShaper('replays/')
        reward_shaper.load()

        episode_rewards = []
        for episode in range(self.episodes):
            # sample data
            states, actions, next_states, rewards, terminal = \
                self.sample_episode(batch_size=self.batch_size, eps=self.eps)
            episode_rewards.extend(rewards)

            if terminal:
                disc_rewards = self.disc_rewards(episode_rewards)
                episode_rewards = []
                total_reward = np.sum(disc_rewards)

                if total_reward == 0:
                    # The game was restarted right after it started
                    continue
                self.total_rewards.append(total_reward)
                with open('saved_rewards.pkl', 'wb') as output_file:
                    pickle.dump(obj=self.total_rewards, file=output_file)

            rewards = np.array(rewards, dtype='float32')
            temp = 'Finished episode {ep} with total reward {rew}. eps={eps}'
            logger.debug(temp.format(ep=episode, rew=np.sum(rewards), eps=self.eps))

            # Potential-based reward shaping from the demo
            for i in range(len(states)):
                rewards[i] += (
                        self.discount * reward_shaper.get_state_potential(next_states[i]) -
                        reward_shaper.get_state_potential(states[i]))

            # Discount rewards
            disc_rewards = self.disc_rewards(rewards)

            # Extend replay buffer with sampled data
            self.replay_buffer.extend(zip(states, actions, disc_rewards))

            # Update the parameter for epsilon-greedy strategy
            self.eps *= self.eps_update

            # If there is enough data in replay buffer, train the model on it
            if len(self.replay_buffer) >= self.batch_size:
                for i in range(10):
                    states, actions, rewards = self.replay_buffer.get_data(self.batch_size)
                    rewards = self.normalize_rewards(rewards)
                    self.train_network((states, actions, rewards))

            print_network_weights(self.network)
        logger.debug('Finished training.')

    def sample_episode(self, batch_size, eps):
        states = []
        actions = []
        next_states = []
        rewards = []
        terminal = False
        state = self.env.reset()
        state = StatePreprocessor.process(state)
        for i in range(batch_size):
            states.append(state)
            action = self.get_action(state=state, eps=eps)
            actions.append(action)
            state, reward, terminal_action, _ = self.env.step(action=action)
            state = StatePreprocessor.process(state)
            next_states.append(state)
            rewards.append(reward)
            terminal = terminal_action
            if terminal_action:
                break
            logger.debug('Step {step} state: {state}, action: {action}.'.format(step=i, rew=reward, action=action, state=state))
        return states, actions, next_states, rewards, terminal

    def get_action(self, state, eps):
        """
        Get action by epsilon-greedy strategy.
        :param state: state
        :return: action
        """
        if random.uniform(0, 1) > eps:
            return self.network.predict(state=state)
        else:
            return random.randint(0, output_shape - 1)

    def disc_rewards(self, rewards):
        disc_rewards = np.zeros_like(rewards)
        cumulative = 0.0
        for i in reversed(range(len(rewards))):
            cumulative = cumulative * self.discount + rewards[i]
            disc_rewards[i] = cumulative
        return disc_rewards

    def normalize_rewards(self, rewards):
        norm_rewards = np.copy(rewards)
        mean = np.mean(norm_rewards)
        std = np.std(norm_rewards)
        norm_rewards -= mean
        if abs(std) > 1e-9:
            norm_rewards /= std
        return norm_rewards

    def train_network(self, batch):
        states, actions, rewards = batch
        states = np.array(states)

        enc = OneHotEncoder(n_values=output_shape)
        actions = np.array(actions).reshape(-1, 1)
        actions = enc.fit_transform(actions).toarray()

        rewards = np.array(rewards, dtype='float32')

        logger.debug('Training network on batch: states {s_shape}, actions {a_shape}, rewards {r_shape}.\n'
                     .format(s_shape=states.shape, a_shape=actions.shape, r_shape=rewards.shape))

        self.replay_buffer.save_data()
        self.network.train(states=states, actions=actions, rewards=rewards)

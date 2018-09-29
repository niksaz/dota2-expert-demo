import logging
import pickle

from policy_gradient import replay_util
from policy_gradient.replay_buffer import ReplayBuffer
from policy_gradient.network import Network
import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger('DotaRL.PGAgent')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# TODO input_shape = 83
# TODO output_shape = 33
input_shape = 2
output_shape = 16


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
                 'rewards')

    def __init__(self, environment, episodes=100, batch_size=100, eps=0.7, discount=0.99, eps_update=0.99):
        self.replay_buffer = ReplayBuffer()
        self.network = Network(input_shape=input_shape, output_shape=output_shape)
        self.env = environment()
        self.episodes = episodes
        self.batch_size = batch_size
        self.eps = eps
        self.discount = discount
        self.eps_update = eps_update
        self.rewards = []

    def train(self):
        for episode in range(self.episodes):
            # sample data
            states, actions, rewards = self.sample_data(steps=self.batch_size)
            rewards = np.array(rewards, dtype='float32')

            logger.debug('Finished episode {ep} with total reward {rew}. eps={eps}'.format(ep=episode,
                                                                                           rew=np.sum(rewards),
                                                                                                       eps=self.eps))

            self.rewards.extend(rewards)

            with open('saved_rewards', 'wb') as output_file:
                pickle.dump(obj=self.rewards, file=output_file)

            # discount and normalize rewards
            disc_rewards = self.discount_rewards(rewards=rewards, gamma=self.discount)
            norm_rewards = self.normalize_rewards(rewards=disc_rewards)

            # extend replay buffer with sampled data
            self.replay_buffer.extend(zip(states, actions, norm_rewards))

            # update epsilon
            self.update_eps(coefficient=self.eps_update)

            # if there are enough data in replay buffer, train the model on it
            if len(self.replay_buffer) >= self.batch_size:
                self.train_network(batch=self.replay_buffer.get_data(self.batch_size))

        logger.debug('Finished training.')

    def sample_data(self, steps=100):
        states = []
        actions = []
        rewards = []
        state = self.env.reset()
        for i in range(steps):
            action = self.get_action(state=state, eps=self.eps)
            state, terminal, reward = self.env.execute(action=action)
            if terminal:
                break
            logger.debug('Step {step} state: {state}, action: {action}.'.format(step=i, rew=reward, action=action, state=state[:2]))
            states.append(state)
            actions.append(action)
            rewards.append(reward)

        return states, actions, rewards

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

    @staticmethod
    def discount_rewards(rewards, gamma):
        """
        Discount rewards backwards.
        :param rewards: rewards numpy array
        :param gamma: discount factor
        :return: discounted rewards array
        """
        running_add = 0.
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * gamma + rewards[t]
            np.put(a=rewards, ind=t, v=running_add)

        return rewards

    @staticmethod
    def normalize_rewards(rewards):
        mean = np.mean(rewards)
        std = np.std(rewards)
        rewards -= mean
        rewards /= std
        return rewards

    def update_eps(self, coefficient=0.9):
        self.eps *= coefficient

    def train_on_replay(self, batch_size=500, epochs=25):
        self.replay_buffer.load_data()
        # states, actions, rewards = self.replay_buffer.get_data(batch_size=len(self.replay_buffer))
        # rewards = np.array(rewards, dtype='float32')

        # discount and normalize rewards
        # rewards = self.discount_rewards(rewards=rewards, gamma=self.discount)
        # rewards = self.normalize_rewards(rewards=rewards)
        # self.replay_buffer.extend(zip(states, actions, rewards))

        for epoch in range(epochs):
            i = 0
            while i < len(self.replay_buffer):
                batch = self.replay_buffer.get_batch(i, batch_size=batch_size)
                self.train_network(batch=batch)
                i += batch_size
            logger.debug('Training: epoch {epoch}.'.format(epoch=epoch))

    def train_network(self, batch):
        states, actions, rewards = batch
        states = np.array(states)

        enc = OneHotEncoder(n_values=output_shape)
        actions = np.array(actions).reshape(-1, 1)
        actions = enc.fit_transform(actions).toarray()

        rewards = np.array(rewards, dtype='float32')

        logger.debug('Training network on batch: states {s_shape}, actions {a_shape}, rewards {r_shape}.\n Reward: {rew}'
                     .format(s_shape=states.shape, a_shape=actions.shape, r_shape=rewards.shape, rew=rewards))

        self.replay_buffer.save_data()
        self.network.train(states=states, actions=actions, rewards=rewards)

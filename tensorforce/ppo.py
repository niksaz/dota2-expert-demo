import pickle
import sys

sys.path.append('../')

from dotaenv import DotaEnvironment

from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner


reload = False
if reload:
    with open('saved_rewards.pkl', 'rb') as output_file:
        rewards = pickle.load(output_file)
else:
    rewards = []

# Create a Dota environment
env = DotaEnvironment()

# Network
network_spec = [
    dict(type='dense', size=9),
    dict(type='nonlinearity', name='relu'),
    dict(type='dense', size=16),
]

agent = PPOAgent(
    states=env.states,
    actions=env.actions,
    network=network_spec,
    update_mode=dict(
        batch_size=100,
    ),
    # PPOAgent
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-3
    ),
    # Model
    scope='ppo',
    discount=0.99,
    saver=dict(
        directory='weights/',
        load=reload
    ),
    # DistributionModel
    distributions=None,
    entropy_regularization=0.01,
    # PGModel
    baseline_mode=None,
    baseline=None,
    baseline_optimizer=None,
    gae_lambda=None,
    # PGLRModel
    actions_exploration=dict(
        type='epsilon_decay',
        initial_epsilon=1.0,
        final_epsilon=0.05,
        timesteps=500000,
    ),
    likelihood_ratio_clipping=0.2,
)

# Create the runner
runner = Runner(agent=agent, environment=env)


# Callback function printing episode statistics
def episode_finished(r):
    reward = r.episode_rewards[-1]
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode,
                                                                                 ts=r.episode_timestep,
                                                                                 reward=reward))
    rewards.append(reward)
    with open('saved_rewards.pkl', 'wb') as output_file:
        pickle.dump(obj=rewards, file=output_file)

    return True


# Start learning
runner.run(episodes=200, max_episode_timesteps=10000,
           episode_finished=episode_finished)
runner.close()

# Print statistics
print("Learning finished. Total episodes: {ep}.".format(ep=runner.episode))

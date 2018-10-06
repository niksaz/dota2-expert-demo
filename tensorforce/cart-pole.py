import pickle

import matplotlib.pyplot as plt
from tensorforce.agents import PPOAgent
from tensorforce.contrib.openai_gym import OpenAIGym
from tensorforce.execution import Runner

# Create an OpenAI gym environment
env = OpenAIGym('CartPole-v0', visualize=True)

# Network as list of layers
network_spec = [
    dict(type='dense', size=32, activation='tanh'),
    dict(type='dense', size=32, activation='tanh')
]

agent = PPOAgent(
    states=env.states,
    actions=env.actions,
    network=network_spec,
    update_mode=dict(
        batch_size=10,
    ),
    # PPOAgent
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-3
    ),
    optimization_steps=10,
    # Model
    scope='ppo',
    discount=0.99,
    # DistributionModel
    distributions=None,
    entropy_regularization=0.01,
    # PGModel
    baseline_mode=None,
    baseline=None,
    baseline_optimizer=None,
    gae_lambda=None,
    # PGLRModel
    likelihood_ratio_clipping=0.2
)

# Create the runner
runner = Runner(agent=agent, environment=env)


# Callback function printing episode statistics
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
    with open('saved_rewards.pkl', 'wb') as output_file:
        pickle.dump(obj=r.episode_rewards, file=output_file)

    return True


# Start learning
runner.run(episodes=5, max_episode_timesteps=200,
           episode_finished=episode_finished)
runner.close()

plt.plot(
    list(map(lambda x: 2*x, range(len(runner.episode_rewards)))),
runner.episode_rewards)
plt.title('Reward per episode')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.savefig('graph.png')
plt.show()

# Print statistics
print("Learning finished. Total episodes: {ep}.".format(ep=runner.episode))

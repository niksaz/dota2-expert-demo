from openai.deepq import models  # noqa
from openai.deepq.build_graph import build_act, build_train  # noqa
from openai.deepq.deepq import learn, load_act  # noqa
from openai.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa

def wrap_atari_dqn(env):
    from baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=False)

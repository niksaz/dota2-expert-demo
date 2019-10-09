import sys
import os.path as osp
import numpy as np
import json

from baselines.common.cmd_util import common_arg_parser, parse_unknown_args
from baselines import logger
from openai.deepq.deepq import learn

from dotaenv import DotaEnvironment

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def train(args, extra_args, config):
    env_type = 'steam'
    env_id = 'dota2'
    print('env_type: {}'.format(env_type))

    alg_kwargs = dict(
        network='mlp',
        num_hidden=128,
        num_layers=1,
        lr=1e-3,
        buffer_size=100000,
        total_timesteps=500000,
        exploration_fraction=1.0,
        exploration_initial_eps=0.1,
        exploration_final_eps=0.1,
        train_freq=4,
        target_network_update_freq=1000,
        gamma=0.999,
        batch_size=32,
        prioritized_replay=True,
        prioritized_replay_alpha=0.6,
        experiment_name=args.exp_name,
        dueling=True)
    alg_kwargs.update(extra_args)
    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))
    print('Shaping config is {}'.format(config))
    config['max_timesteps_to_shape'] += alg_kwargs['total_timesteps']

    seed = args.seed
    learn(seed=seed, config=config, **alg_kwargs)


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'


def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):
        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}


def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)
    np.set_printoptions(precision=3)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    if args.exp_name is None:
        print('Please, specify the name of the experiment in --exp_name')
        exit(0)

    if args.config is None:
        print('Please, specify the path to the config via --config')
        exit(0)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        logger.configure()
    else:
        logger.configure(format_strs=[])
        rank = MPI.COMM_WORLD.Get_rank()

    with open(args.config, 'r') as finput:
        config = json.load(finput)

    train(args, extra_args, config)
    return

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    if args.play:
        logger.log("Running trained model")
        env = DotaEnvironment()
        obs = env.reset()

        def initialize_placeholders(nlstm=128,**kwargs):
            return np.zeros((args.num_env or 1, 2*nlstm)), np.zeros((1))

        state, dones = initialize_placeholders(**extra_args)
        while True:
            actions, _, state, _ = model.step(obs, S=state, M=dones)
            obs, _, done, _ = env.step(actions)
            env.render()
            done = done.any() if isinstance(done, np.ndarray) else done

            if done:
                obs = env.reset()
        env.close()


if __name__ == '__main__':
    main(sys.argv)

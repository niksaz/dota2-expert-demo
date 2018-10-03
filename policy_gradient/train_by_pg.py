import logging
import time

from dotaenv import DotaEnvironment
from policy_gradient import PGAgent

logger = logging.getLogger('DotaRL')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('logs/policy_gradient{time}.log'.format(time=time.strftime("%Y%m%d-%H%M%S")))
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def create_dota_agent():
    return PGAgent(environment=DotaEnvironment,
                   episodes=3000,
                   eps=0.7,
                   batch_size=200,
                   eps_update=0.999)


def main():
    agent = create_dota_agent()
    # agent.train_on_replay(epochs=100000, batch_size=1000)
    agent.train()


if __name__ == '__main__':
    main()

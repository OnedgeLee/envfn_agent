import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import tensorflow as tf
import agent_ddpg
from tensorflow.python.platform import app
import numpy as np

np.random.seed(1)
tf.set_random_seed(1)

FLAGS = None

def main(_):
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    env = agent_ddpg.env.env_ep.Env(FLAGS.env_name, 8760)
    agent = agent_ddpg.model.DDPG(sess, env, FLAGS, rl_mode=True)
    agent_ddpg.model.learn(FLAGS, env, agent)

if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--env_name', type=str, default='ep',
                        help='environment name')
    parser.add_argument('--num_steps', type=int, default=8760,
                        help='episode steps')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--num_episodes', type=int, default=int(1e5),
                        help='episodes to run')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor')
    parser.add_argument('--tau', type=float, default=0.01,
                        help='updating ratio')
    parser.add_argument('--critic_lr', type=float, default=1e-4,
                        help='critic learning rate')
    parser.add_argument('--actor_lr', type=float, default=1e-4,
                        help='actor learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--replayBuffer_size', type=int, default=8760,
                        help='replay buffer size')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run()
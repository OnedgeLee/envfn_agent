import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np

import tensorflow as tf
from tensorflow.python.platform import app

import gym
from gym.envs.registration import register

from stable_baselines.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines import bench, logger
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy
from stable_baselines.common import tf_util
from stable_baselines import PPO2

import agent_ddpg
import env
import conf



np.random.seed(1)
tf.set_random_seed(1)

FLAGS = None

def main(_):

    p_dic = getattr(conf.dic.path_dic, FLAGS.env_name)
    
    register(
        id=FLAGS.env_id,
        entry_point='env.env_ep:Env',
        kwargs={
            'env_name':FLAGS.env_name,
            'done_step':8760
            }
    )

    def make_env():
        env_out = gym.make(FLAGS.env_id)
        env_out = bench.Monitor(env_out, logger.get_dir(), allow_early_resets=True)
        return env_out

    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    policy = {'cnn': CnnPolicy, 'lstm': CnnLstmPolicy, 'lnlstm': CnnLnLstmPolicy, 'mlp': MlpPolicy}[FLAGS.policy]
    model = PPO2(policy=policy, env=env, n_steps=FLAGS.n_steps, nminibatches=FLAGS.nminibatches,
                 lam=FLAGS.lam, gamma=FLAGS.gamma, noptepochs=FLAGS.noptepochs, ent_coef=FLAGS.ent_coef,
                 learning_rate=FLAGS.learning_rate, cliprange=FLAGS.cliprange, verbose=FLAGS.verbose, log_dir=p_dic.get('agent_log_dir'))
    model.learn(total_timesteps=FLAGS.num_timesteps)

if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--env_name', type=str, default='ep',
                        help='(str) Environment name')
    parser.add_argument('--env_id', type=str, default='ep-v0',
                        help='(str) Environment id')
    parser.add_argument('--num_timesteps', type=int, default=8760 * 100,
                        help='(int) The number of timesteps to run')
    parser.add_argument('--policy', choices=['cnn', 'lstm', 'lnlstm', 'mlp'], default='mlp',
                        help='(ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)')
    parser.add_argument('--n_steps', type=int, default=8760,
                        help='(int) The number of steps to run for each environment per update')
    parser.add_argument('--nminibatches', type=int, default=40,
                        help='(int) Number of training minibatches per update. For recurrent policies, the number of environments run in parallel should be a multiple of nminibatches.')
    parser.add_argument('--lam', type=float, default=0.95,
                        help='(float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='(float) Discount factor')
    parser.add_argument('--noptepochs', type=int, default=24,
                        help='(int) Number of epoch when optimizing the surrogate')
    parser.add_argument('--ent_coef', type=float, default=.01,
                        help='(float) Entropy coefficient for the loss caculation')
    parser.add_argument('--learning_rate', default=lambda f: f * 2.5e-4,
                        help='(float or callable) The learning rate, it can be a function')
    parser.add_argument('--cliprange', default=lambda f: f * 0.1,
                        help='(float or callable) Clipping parameter, it can be a function')
    parser.add_argument('--verbose', type=int, default=1,
                        help='(int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run()

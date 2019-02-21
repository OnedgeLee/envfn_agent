import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import tensorflow as tf
import agent_ddpg
from tensorflow.python.platform import app
import numpy as np
import conf
from tools.pipe_io import PipeIo
import subprocess
import signal
from tools.energyplus_env import EnergyPlusEnv
import time

np.random.seed(1)
tf.set_random_seed(1)

FLAGS = None

def normalize(original_feature, scale, translate):
    return (original_feature - translate) / scale

def restore(normalized_feature, scale, translate):
    return normalized_feature * scale + translate


def main(_):
    p_dic = getattr(conf.dic.path_dic, FLAGS.env_name)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    env = agent_ddpg.env.Env(FLAGS.env_name)

    epenv = EnergyPlusEnv(
        energyplus_file="/usr/local/EnergyPlus-8-8-0/energyplus",
        model_file=p_dic.get('idf_path'),
        weather_file="/usr/local/EnergyPlus-8-8-0/WeatherData/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw",
        log_dir=p_dic.get('eplog_dir'))

    os.environ['ENERGYPLUS'] = "/usr/local/EnergyPlus-8-8-0/energyplus"
    os.environ['ENERGYPLUS_MODEL'] = p_dic.get('idf_path')
    os.environ['ENERGYPLUS_WEATHER'] = "/usr/local/EnergyPlus-8-8-0/WeatherData/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw"
    os.environ['ENERGYPLUS_LOG'] = p_dic.get('eplog_dir')

    agent = agent_ddpg.model.DDPG(sess, env, FLAGS, rl_mode=False)
    agent.load()
    
    epenv.start_instance()

    def signal_handler(signal, frame):
        epenv.stop_instance
        print('=====Energy plus terminated=====')
        print('==========Pipe closed==========')
        sys.exit()

    signal.signal(signal.SIGINT, signal_handler)
    
    state = epenv.reset()

    for i in range(10000000000000):
        state = np.array([[state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7], state[8]]])
        state = normalize(state, agent.state_scale, agent.state_translate)
        action = agent.act(state)
        action = restore(action, agent.action_scale, agent.action_translate)
        action = action.reshape([-1])
        state, done = epenv.step(action)
    epenv.stop_instance()

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
    parser.add_argument('--gamma', type=float, default=0.9,
                        help='discount factor')
    parser.add_argument('--tau', type=float, default=0.01,
                        help='updating ratio')
    parser.add_argument('--critic_lr', type=float, default=1e-4,
                        help='critic learning rate')
    parser.add_argument('--actor_lr', type=float, default=1e-4,
                        help='actor learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--replayBuffer_size', type=int, default=64,
                        help='replay buffer size')
    parser.add_argument('--log_dir', default='logs/',
                        help='log directory')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run()
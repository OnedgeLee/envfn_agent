import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import tensorflow as tf
import numpy as np
import pandas as pd
import agent_ddpg
import envfn_dnn
import conf
import gym
from gym import spaces
from tools import tool_fn

class Env(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, env_name, done_step):

        self.envfn = True
        self.num_envs = 1

        self.p_dic = getattr(conf.dic.path_dic, env_name)
        self.c_dic = getattr(conf.dic.col_dic, env_name)
        self.s_dic = getattr(conf.dic.space_dic, env_name)
        self.cols = {
            'agent_a': getattr(conf.list.agent.action, env_name),
            'agent_dvs': getattr(conf.list.agent.dv_state, env_name),
            'agent_ivs': getattr(conf.list.agent.iv_state, env_name),
            'agent_rs': getattr(conf.list.agent.rew_state, env_name),
            'envfn_f': getattr(conf.list.envfn.feature, env_name),
            'envfn_l': getattr(conf.list.envfn.label, env_name),
        }
        self.act_space = np.array([self.s_dic[acts] for acts in self.cols.get('agent_a')], dtype=np.float32).T
        self.obs_space = np.array([self.s_dic[obss] for obss in self.cols.get('agent_dvs')+self.cols.get('agent_ivs')], dtype=np.float32).T
        self.rew_space = np.array([self.s_dic[sts] for sts in self.cols.get('agent_rs' )], dtype=np.float32).T
        self.rew_fn = getattr(conf.fn.agent.env_rew_fn, env_name)
        self.init_fn = getattr(conf.fn.agent.env_init_fn, env_name)
        self.rst_fn = getattr(conf.fn.agent.env_rst_fn, env_name)

        self.ive = envfn_dnn.interactive_ve.IVE(env_name)
        self.sess = self.ive.load()
        act_dim = (len(self.cols.get('agent_a')),)
        obs_dim = (len(self.cols.get('agent_dvs') + self.cols.get('agent_ivs')),)
        init = self.init_fn(act_dim, obs_dim)
        act_low = init[0]
        act_high = init[1]
        obs_low = init[2]
        obs_high = init[3]
        self.dvs = None
        self.ivs = None
        self.obs = None
        self.act = None
        self.rew = None
        self.done = True
        self.info = None

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)
        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {'render.modes': ['human']}
        self.spec = None
        self.done_step = done_step

        ivsss = os.listdir(self.p_dic.get('ivs_data_dir'))
        ivsss.sort()
        ivss_df = None
        for ivss in range(len(ivsss)):
            csv_path = self.p_dic.get('ivs_data_dir') + '/' +  ivsss[ivss]
            ivss_df_part = pd.read_csv(
                csv_path, dtype=np.float32, delimiter=',', error_bad_lines=False, index_col=0)
            if ivss == 0:
                ivss_df = ivss_df_part
            else:
                ivss_df = pd.concat[ivss_df, ivss_df_part]
        self.ivss = ivss_df.values

    def reset(self):

        rst = self.rst_fn()
        self.dvs = np.expand_dims(rst, axis=0)
        self.ivs = np.expand_dims(self.ivss[0], axis=0)
        self.obs = np.concatenate([self.dvs, self.ivs], axis=1)
        self.act = None
        self.rew = None
        done = np.array([False], dtype=np.bool)
        self.info = None
        self.timesteps = 0

        return self.obs
    
    def render(self):
        # 추후 그래프 그릴 때 이용해 보자
        return 0
        

    def step(self, act_norm):
        self.timesteps += 1

        if (self.timesteps) == self.done_step:
            done = np.array([True], dtype=np.bool)
        else:
            done = np.array([False], dtype=np.bool)
        
        self.act = np.expand_dims(act_norm, axis=0)
        # self.act = self.set_action(act_norm)
        batch_size = self.act.shape[0]

        self.whole_data = pd.DataFrame(
            data=np.concatenate([self.act, self.obs], axis=1),
            columns=self.cols.get('agent_a')+self.cols.get('agent_dvs')+self.cols.get('agent_ivs'))

        self.envfn_f = self.whole_data[self.cols.get('envfn_f')].values

        self.dvs = self.ive.predict(self.envfn_f)
        self.ivs = self.ivss[(((self.timesteps % self.ivss.shape[0]) - 1) * batch_size):(self.timesteps % self.ivss.shape[0]) * batch_size]
        self.obs = np.concatenate([self.dvs, self.ivs], axis=1)

        # if np.all([self.obs_space[0] < self.obs, self.obs < self.obs_space[1]]):
        self.rs = self.whole_data[self.cols.get('agent_rs' )].values
        self.rs = tool_fn.transform(self.rs, self.rew_space[0], self.rew_space[1])
        self.rew = self.rew_fn(self.rs)
        # else:
            # done = np.array([True], dtype=np.bool)
            # self.rew = [-100]

        next_obs = self.obs
        reward = self.rew
        
        info = {}

        return next_obs[0], reward[0], done, info

    def set_action(self, act_norm):
        act = self.action_space.low + (act_norm + 1.) * 0.5 * (self.action_space.high - self.action_space.low)
        return np.expand_dims(np.clip(act, self.action_space.low, self.action_space.high), axis=0)

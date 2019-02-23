import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import tensorflow as tf
import numpy as np
import pandas as pd
import agent_ddpg
import envfn_dnn
import conf

class Space:
    def __init__(self, shape, low, high):
        self.shape = shape
        self.low = low
        self.high = high

class Env:
    def __init__(self, env_name):

        self.envfn = True

        self.p_dic = getattr(conf.dic.path_dic, env_name)
        self.c_dic = getattr(conf.dic.col_dic, env_name)
        self.cols = {
            'agent_a': getattr(conf.list.agent.action, env_name),
            'agent_dvs': getattr(conf.list.agent.dv_state, env_name),
            'agent_ivs': getattr(conf.list.agent.iv_state, env_name),
            'agent_rs': getattr(conf.list.agent.rew_state, env_name),
            'envfn_f': getattr(conf.list.envfn.feature, env_name),
            'envfn_l': getattr(conf.list.envfn.label, env_name),
        }
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
        
        self.observation_space = Space(obs_dim, obs_low, obs_high)
        self.action_space = Space(act_dim, act_low, act_high)

    def reset(self):

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

        rst = self.rst_fn()
        self.dvs = np.expand_dims(rst, axis=0)
        self.ivs = np.expand_dims(self.ivss[0], axis=0)
        self.obs = np.concatenate([self.dvs, self.ivs], axis=1)
        self.act = None
        self.rew = None
        self.done = False
        self.info = None
        self.timesteps = 0

        return self.obs
    
    def render(self):
        # 추후 그래프 그릴 때 이용해 보자
        return 0
        

    def step(self, action):
        self.timesteps += 1
        
        self.act = action
        batch_size = self.act.shape[0]

        self.whole_data = pd.DataFrame(
            data=np.concatenate([self.act, self.obs], axis=1),
            columns=self.cols.get('agent_a')+self.cols.get('agent_dvs')+self.cols.get('agent_ivs'))

        self.envfn_f = self.whole_data[self.cols.get('envfn_f')].values

        self.dvs = self.ive.predict(self.envfn_f)
        self.ivs = self.ivss[(((self.timesteps % self.ivss.shape[0]) - 1) * batch_size):(self.timesteps % self.ivss.shape[0]) * batch_size]
        self.obs = np.concatenate([self.dvs, self.ivs], axis=1)
        self.rs = self.whole_data[self.cols.get('agent_rs' )].values
        self.rew = self.rew_fn(self.rs)

        next_obs = self.obs
        reward = self.rew
        done = False
        info = None

        return next_obs, reward, done, info
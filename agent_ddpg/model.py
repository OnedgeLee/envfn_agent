import os
import tensorflow as tf
import time
import datetime
import numpy as np
from matplotlib import pyplot as plt
from replay_buffer import Memory
from architecture import *
from tools import tool_fn
import conf


class DDPG:

    def __init__(self, sess, env, FLAGS, rl_mode):

        self.FLAGS = FLAGS
        self.rl_mode = rl_mode
        self.p_dic = getattr(conf.dic.path_dic, self.FLAGS.env_name)
        self.s_dim = env.observation_space.shape[0]
        self.a_dim = env.action_space.shape[0]
        self.sess = sess
        self._build_graph()
        self.state_translate = env.observation_space.low
        self.state_scale = env.observation_space.high - env.observation_space.low + 1e-5
        self.action_translate = env.action_space.low
        self.action_scale = env.action_space.high - env.action_space.low + 1e-5

        if self.rl_mode:
            self.memory = Memory(self.FLAGS.replayBuffer_size, dims=2 * self.s_dim + self.a_dim + 1)

    def normalize(self, original_feature, scale, translate):
        return (original_feature - translate) / scale

    def restore(self, normalized_feature, scale, translate):
        return normalized_feature * scale + translate

    def _build_graph(self):
        self._placehoders()
        self._actor_critic()
        self._loss_train_op()
        self.score = tf.Variable(0., trainable=False, dtype=tf.float32, name='score')
        self.score_summary = tf.summary.scalar('score', self.score)
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(self.p_dic.get('agent_log_dir'))
        self.writer.add_graph(self.sess.graph)
        self.saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)

    def _placehoders(self):
        with tf.name_scope('inputs'):
            self.current_state = tf.placeholder(tf.float32, shape=[None, self.s_dim], name='s')
            self.reward = tf.placeholder(tf.float32, [None, 1], name='r')
            self.next_state = tf.placeholder(tf.float32, shape=[None, self.s_dim], name='s_')
            self.is_training = tf.placeholder(tf.bool, name='is_training')

    def _actor_critic(self):
        self.actor = build_actor(self.current_state, self.a_dim, self.is_training)
        self.actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Actor')
        actor_ema = tf.train.ExponentialMovingAverage(decay=1 - self.FLAGS.tau)
        self.update_targetActor = actor_ema.apply(self.actor_vars)
        self.targetActor = build_actor(self.next_state, self.a_dim, False,
                                          reuse=True, getter=get_getter(actor_ema))

        self.critic = build_critic(self.current_state, self.actor, self.is_training)
        self.critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Critic')
        critic_ema = tf.train.ExponentialMovingAverage(decay=1 - self.FLAGS.tau)
        self.update_targetCritic = critic_ema.apply(self.critic_vars)
        self.targetCritic = build_critic(self.next_state, self.targetActor, False,
                                            reuse=True, getter=get_getter(critic_ema))

    def _loss_train_op(self):
        max_grad = 2
        with tf.variable_scope('target_q'):
            self.target_q = self.reward + self.FLAGS.gamma * self.targetCritic
        with tf.variable_scope('TD_error'):
            self.critic_loss = tf.squared_difference(self.target_q, self.critic)
        with tf.variable_scope('critic_grads'):
            self.critic_grads = tf.gradients(ys=self.critic_loss, xs=self.critic_vars)
            for ix, grad in enumerate(self.critic_grads):
                self.critic_grads[ix] = grad / self.FLAGS.batch_size
        with tf.variable_scope('C_train'):
            critic_optimizer = tf.train.AdamOptimizer(self.FLAGS.critic_lr, epsilon=1e-5)
            self.train_critic = critic_optimizer.apply_gradients(zip(self.critic_grads, self.critic_vars))
        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.critic, self.actor)[0]
        with tf.variable_scope('actor_grads'):
            self.actor_grads = tf.gradients(ys=self.actor, xs=self.actor_vars, grad_ys=self.a_grads)
            for ix, grad in enumerate(self.actor_grads):
                self.actor_grads[ix] = tf.clip_by_norm(grad / self.FLAGS.batch_size, max_grad)
        with tf.variable_scope('A_train'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                actor_optimizer = tf.train.AdamOptimizer(-self.FLAGS.actor_lr,
                                                         epsilon=1e-5)
                self.train_actor = actor_optimizer.apply_gradients(zip(self.actor_grads, self.actor_vars))

    def choose_action(self, state):

        return self.sess.run(self.actor, feed_dict={self.current_state: state,
                                                    self.is_training: False})

    def train(self, episode=None, ep_reward=None):
        batch_memory = self.memory.sample(self.FLAGS.batch_size)
        batch_s = batch_memory[:, :self.s_dim]
        batch_a = batch_memory[:, self.s_dim: self.s_dim + self.a_dim]
        batch_r = batch_memory[:, -self.s_dim - 1: -self.s_dim]
        batch_s_ = batch_memory[:, -self.s_dim:]

        if episode is None:
            critic_feed_dict = {self.current_state: batch_s, self.actor: batch_a, self.reward: batch_r, self.next_state: batch_s_, self.is_training: True}
            self.sess.run([self.train_critic, self.update_targetCritic],
                          feed_dict=critic_feed_dict)
            actor_feed_dict = {self.current_state: batch_s, self.next_state: batch_s_, self.is_training: True}
            self.sess.run([self.train_actor, self.update_targetActor],
                          feed_dict=actor_feed_dict)
        else:
            update_score = self.score.assign(tf.convert_to_tensor(ep_reward, dtype=tf.float32))
            with tf.control_dependencies([update_score]):
                merged_score = tf.summary.merge([self.score_summary])
            self.critic_summary = tf.summary.merge_all(scope='Critic')
            self.actor_summary =  tf.summary.merge_all(scope='Actor')
            critic_feed_dict = {self.current_state: batch_s, self.actor: batch_a, self.reward: batch_r, self.next_state: batch_s_, self.is_training: True}
            _, _, critic = self.sess.run([self.train_critic, self.update_targetCritic, self.critic_summary],
                                         feed_dict=critic_feed_dict)
            self.writer.add_summary(critic, episode)
            actor_feed_dict = {self.current_state: batch_s, self.next_state: batch_s_, self.is_training: True}
            merged = tf.summary.merge([merged_score, self.actor_summary])
            _, _, actor = self.sess.run([self.train_actor, self.update_targetActor, merged],
                                        feed_dict=actor_feed_dict)
            self.writer.add_summary(actor, episode)
            
            self.saver.save(
                self.sess, 
                self.p_dic.get('agent_log_dir') + '/' + datetime.datetime.now().strftime('%y%m%d-%H:%M:%S') + '_EP' + str(episode) + '.ckpt')

            

    def perceive(self, state, action, reward, next_state, episode=None, ep_reward=None):
        self.memory.store_transition(state, action, reward, next_state)
        if self.memory.pointer > self.FLAGS.replayBuffer_size:
            self.train(episode, ep_reward)


    def load(self):
        # self.saver.restore(self.sess, self.p_dic.get('agent_log_dir') + '/190130-15:35:26_EP1.ckpt')
        # self.saver.restore(self.sess, self.p_dic.get('agent_log_dir') + '/190130-16:42:17_EP30.ckpt')
        # self.saver.restore(self.sess, self.p_dic.get('agent_log_dir') + '/190130-14:59:11_EP40.ckpt')
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.p_dic.get('agent_log_dir')))
        
    def act(self, obs):
        actor_feed_dict = {self.current_state: obs, self.is_training: False}
        action = self.sess.run(self.actor, feed_dict=actor_feed_dict)
        act_low = np.array([16.,16.,16.,16.,7.36,16.,16.,16.,16.,6.57], dtype=np.float32)
        act_high = np.array([30.,30.,30.,30.,7.36,30.,30.,30.,30.,6.57], dtype=np.float32)
        return action
        


def learn(FLAGS, env, agent):
    render = False
    sd = 1
    print('\n\n>> Agent training start')
    for e in range(FLAGS.num_episodes):
        obs = env.reset()
        obs = agent.normalize(obs, agent.state_scale, agent.state_translate)
        ep_reward = 0
        # rss : 추후 env 쪽으로 넘김
        # env에 epsode_render 메서드 만들어서 거기에 포함
        rss = []
        start = time.time()
        for j in range(FLAGS.num_steps):
            if render:
                env.render()
            action = agent.choose_action(obs)
            action = agent.restore(action, agent.action_scale, agent.action_translate)
            action = np.random.normal(loc=action, scale=sd)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            next_obs, reward, done, info = env.step(action)
            next_obs = agent.normalize(next_obs, agent.state_scale, agent.state_translate)
            action = agent.normalize(action, agent.action_scale, agent.action_translate)
            # rss : 추후 env 쪽으로 넘김
            if hasattr(env, 'envfn'):
                rss.append(env.rs.reshape(-1))
            ep_reward += np.sum(reward)
            end = time.time()
            tool_fn.progress_bar(
                j + 1, 
                FLAGS.num_steps, 
                prefix='>> EP:%i' % e, 
                suffix='(STEP:%i, R:%.2f, FPS:%.0f, SD:%.2f)' % (j, ep_reward / (j + 1), (j + 1) / (end - start), sd))
            if j == FLAGS.num_steps - 1:
                agent.perceive(obs, action, reward, next_obs, episode=e, ep_reward=ep_reward)
                if ep_reward > 10000:
                    render = True
                break
            else:
                agent.perceive(obs, action, reward, next_obs)
            obs = next_obs
        if sd >= 0.1:
            # sd *= .995
            sd *= .9
        
        # 플로팅 코드, 추후 rss 변수와 함께 env episode_render 메서드로 옮길 것 #
        if hasattr(env, 'envfn'):
            if not os.path.exists(env.p_dic.get('agent_graph_dir') + '/' + str(e)):
                os.makedirs(env.p_dic.get('agent_graph_dir') + '/' + str(e))
            rss = np.array(rss)
            # indexing = np.arange(FLAGS.num_steps)
            # interval = indexing % (FLAGS.num_steps // 100) == 0
            # for rs in range(len(env.cols.get('agent_rs'))):
            #     plt.title(env.cols.get('agent_rs')[rs] + ' (instant)')
            #     plt.plot(indexing[interval], rss[interval][:, rs])
            #     plt.figure(num=1, figsize=(6, 6), dpi=2000, facecolor='white')
            #     plt.savefig(env.p_dic.get('agent_graph_dir') + '/' + str(e) + '/' + env.cols.get('agent_rs')[rs] + '_inst.png', bbox_inches='tight')
            #     plt.close()

            # 데이터센터만, 일단위 플로팅 #
            rss_ = []
            for i in range(rss.shape[0]):
                if((i % 24) == 0 and (i + 24) <= rss.shape[0]):
                    sum = 0
                    for j in range(24):
                        sum += rss[i+j]
                    rss_.append(sum / 24)
            rss = np.array(rss_)
            indexing = np.arange(rss.shape[0])
            for rs in range(len(env.cols.get('agent_rs'))):
                plt.title(env.cols.get('agent_rs')[rs] + ' (instant)_day')
                plt.plot(indexing, rss[:, rs])
                plt.figure(num=1, figsize=(6, 6), dpi=2000, facecolor='white')
                plt.savefig(env.p_dic.get('agent_graph_dir') + '/' + str(e) + '/' + env.cols.get('agent_rs')[rs] + '_inst_day.png', bbox_inches='tight')
                plt.close()
            # 일단위 플로팅 끝 #
    if hasattr(env, 'envfn'):
        env.sess.close()
    agent.sess.close()

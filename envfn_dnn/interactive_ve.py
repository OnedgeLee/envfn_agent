import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from functools import partial
import numpy as np
import envfn_dnn
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import app
import pandas as pd
import matplotlib.pyplot as plt
import math
import conf

class IVE():

    def __init__(self, env_name):
        self.p_dic = getattr(conf.dic.path_dic, env_name)
        self.m_dic = getattr(conf.dic.envfn_dic, env_name)
        self.f_dim = len(getattr(conf.list.envfn.feature, env_name))
        self.l_dim = len(getattr(conf.list.envfn.label, env_name))

    def data_preproc(self, data, translate, scale):
        return (data - translate) / scale

    def data_transform(self, data, translate, scale):
        return (data * scale) + translate

    def load(self):

        self.g_envfn = tf.Graph()
        with self.g_envfn.as_default():

            architecture = getattr(
                envfn_dnn.architecture, self.m_dic.get('arch'))
            model_function = partial(architecture, predict_size=self.l_dim)
            self.model = envfn_dnn.model.MlpModel(model_function)

            self.feature_translate_restore = tf.get_variable(
                'feature_translate_save', shape=[1, self.f_dim], trainable=False)
            self.feature_scale_restore = tf.get_variable(
                'feature_scale_save', shape=[1, self.f_dim], trainable=False)
            self.label_translate_restore = tf.get_variable(
                'label_translate_save', shape=[1, self.l_dim], trainable=False)
            self.label_scale_restore = tf.get_variable(
                'label_scale_save', shape=[1, self.l_dim], trainable=False)
            
            self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.f_dim])
            preproc_data = self.data_preproc(self.inputs, self.feature_translate_restore, self.feature_scale_restore)
            prediction_intermediate = self.model.predict_model(preproc_data, is_training=False)
            self.prediction = self.data_transform(prediction_intermediate, self.label_translate_restore, self.label_scale_restore)
            
            self.saver = tf.train.Saver()
            sess_config = tf.ConfigProto(allow_soft_placement=True)
            sess_config.gpu_options.allow_growth = True
            sess_config.log_device_placement = True
            self.sess = tf.Session(graph=self.g_envfn, config=sess_config)
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.p_dic.get('envfn_log_dir')))
            
        
        return self.sess

    def predict(self, features):

        with self.g_envfn.as_default():

            prediction_stack = self.sess.run(self.prediction, feed_dict={self.inputs:features})
                
            return prediction_stack

    def close(self):
        self.sess.close()
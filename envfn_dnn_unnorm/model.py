import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class MlpModel(object):

  def __init__(self, model_func):
    
    self.step = tf.train.get_or_create_global_step()
    self.model_func = model_func

  def predict_model(self, features, is_training=True):
    
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
      return self.model_func(features, is_training=is_training)

  def add_mse_loss(self, predictions, labels, weight=1.0, smoothing=0.0):
  
    mse_loss = tf.losses.mean_squared_error(labels, predictions)
    tf.summary.scalar('Loss_mse', mse_loss)

  def create_train_op(self, learning_rate):

    slim.model_analyzer.analyze_vars(
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), print_info=True)

    self.train_loss = tf.losses.get_total_loss()
    # self.train_loss_average = self.add_average(self.train_loss)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    tf.summary.scalar('Learning_Rate', learning_rate)
    tf.summary.scalar('Loss_Total', self.train_loss)
    with tf.control_dependencies(update_ops):
      trainer = tf.train.AdamOptimizer(learning_rate)

    self.train_op = slim.learning.create_train_op(self.train_loss, trainer)
    return self.train_op

  # def add_average(self, variable):
    
  #   tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, self.ema.apply([variable]))
  #   average_variable = tf.identity(
  #       self.ema.average(variable), name=variable.name[:-2] + '_avg')
  #   return average_variable
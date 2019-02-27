import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from functools import partial
import numpy as np
import envfn_dnn
import tensorflow as tf
import tensorflow.contrib.slim as slim
import pandas as pd
import conf
from tools import tool_fn

FLAGS = None

def data_stdize(data):

    mean = tf.reduce_mean(data, 0, True)
    std = tf.reduce_mean(tf.square(data - mean), 0, True)

    translate = mean
    scale = (std + 1e-5)
    
    outputs = (data - translate) / scale

    return outputs, translate, scale


def data_normalize(data):

    high = tf.reduce_max(data, 0, True)
    low = tf.reduce_min(data, 0, True)

    translate = low
    scale = (high - low + 1e-5)

    outputs = (data - translate) / scale

    return outputs, translate, scale


def sample_open(sample_idxs, col_features, col_labels, p_dic):
    tool_fn.progress_bar(0, 1, prefix='>>> Sample extracting...')
    feature_samples = os.listdir(p_dic.get('sample_data_dir') + '/feature')
    label_samples = os.listdir(p_dic.get('sample_data_dir') + '/label')
    feature_samples.sort()
    label_samples.sort()
    feature = np.empty([0, len(col_features)], dtype=np.float32)
    label = np.empty([0, len(col_labels)], dtype=np.float32)
    loads = 0
    for sample_idx in sample_idxs:
        feature_csv_path = p_dic.get('sample_data_dir') + '/feature/' +  feature_samples[sample_idx]
        label_csv_path = p_dic.get('sample_data_dir') + '/label/' +  label_samples[sample_idx]
        feature_part = pd.read_csv(feature_csv_path, dtype=np.float32,
                               delimiter=',', error_bad_lines=False, index_col=0).values
        label_part = pd.read_csv(label_csv_path, dtype=np.float32,
                               delimiter=',', error_bad_lines=False, index_col=0).values
        feature = np.concatenate((feature, feature_part))
        label = np.concatenate((label, label_part))
        loads += 1
        tool_fn.progress_bar(loads, len(sample_idxs), prefix='>>> Sample extracting...')
    print("Sample extracting done")
    return feature, label


def shuffle_n_batch(data_tuple, data_length):
    tool_fn.progress_bar(0, 1, prefix='>>> Shuffling and batch making...')
    dataset = tf.data.Dataset.from_tensor_slices(data_tuple)
    batch = dataset.shuffle(data_length).repeat().batch(FLAGS.batch_size)
    tool_fn.progress_bar(1, 1, prefix='>>> Shuffling and batch making...')
    return batch


def main(_):

    p_dic = getattr(conf.dic.path_dic, FLAGS.env_name)
    c_dic = getattr(conf.dic.col_dic, FLAGS.env_name)
    m_dic = getattr(conf.dic.envfn_dic, FLAGS.env_name)
    cols = {
        'envfn_f': getattr(conf.list.envfn.feature, FLAGS.env_name),
        'envfn_l': getattr(conf.list.envfn.label, FLAGS.env_name)
    }
    col_features = [c_dic[col_feature] for col_feature in cols.get('envfn_f')]
    col_labels = [c_dic[col_label] for col_label in cols.get('envfn_l')]

    train_data = sample_open(FLAGS.sample_idxs, col_features, col_labels, p_dic)
    data_length = train_data[0].shape[0]

    graph = tf.Graph()
    with graph.as_default():
        with tf.device(tf.train.replica_device_setter(0, merge_devices=True)):

            architecture = getattr(
                envfn_dnn.architecture, m_dic.get('arch'))
            model_function = partial(architecture, predict_size=len(col_labels), l2_weight=FLAGS.l2_weight, batch_norm_decay=FLAGS.batch_norm_decay)
            model = envfn_dnn.model.MlpModel(model_function)

            if m_dic.get('preproc') == 'std':
                
                feature, feature_translate, feature_scale = data_stdize(train_data[0])
                label, label_translate, label_scale = data_stdize(train_data[1])
                
            elif m_dic.get('preproc') == 'norm':

                feature, feature_translate, feature_scale = data_normalize(train_data[0])
                label, label_translate, label_scale = data_normalize(train_data[1])

            feature_translate_save = tf.get_variable(
                'feature_translate_save', shape=[1, feature.shape[1]])
            feature_scale_save = tf.get_variable(
                'feature_scale_save', shape=[1, feature.shape[1]])
            feature_translate_assign = feature_translate_save.assign(feature_translate)
            feature_scale_assign = feature_scale_save.assign(feature_scale)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, feature_translate_assign)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, feature_scale_assign)
            
            label_translate_save = tf.get_variable(
                'label_translate_save', shape=[1, label.shape[1]])
            label_scale_save = tf.get_variable(
                'label_scale_save', shape=[1, label.shape[1]])
            label_translate_assign = label_translate_save.assign(label_translate)
            label_scale_assign = label_scale_save.assign(label_scale)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, label_translate_assign)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, label_scale_assign)

            batch = shuffle_n_batch(
                (feature, label), data_length)

            iterator = batch.make_one_shot_iterator()
            feature_batch, label_batch = iterator.get_next()

            prediction_batch = model.predict_model(feature_batch)
            model.add_mse_loss(prediction_batch, label_batch)
            train_op = model.create_train_op(FLAGS.learning_rate)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.log_device_placement = True
            saver = tf.train.Saver(max_to_keep=FLAGS.max_checkpoints,
                                   keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours)
            slim.learning.train(
                train_op,
                logdir=p_dic.get('envfn_log_dir'),
                save_summaries_secs=FLAGS.save_summaries_secs,
                save_interval_secs=FLAGS.save_interval_secs,
                master=FLAGS.master,
                is_chief=(FLAGS.task == 0),
                startup_delay_steps=(FLAGS.task * 20),
                log_every_n_steps=FLAGS.log_every_n_steps,
                session_config=config,
                trace_every_n_steps=1000,
                saver=saver,
                number_of_steps=FLAGS.max_steps,
            )


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str,
                        default='ep', help='environment name')
    parser.add_argument('--sample_idxs', nargs='+', type=int, 
                        default=[0,1,2,3,4,5,6,7,8])
    parser.add_argument('--batch_size', type=int, 
                        default=32)
    parser.add_argument('--l2_weight', type=float, 
                        default=1e-4)
    parser.add_argument('--batch_norm_decay', type=float, 
                        default=0.99)
    parser.add_argument('--learning_rate', type=float, 
                        default=1e-4)
    parser.add_argument('--max_steps', type=int,
                        default=5000000, help='Number of training steps.')
    parser.add_argument('--save_summaries_secs', type=int, 
                        default=30)
    parser.add_argument('--save_interval_secs', type=int, 
                        default=30)
    parser.add_argument('--log_every_n_steps', type=int, 
                        default=100)
    parser.add_argument('--max_checkpoints', type=int, 
                        default=5)
    parser.add_argument('--keep_checkpoint_every_n_hours', type=float, 
                        default=5.0)
    parser.add_argument('--master', type=str, 
                        default='')
    parser.add_argument('--ps_tasks', type=int, 
                        default=0)
    parser.add_argument('--task', type=int, 
                        default=0)

    FLAGS, unparsed = parser.parse_known_args()

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

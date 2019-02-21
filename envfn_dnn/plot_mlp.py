import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from functools import partial
import numpy as np
import envfn_dnn
import tensorflow as tf
import tensorflow.contrib.slim as slim
import pandas as pd
import matplotlib.pyplot as plt
import math
import conf
from tools import tool_fn

FLAGS = None

def data_preproc(data, translate, scale):

    return (data - translate) / scale


def data_transform(data, translate, scale):

    return (data * scale) + translate


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


def make_batch(data_tuple):

    tf.logging.info("batch making...")
    dataset = tf.data.Dataset.from_tensor_slices(data_tuple)
    batch = dataset.batch(FLAGS.batch_size)
    tf.logging.info("batch making done")
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

    predict_data = sample_open(FLAGS.sample_idxs, col_features, col_labels, p_dic)
    data_length = predict_data[0].shape[0]

    graph = tf.Graph()
    with graph.as_default():
        with tf.device(tf.train.replica_device_setter(0, merge_devices=True)):

            architecture = getattr(
                envfn_dnn.architecture, m_dic.get('arch'))
            model_function = partial(architecture, predict_size=len(cols.get('envfn_l')))
            model = envfn_dnn.model.MlpModel(model_function)

            batch = make_batch(predict_data)

            iterator = batch.make_one_shot_iterator()
            next_batch = iterator.get_next()
            feature_batch, label_batch = next_batch

            feature_translate_restore = tf.get_variable(
                'feature_translate_save', shape=[1, feature_batch.shape[1]], trainable=False)
            feature_scale_restore = tf.get_variable(
                'feature_scale_save', shape=[1, feature_batch.shape[1]], trainable=False)
            label_translate_restore = tf.get_variable(
                'label_translate_save', shape=[1, label_batch.shape[1]], trainable=False)
            label_scale_restore = tf.get_variable(
                'label_scale_save', shape=[1, label_batch.shape[1]], trainable=False)

            preproc_feature_batch = data_preproc(
                feature_batch, feature_translate_restore, feature_scale_restore)
            prediction_intermediate_batch = model.predict_model(
                preproc_feature_batch, is_training=False)
            prediction_batch = data_transform(
                prediction_intermediate_batch, label_translate_restore, label_scale_restore)

            num_batches = math.ceil(
                data_length * len(FLAGS.sample_idxs) / float(FLAGS.batch_size))

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.log_device_placement = True

            saver = tf.train.Saver()

            with tf.Session() as sess:

                saver.restore(sess, tf.train.latest_checkpoint(p_dic.get('envfn_log_dir')))

                label_stack, prediction_stack = sess.run(
                    [label_batch, prediction_batch])

                with slim.queues.QueueRunners(sess):
                    for i in range(num_batches - 1):
                        labels, predictions = sess.run(
                            [label_batch, prediction_batch]
                        )
                        label_stack = np.concatenate((label_stack, labels))
                        prediction_stack = np.concatenate(
                            (prediction_stack, predictions))

                indexing = np.arange(data_length)

                if not os.path.exists(p_dic.get('envfn_graph_dir') + '/' + str(FLAGS.sample_idxs)):
                    os.makedirs(p_dic.get('envfn_graph_dir') + '/' + str(FLAGS.sample_idxs))

                interval = indexing % FLAGS.inst_interval == 0

                prediction_stack_accum = np.zeros_like(prediction_stack)
                prediction_stack_accum[0] = prediction_stack[0]
                for i in range(len(prediction_stack) - 1):
                    prediction_stack_accum[i +
                                           1] = prediction_stack_accum[i] + prediction_stack[i+1]
                label_stack_accum = np.zeros_like(label_stack)
                label_stack_accum[0] = label_stack[0]
                for i in range(len(label_stack) - 1):
                    label_stack_accum[i +
                                      1] = label_stack_accum[i] + label_stack[i+1]

                for label_idx in range(len(cols.get('envfn_l'))):
                    
                    plt.title(cols.get('envfn_l')[label_idx] + ' (instant)')
                    plt.plot(indexing[interval],
                             prediction_stack[interval][:, label_idx], label="prediction")
                    plt.plot(indexing[interval],
                             label_stack[interval][:, label_idx], label="label")
                    plt.legend(loc=2)

                    # avg_inst_err = 100 * np.mean(np.absolute(prediction_stack - label_stack) / label_stack, axis=0)
                    # plt_inst_text = "average instance error : %.2f percent " % avg_inst_err[label_idx]
                    # plt.text(0, 0, plt_inst_text)

                    plt.figure(num=1, figsize=(36, 6), dpi=2000, facecolor='white')
                    plt.savefig(p_dic.get('envfn_graph_dir') + '/' + str(FLAGS.sample_idxs) + '/' + cols.get('envfn_l')[label_idx] + '_inst.png', bbox_inches='tight')
                    plt.close()

                    print(p_dic.get('envfn_graph_dir') + '/' + str(FLAGS.sample_idxs) + '/' + cols.get('envfn_l')[label_idx] + '_inst.png' + ' saved')

                    plt.title(cols.get('envfn_l')[label_idx] + ' (accumulative)')
                    plt.plot(indexing,
                             prediction_stack_accum[:, label_idx], label="prediction")
                    plt.plot(indexing,
                             label_stack_accum[:, label_idx], label="label")
                    plt.legend(loc=2)

                    accum_err = 100 * \
                        (prediction_stack_accum[-1] -
                         label_stack_accum[-1]) / label_stack_accum[-1]
                    plt_accum_text_lines = list()
                    plt_accum_text_lines.append(
                        "label accum : %.0f W" % label_stack_accum[-1, label_idx])
                    plt_accum_text_lines.append(
                        "prediction accum : %.0f W" % prediction_stack_accum[-1, label_idx])
                    plt_accum_text_lines.append(
                        "accumulative error : %.2f percent" % accum_err[label_idx])
                    plt_accum_text = "\n".join(plt_accum_text_lines)
                    plt.text(0, 0, plt_accum_text)
                    plt.figure(num=1, figsize=(36, 6),
                               dpi=2000, facecolor='white')
                    plt.savefig(p_dic.get('envfn_graph_dir') + '/' + str(FLAGS.sample_idxs) + '/' + cols.get('envfn_l')[label_idx] + '_accum.png', bbox_inches='tight')
                    plt.close()

                    print(p_dic.get('envfn_graph_dir') + '/' + str(FLAGS.sample_idxs) + '/' + cols.get('envfn_l')[label_idx] + '_accum.png' + ' saved')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str,
                        default='ep', help='environment name')
    parser.add_argument('--sample_idxs', nargs='+', type=int, default=[9])
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--inst_interval', type=int, default=200)
    FLAGS, unparsed = parser.parse_known_args()

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

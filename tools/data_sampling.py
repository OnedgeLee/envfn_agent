import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import pandas as pd
import conf
from tools import tool_fn

FLAGS = None
sample_seed = np.random.RandomState()

def main():

    p_dic = getattr(conf.dic.path_dic, FLAGS.env_name)
    c_dic = getattr(conf.dic.col_dic, FLAGS.env_name)
    cols = {
        'ENV_F': getattr(conf.list.envfn.feature, FLAGS.env_name),
        'ENV_L': getattr(conf.list.envfn.label, FLAGS.env_name),
        'AGENT_IVS': getattr(conf.list.agent.iv_state, FLAGS.env_name)
    }

    features = [c_dic[feature] for feature in cols.get('ENV_F')]
    labels = [c_dic[label] for label in cols.get('ENV_L')]
    independents = [c_dic[ivs] for ivs in cols.get('AGENT_IVS')]

    preprocs = os.listdir(p_dic.get('preproc_data_dir'))
    preprocs.sort()

    print('Extract preprocessed data from dir: ' + p_dic.get('preproc_data_dir'))
    tool_fn.progress_bar(0, 1, prefix='>>> Extracting...')
    for preproc in range(len(preprocs)):

        csv_path = p_dic.get('preproc_data_dir') + '/' +  preprocs[preproc]
        preproc_feature_df_part = pd.read_csv(csv_path, dtype=np.float32,
                            delimiter=',', error_bad_lines=False, usecols=features)[features][:-1]
        preproc_label_df_part = pd.read_csv(csv_path, dtype=np.float32,
                            delimiter=',', error_bad_lines=False, usecols=labels)[labels][1:]
        preproc_ivs_df_part = pd.read_csv(csv_path, dtype=np.float32,
                            delimiter=',', error_bad_lines=False, usecols=independents)[independents][:-1]

        if preproc == 0:
            preproc_feature_df = preproc_feature_df_part
            preproc_label_df = preproc_label_df_part
            preproc_ivs_df = preproc_ivs_df_part
        else:
            preproc_feature_df = pd.concat([preproc_feature_df, preproc_feature_df_part])
            preproc_label_df = pd.concat([preproc_label_df, preproc_label_df_part])
            preproc_ivs_df = pd.concat([preproc_ivs_df, preproc_ivs_df_part])
    
        tool_fn.progress_bar(preproc + 1, len(preprocs), prefix='>>> Extracting...')

    if not os.path.exists(p_dic.get('sample_data_dir') + '/feature'):
            os.makedirs(p_dic.get('sample_data_dir') + '/feature')
    if not os.path.exists(p_dic.get('sample_data_dir') + '/label'):
            os.makedirs(p_dic.get('sample_data_dir') + '/label')
    if not os.path.exists(p_dic.get('sample_data_dir') + '/ivs'):
            os.makedirs(p_dic.get('sample_data_dir') + '/ivs')
            
    print('Sample to dir: ' + p_dic.get('sample_data_dir'))
    tool_fn.progress_bar(0, 1, prefix='>>> Sampling...')
    for split in range(FLAGS.frac_num):

        sampled_feature_df = preproc_feature_df.sample(frac=1./FLAGS.frac_num, random_state=sample_seed)
        preproc_feature_df.drop(sampled_feature_df.index)
        sampled_feature_df.to_csv(p_dic.get('sample_data_dir') + '/feature/' + FLAGS.sample_name + '_feature_' + str(split) + '.csv')

        sampled_label_df = preproc_label_df.iloc[sampled_feature_df.index]
        sampled_label_df.to_csv(p_dic.get('sample_data_dir') + '/label/' + FLAGS.sample_name + '_label_' + str(split) + '.csv')

        tool_fn.progress_bar(split + 1, FLAGS.frac_num, prefix='>>> Sampling...')

    preproc_ivs_df.to_csv(p_dic.get('sample_data_dir') + '/ivs/' + 'extracted_ivs' + '.csv')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default="ep")
    parser.add_argument('--sample_name', type=str, default="sampled")
    parser.add_argument('--frac_num', type=int, default=10)
    FLAGS, unparsed = parser.parse_known_args()
    
    main()

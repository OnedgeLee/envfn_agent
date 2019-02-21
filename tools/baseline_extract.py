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

    origins = os.listdir(p_dic.get('origin_data_dir'))
    origins.sort()

    print('Extract original data from dir: ' + p_dic.get('origin_data_dir'))
    tool_fn.progress_bar(0, 1, prefix='>>> Extracting...')
    for origin in range(len(origins)):

        csv_path = p_dic.get('origin_data_dir') + '/' +  origins[origin]
        origin_feature_df_part = pd.read_csv(csv_path, dtype=np.float32,
                            delimiter=',', error_bad_lines=False, usecols=features)[features]
        origin_label_df_part = pd.read_csv(csv_path, dtype=np.float32,
                            delimiter=',', error_bad_lines=False, usecols=labels)[labels]
        origin_ivs_df_part = pd.read_csv(csv_path, dtype=np.float32,
                            delimiter=',', error_bad_lines=False, usecols=independents)[independents]

        origin_feature_df_part = origin_feature_df_part.loc[origin_feature_df_part.index %
                                4 == 3].iloc[:-1]
        origin_label_df_part = origin_label_df_part.loc[origin_label_df_part.index %
                                4 == 3].iloc[1:]
        origin_ivs_df_part = origin_ivs_df_part.loc[origin_ivs_df_part.index %
                                4 == 3].iloc[:-1]

        if origin == 0:
            origin_feature_df = origin_feature_df_part
            origin_label_df = origin_label_df_part
            origin_ivs_df = origin_ivs_df_part
        else:
            origin_feature_df = pd.concat([origin_feature_df, origin_feature_df_part])
            origin_label_df = pd.concat([origin_label_df, origin_label_df_part])
            origin_ivs_df = pd.concat([origin_ivs_df, origin_ivs_df_part])
    
        tool_fn.progress_bar(origin + 1, len(origins), prefix='>>> Extracting...')

    origin_feature_df = origin_feature_df.reset_index(drop=True)
    origin_label_df = origin_label_df.reset_index(drop=True)

    if not os.path.exists(p_dic.get('sample_data_dir') + '/feature'):
            os.makedirs(p_dic.get('sample_data_dir') + '/feature')
    if not os.path.exists(p_dic.get('sample_data_dir') + '/label'):
            os.makedirs(p_dic.get('sample_data_dir') + '/label')
    if not os.path.exists(p_dic.get('sample_data_dir') + '/ivs'):
            os.makedirs(p_dic.get('sample_data_dir') + '/ivs')
            
    print('Sample to dir: ' + p_dic.get('sample_data_dir'))
    tool_fn.progress_bar(0, 1, prefix='>>> Sampling...')
    
    for split in range(FLAGS.frac_num):

        sampled_feature_df = origin_feature_df.sample(frac=1./FLAGS.frac_num, random_state=sample_seed)
        origin_feature_df.drop(sampled_feature_df.index)
        sampled_feature_df.to_csv(p_dic.get('sample_data_dir') + '/feature/' + FLAGS.sample_name + '_feature_' + str(split) + '.csv')

        sampled_label_df = origin_label_df.iloc[sampled_feature_df.index]
        sampled_label_df.to_csv(p_dic.get('sample_data_dir') + '/label/' + FLAGS.sample_name + '_label_' + str(split) + '.csv')

        tool_fn.progress_bar(split + 1, FLAGS.frac_num, prefix='>>> Sampling...')

    origin_ivs_df.to_csv(p_dic.get('sample_data_dir') + '/ivs/' + 'extracted_ivs' + '.csv')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default="ep")
    parser.add_argument('--sample_name', type=str, default="sampled")
    parser.add_argument('--frac_num', type=int, default=10)
    FLAGS, unparsed = parser.parse_known_args()
    
    main()
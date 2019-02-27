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
    s_dic = getattr(conf.dic.space_dic, FLAGS.env_name)

    cols = {
        'ORIGIN_F': getattr(conf.list.preproc.origin, FLAGS.env_name),
        'PREPROC_F': getattr(conf.list.preproc.preproc, FLAGS.env_name)
    }
    origin_f = [c_dic[feature] for feature in cols.get('ORIGIN_F')]
    origins = os.listdir(p_dic.get('origin_data_dir'))
    origins.sort()

    print('Extract original data from dir: ' + p_dic.get('origin_data_dir'))
    tool_fn.progress_bar(0, 1, prefix='>>> Extracting...')

    for origin in range(len(origins)):

        csv_path = p_dic.get('origin_data_dir') + '/' +  origins[origin]
        origin_df_part = pd.read_csv(csv_path, delimiter=',', error_bad_lines=False, usecols=origin_f)[origin_f]
        origin_df_part = origin_df_part.loc[origin_df_part.index % 4 == 3]

        if origin == 0:
            origin_df = origin_df_part
        else:
            origin_df = pd.concat([origin_df, origin_df_part])

        tool_fn.progress_bar(origin + 1, len(origins), prefix='>>> Extracting...')


    striped_date = pd.Series(origin_df['Date/Time'].str.strip())
    splited_date = pd.DataFrame(striped_date.str.split(r'[/  :]').tolist(), columns = ['Month', 'Day', '', 'Hour', '', ''])
    splited_date = splited_date.drop([''], axis=1)
    splited_date.reset_index(drop=True, inplace=True)
    origin_df = origin_df.drop(['Date/Time'], axis=1)
    origin_df.reset_index(drop=True, inplace=True)
    origin_df = pd.concat([splited_date, origin_df], axis=1)

    if not os.path.exists(p_dic.get('preproc_data_dir')):
        os.makedirs(p_dic.get('preproc_data_dir'))

    print('Preprocess to dir: ' + p_dic.get('preproc_data_dir'))
    tool_fn.progress_bar(0, 1, prefix='>>> Preprocessing...')

    f_space = np.array([s_dic[feature] for feature in cols.get('PREPROC_F')], dtype=np.float32).T
    origin_df_size = origin_df.shape[0]
    for split in range(FLAGS.frac_num):

        preprocessed_df = origin_df.head(np.ceil(origin_df_size / FLAGS.frac_num).astype(int)).apply(pd.to_numeric)
        preprocessed_df = tool_fn.normalize(preprocessed_df, f_space[0], f_space[1])
        origin_df.drop(preprocessed_df.index)
        preprocessed_df.to_csv(p_dic.get('preproc_data_dir') + '/' + FLAGS.preproc_name + '_' + str(split) + '.csv')

        tool_fn.progress_bar(split + 1, FLAGS.frac_num, prefix='>>> Preprocessing...')

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default="ep")
    parser.add_argument('--preproc_name', type=str, default="preprocessed")
    parser.add_argument('--frac_num', type=int, default=1)
    FLAGS, unparsed = parser.parse_known_args()
    
    main()
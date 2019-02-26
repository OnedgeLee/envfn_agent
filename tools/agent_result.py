import os
import shutil
import gzip
import math
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
plt.switch_backend('agg')

column_names = {
    'out_temp': "Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)",
    'east_temp': "EAST ZONE:Zone Air Temperature [C](TimeStep)",
    'west_temp': "WEST ZONE:Zone Air Temperature [C](TimeStep)",
    'east_dec_set_t': "EAST ZONE DEC OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)",
    'east_iec_set_t': "EAST ZONE IEC OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)",
    'east_ccoil_set_t': "EAST ZONE CCOIL AIR OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)",
    'east_airloop_set_t': "EAST AIR LOOP OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)",
    'east_fan_set_f': "EAST ZONE SUPPLY FAN:Fan Air Mass Flow Rate [kg/s](Hourly)",
    'east_ite': "EAST ZONE:Zone ITE Total Heat Gain to Zone Rate [W](Hourly)",
    'west_dec_set_t': "WEST ZONE DEC OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)",
    'west_iec_set_t': "WEST ZONE IEC OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)",
    'west_ccoil_set_t': "WEST ZONE CCOIL AIR OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)",
    'west_airloop_set_t': "WEST AIR LOOP OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)",
    'west_fan_set_f': "WEST ZONE SUPPLY FAN:Fan Air Mass Flow Rate [kg/s](Hourly)",
    'west_ite': "WEST ZONE:Zone ITE Total Heat Gain to Zone Rate [W](Hourly)",
    'total_ite': "Whole Building:Facility Total Building Electric Demand Power [W](Hourly)",
    'total_hvac': "Whole Building:Facility Total HVAC Electric Demand Power [W](Hourly)"
}

output_names = {
    'out_temp': 'Outdoor_Temperature',
    'east_temp': 'East_Zone_Temperature',
    'west_temp': 'West_Zone_Temperature',
    'east_dec_set_t': 'East_DEC_Set_Temperature',
    'east_iec_set_t': 'East_IEC_Set_Temperature',
    'east_ccoil_set_t': 'East_Ccoil_Set_Temperature',
    'east_airloop_set_t': 'East_Airloop_Set_Temperature',
    'east_fan_set_f': 'East_Fan_Set_Flowrate',
    'west_dec_set_t': 'West_DEC_Set_Temperature',
    'west_iec_set_t': 'West_IEC_Set_Temperature',
    'west_ccoil_set_t': 'West_Ccoil_Set_Temperature',
    'west_airloop_set_t': 'West_Airloop_Set_Temperature',
    'west_fan_set_f': 'West_Fan_Set_Flowrate',
    'total_ite': 'ITE_Total_Power_Consumption',
    'total_hvac': 'HVAC_Total_Power_Consumption',
    'hvac_accum': 'HVAC_Total_Power_Consumption_Accum',
    'pue': 'Power_Utilization_Effectiveness',
}


def select_columns_from_files(*, file_paths=None, column_name=None):

    column_name = [column_name]*len(file_paths)

    data = dict()
    for i, path_column in enumerate(zip(file_paths, column_name)):
        path, column = path_column
        csv = pd.read_csv(path, usecols=['Date/Time', column])
        key = path.split('_')[-1][:-4]

        if i == 0:
            striped_date = pd.Series(csv['Date/Time'].str.strip())
            splited_date = pd.DataFrame(striped_date.str.split(' ', 1).tolist(), columns = ['Date','Time'])
            data['Date'] = splited_date['Date']
            data['Time'] = splited_date['Time']
            data['{0}'.format(key)] = csv[column]
        else:
            data['{0}'.format(key)] = csv[column]

    return data

def compare_before_after(type,dirname,logidx):

    column_name = column_names[type]
    output_name = output_names[type]

    eplusout_path = '/home/csle/ksb-csle_v0_8/Datacenter/eplusout'

    data = select_columns_from_files(
        file_paths=['/hdd/project/datacenter/rl/log/energyplus/output/episode/eplusout_ours.csv', eplusout_path + '/eplusout_last.csv', eplusout_path + '/eplusout_base.csv'],
        column_name=column_name
    )
    pue = pd.DataFrame(data=data)
    data = pue.groupby(['Date']).mean()
    data.reset_index(level=0, inplace=True)

    plt.title(output_name)
    plt.plot(data['Date'], data['ours'], label="ours")
    plt.plot(data['Date'], data['last'], label="ibms")
    plt.plot(data['Date'], data['base'], label="eplus")

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('date')

    out_key = output_name.split('_')[-1][:4]

    if out_key == 'Temp':
        plt.ylabel('temperature, C')
    elif out_key == 'Cons': 
        plt.ylabel('power, W')
    elif out_key == 'Flow': 
        plt.ylabel('flowrate, kg/s')
    elif out_key == 'Effe': 
        plt.ylabel('value')
    else:
        plt.ylabel('result')
    plt.figure(num=1,figsize=(36,6), dpi=500, facecolor='white')
    plt.savefig('/tmp/'+dirname+'/'+output_name+'_comp_ibm.png', bbox_inches='tight')
    plt.close()
    print(output_name + '_before_after is saved')

# def stepwise(type,dirname,logidx):

#     column_name = column_names[type]
#     output_name = output_names[type]

#     eplusout_path = '/hdd/project/datacenter/tmpeplog/output'

#     data = select_columns_from_files(
#         file_paths=[eplusout_path + '/eplusout_earl.csv', eplusout_path + '/eplusout_midd.csv', eplusout_path + '/eplusout_last.csv'],
#         column_name=column_name
#     )
#     pue = pd.DataFrame(data=data)
#     data = pue.groupby(['Date']).mean()
#     data.reset_index(level=0, inplace=True)

#     plt.title(output_name)
#     plt.plot(data['Date'], data['earl'], label="early")
#     plt.plot(data['Date'], data['midd'], label="middle")
#     plt.plot(data['Date'], data['last'], label="last")

#     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     plt.xlabel('date')

#     out_key = output_name.split('_')[-1][:4]

#     if out_key == 'Temp':
#         plt.ylabel('temperature, C')
#     elif out_key == 'Cons': 
#         plt.ylabel('power, W')
#     elif out_key == 'Flow': 
#         plt.ylabel('flowrate, kg/s')
#     elif out_key == 'Effe': 
#         plt.ylabel('value')
#     else:
#         plt.ylabel('result')
#     plt.figure(num=1,figsize=(36,6), dpi=500, facecolor='white')
#     plt.savefig('/tmp/'+dirname+'/'+output_name+'_stepwise.png', bbox_inches='tight')
#     plt.close()
#     print('/tmp/'+dirname+'/'+output_name+'_stepwise.png saved')

# def accumulative(dirname,logidx):

#     column_name = column_names['hvac']
#     output_name = output_names['hvac_accum']

#     eplusout_path = '/'.join(os.getcwd().split('/')[0:-1]) + '/eplusout'

#     data = select_columns_from_files(
#         file_paths=[ '/hdd/project/datacenter/tmpeplog/output/episode/eplusout_ours.csv', eplusout_path + '/eplusout_last.csv'],
#         column_name=column_name
#     )
#     pue = pd.DataFrame(data=data)
#     data = pue.groupby(['Date']).mean()
#     data.reset_index(level=0, inplace=True)

#     plt.title(output_name)

#     for i in range(len(data['ours'])):
#         if (i == 0):
#             continue
#         else:
#             data['ours'][i] = data['ours'][i] + data['ours'][i - 1]

#     for i in range(len(data['last'])):
#         if (i == 0):
#             continue
#         else:
#             data['last'][i] = data['last'][i] + data['last'][i - 1]

#     data['ours'] = data['ours'] * 3600 / 1000
#     data['last'] = data['last'] * 3600 / 1000

#     plt.plot(data['Date'], data['ours'], label="ours")
#     plt.plot(data['Date'], data['last'], label="ibms")

#     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     plt.xlabel('date')

#     plt.ylabel('energy, kWh')
#     plt.figure(num=1,figsize=(36,6), dpi=500, facecolor='white')
#     our_acc = data['ours'][len(data['ours']) - 1]
#     ibm_acc = data['last'][len(data['last']) - 1]
#     ibm_diff = (our_acc - ibm_acc) / ibm_acc * 100
#     text = 'Our Total HVAC Accum : %.0f kWh\n' % our_acc +\
#             'IBM Total HVAC Accum : %.0f kWh\n' % ibm_acc +\
#             'Difference : %.2f percent' % ibm_diff
#     plt.text(0,0,text)
    
#     plt.savefig('/tmp/'+dirname+'/'+output_name+'_comp_ibm.png', bbox_inches='tight')
#     plt.close()
#     print(output_name + '_before_after is saved')


# def base_comp_accum(dirname,logidx):

#     column_name = column_names['hvac']
#     output_name = output_names['hvac_accum']

#     eplusout_path = '/'.join(os.getcwd().split('/')[0:-1]) + '/eplusout'

#     data = select_columns_from_files(
#         file_paths=[ '/hdd/project/datacenter/tmpeplog/output/episode/eplusout_ours.csv', eplusout_path + '/eplusout_base.csv'],
#         column_name=column_name
#     )
#     pue = pd.DataFrame(data=data)
#     data = pue.groupby(['Date']).mean()
#     data.reset_index(level=0, inplace=True)

#     plt.title(output_name)

#     for i in range(len(data['ours'])):
#         if (i == 0):
#             continue
#         else:
#             data['ours'][i] = data['ours'][i] + data['ours'][i - 1]

#     for i in range(len(data['base'])):
#         if (i == 0):
#             continue
#         else:
#             data['base'][i] = data['base'][i] + data['base'][i - 1]

#     data['ours'] = data['ours'] * 3600 / 1000
#     data['base'] = data['base'] * 3600 / 1000

#     plt.plot(data['Date'], data['ours'], label="ours")
#     plt.plot(data['Date'], data['base'], label="eplus")

#     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     plt.xlabel('date')

#     plt.ylabel('energy, kWh')
#     plt.figure(num=1,figsize=(36,6), dpi=500, facecolor='white')
#     our_acc = data['ours'][len(data['ours']) - 1]
#     eplus_acc = data['base'][len(data['base']) - 1]
#     eplus_diff = (our_acc - eplus_acc) / eplus_acc * 100
#     text = 'Our Total HVAC Accum : %.0f kWh\n' % our_acc +\
#             'Eplus Total HVAC Accum : %.0f kWh\n' % eplus_acc +\
#             'Difference : %.2f percent' % eplus_diff
#     plt.text(0,0,text)
#     # plt.text(0, 0, 'Total HVAC Accum : %.0f kWh' % data['base'][len(data['base']) - 1])
#     plt.savefig('/tmp/'+dirname+'/'+output_name+'_comp_eplus.png', bbox_inches='tight')
#     plt.close()
#     print(output_name + '_before_after is saved')
    
# def acc_stepwise(dirname,logidx):

#     column_name = column_names['hvac']
#     output_name = output_names['hvac_accum']

#     eplusout_path = '/hdd/project/datacenter/tmpeplog/output'

#     data = select_columns_from_files(
#         file_paths=[eplusout_path + '/eplusout_earl.csv', eplusout_path + '/eplusout_midd.csv', eplusout_path + '/eplusout_last.csv'],
#         column_name=column_name
#     )
#     pue = pd.DataFrame(data=data)
#     data = pue.groupby(['Date']).mean()
#     data.reset_index(level=0, inplace=True)

#     plt.title(output_name)

#     for i in range(len(data['earl'])):
#         if (i == 0):
#             continue
#         else:
#             data['earl'][i] = data['earl'][i] + data['earl'][i - 1]

#     for i in range(len(data['midd'])):
#         if (i == 0):
#             continue
#         else:
#             data['midd'][i] = data['midd'][i] + data['midd'][i - 1]

#     for i in range(len(data['last'])):
#         if (i == 0):
#             continue
#         else:
#             data['last'][i] = data['last'][i] + data['last'][i - 1]

#     data['earl'] = data['earl'] * 3600 / 1000
#     data['midd'] = data['midd'] * 3600 / 1000
#     data['last'] = data['last'] * 3600 / 1000

#     plt.plot(data['Date'], data['earl'], label="early")
#     plt.plot(data['Date'], data['midd'], label="middle")
#     plt.plot(data['Date'], data['last'], label="last")

#     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     plt.xlabel('date')

#     plt.ylabel('energy, kWh')
#     plt.figure(num=1,figsize=(36,6), dpi=500, facecolor='white')
#     our_acc = data['earl'][len(data['earl']) - 1]
#     eplus_acc = data['midd'][len(data['midd']) - 1]
#     ibm_acc = data['last'][len(data['last']) - 1]
#     eplus_diff = (our_acc - eplus_acc) / eplus_acc * 100
#     text = 'Early Total HVAC Accum : %.0f kWh\n' % our_acc +\
#             'Middle Total HVAC Accum : %.0f kWh\n' % eplus_acc +\
#             'Last Total HVAC Accum : %.0f kWh' % ibm_acc
#     plt.text(0,0,text)
#     plt.savefig('/tmp/'+dirname+'/'+output_name+'_stepwise_acc.png', bbox_inches='tight')
#     plt.close()
#     print('/tmp/'+dirname+'/'+output_name+'_stepwise_acc.png saved')

def acc_compare_all(dirname,logidx):

    column_name = column_names['total_hvac']
    output_name = output_names['hvac_accum']

    eplusout_path = '/home/csle/ksb-csle_v0_8/Datacenter/eplusout'

    data = select_columns_from_files(
        file_paths=['/hdd/project/datacenter/rl/log/energyplus/output/episode/eplusout_ours.csv', eplusout_path + '/eplusout_last.csv', eplusout_path + '/eplusout_base.csv'],
        column_name=column_name
    )
    pue = pd.DataFrame(data=data)
    data = pue.groupby(['Date']).mean()
    data.reset_index(level=0, inplace=True)

    plt.title(output_name)

    for i in range(len(data['ours'])):
        if (i == 0):
            continue
        else:
            data['ours'][i] = data['ours'][i] + data['ours'][i - 1]

    for i in range(len(data['base'])):
        if (i == 0):
            continue
        else:
            data['base'][i] = data['base'][i] + data['base'][i - 1]

    for i in range(len(data['last'])):
        if (i == 0):
            continue
        else:
            data['last'][i] = data['last'][i] + data['last'][i - 1]

    data['ours'] = data['ours'] * 3600 / 1000
    data['base'] = data['base'] * 3600 / 1000
    data['last'] = data['last'] * 3600 / 1000

    plt.plot(data['Date'], data['ours'], label="ours")
    plt.plot(data['Date'], data['base'], label="eplus")
    plt.plot(data['Date'], data['last'], label="ibm")

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('date')

    plt.ylabel('energy, kWh')
    plt.figure(num=1,figsize=(36,6), dpi=500, facecolor='white')
    our_acc = data['ours'][len(data['ours']) - 1]
    eplus_acc = data['base'][len(data['base']) - 1]
    ibm_acc = data['last'][len(data['last']) - 1]
    eplus_diff = (our_acc - eplus_acc) / eplus_acc * 100
    text = 'Our Total HVAC Accum : %.0f kWh\n' % our_acc +\
            'Eplus Total HVAC Accum : %.0f kWh\n' % eplus_acc +\
            'IBM Total HVAC Accum : %.0f kWh' % ibm_acc
    plt.text(0,0,text)
    plt.savefig('/tmp/'+dirname+'/'+output_name+'_comp_all.png', bbox_inches='tight')
    plt.close()
    print(output_name + '_before_after is saved')

# def baseline_export():

#     cols = ['eastemp', 'westemp', 'hvac']
#     column_name = [column_names[col] for col in cols]

#     eplusout_path = '/'.join(os.getcwd().split('/')[0:-1]) + '/eplusout'

#     # data = select_columns_from_files(
#     #     file_paths=[eplusout_path + '/eplusout_base.csv'],
#     #     column_name=column_name
#     # )

#     data = pd.read_csv(eplusout_path + '/eplusout_base.csv', dtype=np.float32,
#                         delimiter=',', error_bad_lines=False, usecols=column_name)[column_name]

    
#     data = data.loc[data.index % 4 == 3].iloc[:]
#     print(data)
#     data.to_csv('/home/csle/ksb-csle_v0_8/Datacenter/rl-testbed-for-energyplus' + '/base_ext.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', help='column name', type=str, default='total')
    parser.add_argument('--dir', help='dir name', type=str, default='comparison')
    parser.add_argument('--idx', help='log index', type=int, default=-1)
    args, unparsed = parser.parse_known_args()

    if not os.path.exists('/tmp/'+args.dir):
        os.makedirs('/tmp/'+args.dir)

    # baseline_export()
  
    compare_before_after('east_temp', args.dir, args.idx)
    compare_before_after('west_temp', args.dir, args.idx)
    compare_before_after('total_hvac', args.dir, args.idx)
    compare_before_after('east_dec_set_t', args.dir, args.idx)
    compare_before_after('east_iec_set_t', args.dir, args.idx)
    compare_before_after('east_ccoil_set_t', args.dir, args.idx)
    compare_before_after('east_airloop_set_t', args.dir, args.idx)
    compare_before_after('east_fan_set_f', args.dir, args.idx)
    compare_before_after('west_dec_set_t', args.dir, args.idx)
    compare_before_after('west_iec_set_t', args.dir, args.idx)
    compare_before_after('west_ccoil_set_t', args.dir, args.idx)
    compare_before_after('west_airloop_set_t', args.dir, args.idx)
    compare_before_after('west_fan_set_f', args.dir, args.idx)
    
    # stepwise('total', args.dir, args.idx)
    # stepwise('hvac', args.dir, args.idx)
    # stepwise('ite', args.dir, args.idx)
    # stepwise('westemp', args.dir, args.idx)
    # stepwise('eastemp', args.dir, args.idx)
    # stepwise('outemp', args.dir, args.idx)
    # stepwise('pue', args.dir, args.idx)

    # stepwise('east_dec_set_t', args.dir, args.idx)
    # stepwise('east_iec_set_t', args.dir, args.idx)
    # stepwise('east_ccoil_set_t', args.dir, args.idx)
    # stepwise('east_airloop_set_t', args.dir, args.idx)
    # stepwise('east_fan_set_f', args.dir, args.idx)
    # stepwise('west_dec_set_t', args.dir, args.idx)
    # stepwise('west_iec_set_t', args.dir, args.idx)
    # stepwise('west_ccoil_set_t', args.dir, args.idx)
    # stepwise('west_airloop_set_t', args.dir, args.idx)
    # stepwise('west_fan_set_f', args.dir, args.idx)

    # accumulative(args.dir, args.idx)
    # base_comp_accum(args.dir, args.idx)
    acc_compare_all(args.dir, args.idx)
    # acc_stepwise(args.dir, args.idx)


    # for type, dirname, logidx in args.type.split(' '):
    #     os.makedirs('/tmp/'+dirname, logidx)
    #     compare_episode('total', dirname, logidx)
    #     compare_episode('hvac', dirname, logidx)
    #     compare_episode('ite', dirname, logidx)
    #     compare_episode('westemp', dirname, logidx)
    #     compare_episode('eastemp', dirname, logidx)
    #     compare_episode('outemp', dirname, logidx)
        # compare_before_after(type)

    



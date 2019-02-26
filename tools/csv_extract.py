import os
import pandas as pd
import numpy as np

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

def base_export():

    cols = ['east_temp', 'west_temp', 'total_hvac']
    column_name = [column_names[col] for col in cols]

    eplusout_path = '/home/csle/ksb-csle_v0_8/Datacenter/eplusout'

    # data = select_columns_from_files(
    #     file_paths=[eplusout_path + '/eplusout_base.csv'],
    #     column_name=column_name
    # )

    data = pd.read_csv(eplusout_path + '/eplusout_base.csv', dtype=np.float32,
                        delimiter=',', error_bad_lines=False, usecols=column_name)[column_name]

    
    data = data.loc[data.index % 4 == 3].iloc[:]
    data.to_csv('/hdd/project/datacenter/rl/csv' + '/base_ext.csv')

def agent_export():

    cols = ['east_temp', 'west_temp', 'total_hvac']
    column_name = [column_names[col] for col in cols]

    eplusout_path = '/hdd/project/datacenter/rl/log/energyplus/output/episode'

    # data = select_columns_from_files(
    #     file_paths=[eplusout_path + '/eplusout_base.csv'],
    #     column_name=column_name
    # )

    data = pd.read_csv(eplusout_path + '/eplusout.csv', dtype=np.float32,
                        delimiter=',', error_bad_lines=False, usecols=column_name)[column_name]

    
    data = data.loc[data.index % 4 == 3].iloc[:]
    data.to_csv('/hdd/project/datacenter/rl/csv' + '/agent_ext.csv')

if not os.path.exists('/hdd/project/datacenter/rl/csv'):
    os.makedirs('/hdd/project/datacenter/rl/csv')
    
base_export()
agent_export()

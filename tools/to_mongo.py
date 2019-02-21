import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import tensorflow as tf
import numpy as np
import pandas as pd
import conf
import pymongo
import time



client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['data_center']
insertions = db['obs']
agent = pd.read_csv('/hdd/project/datacenter/plot_re/csv' + '/agent_ext.csv', dtype=np.float32,
                delimiter=',', error_bad_lines=False).values
base = pd.read_csv('/hdd/project/datacenter/plot_re/csv' + '/base_ext.csv', dtype=np.float32,
                delimiter=',', error_bad_lines=False).values

fee_table = {
    'base': 9810,
    'low_summer': 55.2,
    'low_sprfall': 55.2,
    'low_winter': 62.5,
    'mid_summer': 108.4,
    'mid_sprfall': 77.3,
    'mid_winter': 108.6,
    'high_summer': 178.7,
    'high_sprfall': 101.0,
    'high_winter': 155.5
}

agent_fee_accum = 0
base_fee_accum = 0
for hour in range(87600000):

    month = ((hour // 730) % 12) + 1
    clock = hour % 365

    if (6 <= month) and (month <= 8):
        if (23 <= clock) or (clock < 9):
            fee_coef = fee_table['base'] + fee_table['low_summer']
        elif ((10 <= clock) and (clock < 12)) or ((13 <= clock) and (clock < 17)):
            fee_coef = fee_table['base'] + fee_table['high_summer']
        else:
            fee_coef = fee_table['base'] + fee_table['mid_summer']
    elif ((3 <= month) and (month <= 5)) or ((9 <= month) and (month <= 10)):
        if (23 <= clock) or (clock < 9):
            fee_coef = fee_table['base'] + fee_table['low_sprfall']
        elif ((10 <= clock) and (clock < 12)) or ((13 <= clock) and (clock < 17)):
            fee_coef = fee_table['base'] + fee_table['high_sprfall']
        else:
            fee_coef = fee_table['base'] + fee_table['mid_sprfall']
    else:
        if (23 <= clock) or (clock < 9):
            fee_coef = fee_table['base'] + fee_table['low_winter']
        elif ((10 <= clock) and (clock < 12)) or ((17 <= clock) and (clock < 20)) or ((22 <= clock) and (clock < 23)):
            fee_coef = fee_table['base'] + fee_table['high_winter']
        else:
            fee_coef = fee_table['base'] + fee_table['mid_winter']

    
    insertion = {
        "hour":hour
        "agent_east_temp":float(agent[(hour % 8760) - 1][1]),
        "agent_west_temp":float(agent[(hour % 8760) - 1][2]),
        "agent_hvac_power":float(agent[(hour % 8760) - 1][3]),
        "base_east_temp":float(base[(hour % 8760) - 1][1]),
        "base_west_temp":float(base[(hour % 8760) - 1][2]),
        "base_hvac_power":float(base[(hour % 8760) - 1][3]),
    }
    insertion["hvac_difference"] = insertion["agent_hvac_power"] - insertion["base_hvac_power"]
    agent_fee = fee_coef * insertion["agent_hvac_power"] * 3600 / 1000
    base_fee = fee_coef * insertion["base_hvac_power"] * 3600 / 1000
    agent_fee_accum += agent_fee
    base_fee_accum += base_fee

    insertion["agent_fee"] = agent_fee_accum
    insertion["base_fee"] = base_fee_accum


    insertion_id = insertions.insert_one(insertion)
    time.sleep(0.01)
import sys
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import pymongo
import time



client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['data_center']
insertions = db['obs']
agent = pd.read_csv('/hdd/project/datacenter/bill/csv' + '/agent_bill_ext.csv', dtype=np.float32,
                delimiter=',', error_bad_lines=False).values
base = pd.read_csv('/hdd/project/datacenter/bill/csv' + '/base_ext.csv', dtype=np.float32,
                delimiter=',', error_bad_lines=False).values

# 일반용 전력(을), 고압A, 선택3 적용
bill_table = {
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

agent_bill_accum = 0
base_bill_accum = 0

agent_hvac_energy_acc = 0
base_hvac_energy_acc = 0

insertion_id = insertions.insert_many([
    {
        "id":i,
        "hour":0,
        "agent_east_temp":0,
        "agent_west_temp":0,
        "agent_hvac_power":0,
        "base_east_temp":0,
        "base_west_temp":0,
        "base_hvac_power":0,
        "hvac_difference":0,
        "agent_bill":0,
        "base_bill":0,
        "bill_difference":0,
        "agent_hvac_energy":0,
        "base_hvac_energy":0
    } for i in range(10000)])

for hour in range(87600000):

    month = ((hour // 730) % 12) + 1
    clock = hour % 24

    if (6 <= month) and (month <= 8):
        if (23 <= clock) or (clock < 9):
            bill_coef = bill_table['low_summer']
        elif ((10 <= clock) and (clock < 12)) or ((13 <= clock) and (clock < 17)):
            bill_coef = bill_table['high_summer']
        else:
            bill_coef = bill_table['mid_summer']
    elif ((3 <= month) and (month <= 5)) or ((9 <= month) and (month <= 10)):
        if (23 <= clock) or (clock < 9):
            bill_coef = bill_table['low_sprfall']
        elif ((10 <= clock) and (clock < 12)) or ((13 <= clock) and (clock < 17)):
            bill_coef = bill_table['high_sprfall']
        else:
            bill_coef = bill_table['mid_sprfall']
    else:
        if (23 <= clock) or (clock < 9):
            bill_coef = bill_table['low_winter']
        elif ((10 <= clock) and (clock < 12)) or ((17 <= clock) and (clock < 20)) or ((22 <= clock) and (clock < 23)):
            bill_coef = bill_table['high_winter']
        else:
            bill_coef = bill_table['mid_winter']

    # 연간 3천만kWh 데이터센터
    # 계약전력 1kW는 하루 15시간, 한달 30일 기준 = 한달 450kWh
    # 가장 높은 달 일평균 2000W로 계산, 2000 * 3600 / 1000 = 7200kWh/시간
    # 하루 7200 * 24 = 172800, 한달 172800 * 30 = 5184000kWh
    # 계약전력으로 계산시 11520kW에 해당
    # 계약전력 11520kW로 계산 : 한달 + 11520 * bill_table['base'] 추가
    # 다달이 추가하지 않고 시간으로 쪼개 포함
    agent_east_temp = float(agent[(hour % 8760) - 1][1])
    agent_west_temp = float(agent[(hour % 8760) - 1][2])
    agent_hvac_power = float(agent[(hour % 8760) - 1][3]) / 20
    base_east_temp = float(base[(hour % 8760) - 1][1])
    base_west_temp = float(base[(hour % 8760) - 1][2])
    base_hvac_power = float(base[(hour % 8760) - 1][3]) / 20

    insertion = {
        "hour":hour,
        "agent_east_temp":agent_east_temp,
        "agent_west_temp":agent_west_temp,
        "agent_hvac_power":agent_hvac_power,
        "base_east_temp":base_east_temp,
        "base_west_temp":base_west_temp,
        "base_hvac_power":base_hvac_power,
    }
    insertion["hvac_difference"] = base_hvac_power - agent_hvac_power
    agent_bill = bill_coef * (agent_hvac_power * 3600 / 1000) + bill_table['base'] * 11520 / (30 * 24)
    base_bill = bill_coef * (base_hvac_power * 3600 / 1000) + bill_table['base'] * 11520 / (30 * 24)
    agent_bill_accum += agent_bill
    base_bill_accum += base_bill

    insertion["agent_bill"] = agent_bill_accum / 100000000
    insertion["base_bill"] = base_bill_accum / 100000000
    insertion["bill_difference"] = (base_bill_accum - agent_bill_accum) / 100000000

    agent_hvac_energy = agent_hvac_power * 3600 / 1000
    base_hvac_energy = base_hvac_power * 3600 / 1000
    insertion["agent_hvac_energy"] = agent_hvac_energy
    insertion["base_hvac_energy"] = base_hvac_energy
    insertion["hvac_energy_difference"] = base_hvac_energy - agent_hvac_energy

    agent_hvac_energy_acc += agent_hvac_energy
    base_hvac_energy_acc += base_hvac_energy
    insertion["agent_hvac_energy_acc"] = agent_hvac_energy_acc
    insertion["base_hvac_energy_acc"] = base_hvac_energy_acc
    insertion["hvac_energy_difference_acc"] = base_hvac_energy_acc - agent_hvac_energy_acc

    insertion_id = insertions.insert_one(insertion)
    time.sleep(0.01)
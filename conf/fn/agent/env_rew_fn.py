import numpy as np
import conf

def ep(rew_state):

    temp_min = 18
    temp_max = 27
    temp_weight = 5
    total_hvac_weight = 1e-4
    # bill_weight = 1e-6

    # month = rew_state[:,0]
    # day = rew_state[:,1]
    # hour = rew_state[:,2]
    # east_temp = rew_state[:,3]
    # west_temp = rew_state[:,4]
    # total_hvac = rew_state[:,5]
    # action = rew_state[:,6:10]
    east_temp = rew_state[:,0]
    west_temp = rew_state[:,1]
    total_hvac = rew_state[:,2]
    action = rew_state[:,3:7]

    if east_temp < temp_min:
        rew_east_temp = (-east_temp + temp_min) * (-temp_weight)
    elif east_temp > temp_max:
        rew_east_temp = (east_temp - temp_max) * (-temp_weight)
    else:
        rew_east_temp = np.zeros_like(east_temp, dtype=np.float32)
    if west_temp < temp_min:
        rew_west_temp = (-west_temp + temp_min) * (-temp_weight)
    elif west_temp > temp_max:
        rew_west_temp = (west_temp - temp_max) * (-temp_weight)
    else:
        rew_west_temp = np.zeros_like(west_temp, dtype=np.float32)

    # # 일반용 전력(을), 고압A, 선택3 적용 요금표
    # bill_table = {
    #     'base': 9810,
    #     'low_summer': 55.2,
    #     'low_sprfall': 55.2,
    #     'low_winter': 62.5,
    #     'mid_summer': 108.4,
    #     'mid_sprfall': 77.3,
    #     'mid_winter': 108.6,
    #     'high_summer': 178.7,
    #     'high_sprfall': 101.0,
    #     'high_winter': 155.5
    # }
    # # 요금 계수 산정
    # if (6 <= month) and (month <= 8):
    #     if (23 <= hour) or (hour < 9):
    #         bill_coef = bill_table['low_summer']
    #     elif ((10 <= hour) and (hour < 12)) or ((13 <= hour) and (hour < 17)):
    #         bill_coef = bill_table['high_summer']
    #     else:
    #         bill_coef = bill_table['mid_summer']
    # elif ((3 <= month) and (month <= 5)) or ((9 <= month) and (month <= 10)):
    #     if (23 <= hour) or (hour < 9):
    #         bill_coef = bill_table['low_sprfall']
    #     elif ((10 <= hour) and (hour < 12)) or ((13 <= hour) and (hour < 17)):
    #         bill_coef = bill_table['high_sprfall']
    #     else:
    #         bill_coef = bill_table['mid_sprfall']
    # else:
    #     if (23 <= hour) or (hour < 9):
    #         bill_coef = bill_table['low_winter']
    #     elif ((10 <= hour) and (hour < 12)) or ((17 <= hour) and (hour < 20)) or ((22 <= hour) and (hour < 23)):
    #         bill_coef = bill_table['high_winter']
    #     else:
    #         bill_coef = bill_table['mid_winter']

    # bill = (total_hvac * 3600 / 1000) * bill_coef
    
    # rew_bill = - bill * bill_weight
    # rew_temp = rew_east_temp + rew_west_temp
    # rew = rew_temp + rew_bill


    rew_hvac = - total_hvac * total_hvac_weight
    rew_temp = rew_east_temp + rew_west_temp
    rew = rew_temp + rew_hvac

    return rew



import os

ibm_ep = [
    105, 115, 125, 135, 145, 155, 165, 175, 185, 195, 
    205, 215, 225, 235, 245, 255, 265, 275, 285, 295, 
    305, 315, 325, 335, 345, 355, 365, 375, 385, 395,
    405, 415, 425, 435, 445, 455, 465, 475, 485, 495]
ep_from_0 = '/hdd/project/datacenter/eplog/openai-2018-11-16-08-18-21-488018/output/episode-00000'
ep_from_2 = '/eplusout.csv.gz'
ep_to_0 = '/home/iglee/project/envfn_agent/data/data_origin/eplusout_'
ep_to_2 = '.csv.gz'

for i in ibm_ep:
    os.system('cp '+ep_from_0+str(i)+ep_from_2+' '+ep_to_0+str(i)+ep_to_2)
# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Reinforcement Learning Testbed for Power Consumption Optimization
# This project is licensed under the MIT License, see LICENSE

import sys, os, subprocess, time, signal, stat
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from glob import glob
import gzip
import shutil
import numpy as np
from scipy.special import expit
import pandas as pd
from argparse import ArgumentParser
import tensorflow as tf
import agent_ddpg
import numpy as np
import conf
from tools.pipe_io import PipeIo


class EnergyPlusEnv():
    metadata = {'render.modes': ['human']}
    
    def __init__(self,
                 energyplus_file=None,
                 model_file=None,
                 weather_file=None,
                 log_dir=None,
                 verbose=False):
        self.energyplus_process = None
        self.pipe_io = None

        p_dic = getattr(conf.dic.path_dic, 'ep')
        
        # Verify path arguments
        if energyplus_file is None:
            energyplus_file = os.getenv('ENERGYPLUS')
        if energyplus_file is None:
            print('energyplus_env: FATAL: EnergyPlus executable is not specified. Use environment variable ENERGYPLUS.')
            return None
        if model_file is None:
            model_file = os.getenv('ENERGYPLUS_MODEL')
        if model_file is None:
            print('energyplus_env: FATAL: EnergyPlus model file is not specified. Use environment variable ENERGYPLUS_MODEL.')
            return None
        if weather_file is None:
            weather_file = os.getenv('ENERGYPLUS_WEATHER')
        if weather_file is None:
            print('energyplus_env: FATAL: EnergyPlus weather file is not specified. Use environment variable ENERGYPLUS_WEATHER.')
            return None
        if log_dir is None:
            log_dir = os.getenv('ENERGYPLUS_LOG')
        if log_dir is None:
            log_dir = 'log'
        
        # Initialize paths
        self.energyplus_file = energyplus_file
        self.model_file = model_file
        self.log_dir = log_dir
        self.weather_file = weather_file

        self.pipe_io = PipeIo()

        self.verbose = verbose


    def start_instance(self):
        print('Starting new environment')
        assert(self.energyplus_process is None)

        output_dir = self.log_dir + '/output/episode'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.pipe_io.start()

        # Make copies of model file and weather file into output dir, and use it for execution
        # This allow update of these files without affecting active simulation instances
        shutil.copy(self.model_file, output_dir)
        shutil.copy(self.weather_file, output_dir)
        copy_model_file = output_dir + '/' + os.path.basename(self.model_file)
        copy_weather_file = output_dir + '/' + os.path.basename(self.weather_file)

        # Spawn a process
        cmd = self.energyplus_file \
              + ' -r -x' \
              + ' -d ' + output_dir \
              + ' -w ' + copy_weather_file \
              + ' ' + copy_model_file
        print('Starting EnergyPlus with command: %s' % cmd)
        self.energyplus_process = subprocess.Popen(cmd.split(' '), shell=False)

    def stop_instance(self):
        if self.energyplus_process is not None:
            self.energyplus_process.terminate()
            self.energyplus_process = None
        if self.pipe_io is not None:
            self.pipe_io.stop()
        
        def count_severe_errors(file):
            if not os.path.isfile(file):
                return -1 # Error count is unknown
            # Sample: '   ************* EnergyPlus Completed Successfully-- 6214 Warning; 2 Severe Errors; Elapsed Time=00hr 00min  7.19sec'
            fd = open(file)
            lines = fd.readlines()
            fd.close()
            for line in lines:
                if line.find('************* EnergyPlus Completed Successfully') >= 0:
                    tokens = line.split()
                    return int(tokens[6])
            return -1
        epsode_dir = self.log_dir + '/output/episode-'
        file_csv = epsode_dir + '/eplusout.csv'
        file_csv_gz = epsode_dir + '/eplusout.csv.gz'
        file_err = epsode_dir + '/eplusout.err'
        files_to_preserve = ['eplusout.csv', 'eplusout.err', 'eplustbl.htm']
        files_to_clean = ['eplusmtr.csv', 'eplusout.audit', 'eplusout.bnd',
                            'eplusout.dxf', 'eplusout.eio', 'eplusout.edd',
                            'eplusout.end', 'eplusout.eso', 'eplusout.mdd',
                            'eplusout.mtd', 'eplusout.mtr', 'eplusout.rdd',
                            'eplusout.rvaudit', 'eplusout.shd', 'eplusssz.csv',
                            'epluszsz.csv', 'sqlite.err']

        # Check for any severe error
        nerr = count_severe_errors(file_err)
        if nerr != 0:
            print('EnergyPlusEnv: Severe error(s) occurred. Error count: {}'.format(nerr))
            print('EnergyPlusEnv: Check contents of {}'.format(file_err))
            #sys.exit(1)

        # Compress csv file and remove unnecessary files
        # If csv file is not present in some reason, preserve all other files for inspection
        if os.path.isfile(file_csv):
            with open(file_csv, 'rb') as f_in:
                with gzip.open(file_csv_gz, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(file_csv)

            if not os.path.exists("/tmp/verbose"):
                for file in files_to_clean:
                    file_path = epsode_dir + '/' + file
                    if os.path.isfile(file_path):
                        os.remove(file_path)

    def step(self, action):

        # Send action to the environment
        if action is not None:

            if not self.send_action(action):
                print('EnergyPlusEnv.step(): Failed to send an action. Quitting.')
        
        state, done = self.receive_observation()
        
        return state, done
    
    def send_action(self, action):
        num_data = len(action)
        if self.pipe_io.writeline('{0:d}'.format(num_data)):
            return False
        for i in range(num_data):
            self.pipe_io.writeline('{0:f}'.format(action[i]))
        self.pipe_io.flush()
        return True

    def receive_observation(self):
        line = self.pipe_io.readline()
        if (line == ''):
            return None, True
        num_data = int(line)
        raw_state = np.zeros(num_data)
        for i in range(num_data):
            line = self.pipe_io.readline()
            if (line == ''):
                return None, True
            val = float(line)
            raw_state[i] = val
        return raw_state, False
    
    def render(self, mode='human'):
        if mode == 'human':
            return False
        
    def close(self):
        self.stop_instance()

    def reset(self):
        self.stop_instance()
        self.start_instance()
        return self.step(None)[0]
#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 12:07:10 2023

@author: MaxGr
"""


import os

import time
import copy
import math
import shutil
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


import torch
# import torchvision
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
# import torch.backends.cudnn as cudnn
# import torchvision.transforms as transformtransforms

# from tqdm import tqdm
# from torchvision import models
# from torchsummary import summary
# from torch.autograd import Variable
# from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import ToPILImage

from collections import namedtuple    
import pytoml
from sqlalchemy import create_engine, text

import yaml
import pickle


###############################################################################

def temp_c_to_f(temp_c):#: float, arbitrary_arg=None):
    """Convert temp from C to F. Test function with arbitrary argument, for example."""
    return 1.8 * temp_c + 32

def temp_f_to_c(temp_f):
    return (temp_f-32)/1.8



'''
5 Layer DNN
'''
class DNN_5(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DNN_5, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(state_dim, 128))
        # self.layer2 = nn.Sequential(nn.Linear(128, 512))
        # self.layer3 = nn.Sequential(nn.Linear(256, 512))
        # self.layer4 = nn.Sequential(nn.Linear(512, 256))
        self.layer5 = nn.Sequential(nn.Linear(128, 128))
        self.layer6 = nn.Sequential(nn.Linear(128, action_dim))
        
                
    def forward(self, x):
        x = F.relu(self.layer1(x))
        # x = F.relu(self.layer2(x))
        # x = F.relu(self.layer3(x))
        # x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = self.layer6(x)
                
        return x
    
    
    
# device = torch.device("cuda")
# model = DNN_5(5,2).to(device)
# summary(model, (4320,5))


###############################################################################



import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # FIFO

    def add(self, state, action, reward, next_state, done):  # add data to buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # sample from buffer with batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)



class DQN:
    def __init__(self, state_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        # Q Network
        self.q_net = DNN_5(state_dim, action_dim).to(device)  
        # Target Network
        self.target_q_net = DNN_5(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma  # discout
        self.epsilon = epsilon  # epsilon-greedy
        self.target_update = target_update  # target update period
        self.count = 0  # record updates
        self.device = device

    def take_action(self, state):  # epsilon-greedy
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action
    
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(self.device)
        
        q_values = self.q_net(states).gather(1, actions)  # Q value
        
        # maxQ
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD error
        
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # MSE
        self.optimizer.zero_grad()  # clear grad
        dqn_loss.backward()  # back-propagation & update network
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict( 
                self.q_net.state_dict())  # update target network
        self.count += 1
        return dqn_loss

    
def HVAC_action(action, temp):
    
        if action == 0:
            H_new = config.F_bottom
            C_new = config.F_top
            
        elif action == 1:
            H_new = temp[0]
            C_new = temp[1]
            
        return int(H_new), int(C_new)
    

    
def generate_action_list():
        
    HVAC_action_list = []
    for HC_1 in [0,1]:
        for HC_2 in [0,1]:
            for HC_3 in [0,1]:
                for HC_4 in [0,1]:
                    for HC_5 in [0,1]:
                        for HC_6 in [0,1]:
                            HVAC_action_list.append([HC_1,HC_2,HC_3,HC_4,HC_5,HC_6])
        
    return HVAC_action_list


###############################################################################



            
def read_parameters_from_txt(file_path):
    parameters = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                key, value = line.split(':')
                parameters[key.strip()] = value.strip()
                print(f"{key.strip()}: {value.strip()}")
    return parameters



def delete_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' deleted successfully.")
    except FileNotFoundError:
        print(f"Folder '{folder_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred while deleting folder '{folder_path}': {e}")



        
###############################################################################
'''
main function
'''


if __name__ == '__main__':
    time_start = time.time()


    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    TORCH_CUDA_ARCH_LIST="8.6"
    
    # import shutil
    folder_path = "./out"
    delete_folder(folder_path)
    
    
    

    print('torch.version: ',torch. __version__)
    print('torch.version.cuda: ',torch.version.cuda)
    print('torch.cuda.is_available: ',torch.cuda.is_available())
    print('torch.cuda.device_count: ',torch.cuda.device_count())
    print('torch.cuda.current_device: ',torch.cuda.current_device())
    device_default = torch.cuda.current_device()
    torch.cuda.device(device_default)
    print('torch.cuda.get_device_name: ',torch.cuda.get_device_name(device_default))
    device = torch.device("cuda")
        
        
    
    
    # Specify the file name you want to read parameters from
    input_file = 'config.yaml'
    
    # Read parameters from the YAML file
    with open(input_file, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    
    # Now 'parameters' contains the data read from the YAML file
    print(config)
    
    
    
    
    # Define the Parameters class
    class Parameters:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
                
    config = Parameters(**config)

    

    '''
    read df data into dataframe
    '''
    
    if os.path.exists('./data/History.csv'):
        # Read data from CSV file
        df = pd.read_csv('./data/History.csv')
        
        # Read data from Excel file
        # df_excel = pd.read_excel('./HVAC_data/df.xlsx')
    
    
    HVAC_action_list = generate_action_list()    
    
    
    
    
    '''
    Hyperparameters for DQN
    
    '''
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    agent = DQN(config.state_dim, 
                config.action_dim, 
                config.lr, 
                config.gamma, 
                config.epsilon, 
                config.target_update, device)
    
    if os.path.exists(config.weight_file):
        agent.q_net.load_state_dict(torch.load(config.weight_file))

    if os.path.exists(config.replay_buffer):
        with open(config.replay_buffer, 'rb') as file:
            replay_buffer = pickle.load(file)
    else:
        replay_buffer = ReplayBuffer(config.buffer_size)
    

    '''
    DQN
    '''

    
    # Benchmark = np.zeros((1,30), dtype=object)

    print(f'\n Training start: {datetime.datetime.now()} \n')
    
    new_row = len(df)
    time_interval = new_row



    
    '''
    read data from server
    '''

    with open("/home/ubuntu/mysql-credentials.toml") as f:
        c = pytoml.load(f)

    engine = create_engine("mysql+pymysql://{user}:{passw}@{host}/{schema}".format(**c))
    print(f"{engine=}")

    Status = namedtuple("Status", "ts building zone vav occupied temperature outside flowrate")
    query = """
    SELECT *
      FROM status
      WHERE building = '10.21'
        AND zone = 'L1 NORTH'
        AND (building, zone, vav, ts) IN ( SELECT building, zone, vav, MAX(ts) FROM status Group BY building, zone, vav)
    """
    results = [ ]
    with engine.connect() as con:
        result = con.execute(text(query))
        # for ts,building,zone,vav,occupied,temperature in result:
        for row in result:
            results.append(Status(*row))
            
    print(results)
    
    
    
    
    # query = """
    #     WITH latest AS (
    #   SELECT building, zone, vav, MAX(ts) ts
    #     FROM status
    #     GROUP BY building, zone, vav
    # )
    # SELECT *
    #   FROM status
    #   WHERE building = '10.21'
    #     AND zone = 'L1 NORTH'
    #     AND (building, zone, vav, ts) IN (
    #       SELECT S.building, S.zone, S.vav, MAX(S.ts)
    #         FROM status S, latest L
    #         WHERE S.building = L.building
    #           AND S.zone = L.zone
    #           AND S.vav = L.vav
    #           AND S.ts != L.ts
    #         GROUP BY S.building, S.zone, S.vav
    
    #     )
    #     """
    
    # results_tm1 = [ ]
    # with engine.connect() as con:
    #     result = con.execute(text(query))
    #     # for ts,building,zone,vav,occupied,temperature in result:
    #     for row in result:
    #         results_tm1.append(Status(*row))
<<<<<<< Updated upstream
            
    # print(results_tm1)
        
=======
            
<<<<<<< HEAD
            
            
=======
    # print(results_tm1)
        
>>>>>>> 9545d3f0cdf56bba5d6f50d9e73d6ebbee27756b
>>>>>>> Stashed changes
    # class DataPoint:
    #     def __init__(self, ts, temperature, occupied):
    #         self.ts = ts
    #         self.temperature = temperature
    #         self.occupied = occupied
    #         self.outside_temperature = 75

    # results = [
    #     DataPoint(datetime.datetime.now(), 50, 1),
    #     DataPoint(datetime.datetime.now(), 80, 0),
    #     DataPoint(datetime.datetime.now(), 90, 1),
    #     DataPoint(datetime.datetime.now(), 30, 1),
    #     DataPoint(datetime.datetime.now(), 60, 1),
    #     DataPoint(datetime.datetime.now(), 75, 0)
    # ]
                    
    
    
    
            
    

    # print(results)

    ''' Time '''
    year = results[0].ts.year
    month = results[0].ts.month
    day = results[0].ts.day
    hour = results[0].ts.hour
    minute = results[0].ts.minute

    # time_step = time_step
    
    
    
    '''Temperature'''

    # oa_temp = 75
    oa_temp = results[0].outside_temperature
    
    zone_temp_2001 = results[0].temperature
    zone_temp_2002 = results[1].temperature
    zone_temp_2003 = results[2].temperature
    zone_temp_2004 = results[3].temperature
    zone_temp_2005 = results[4].temperature
    zone_temp_2006 = results[5].temperature
    
    
    occ = [[60,90], [71,74]]
    
    hvac_2001 = results[0].occupied
    hvac_2002 = results[1].occupied
    hvac_2003 = results[2].occupied
    hvac_2004 = results[3].occupied
    hvac_2005 = results[4].occupied
    hvac_2006 = results[5].occupied




    '''
    Store data
    '''
    
    # Append new data points to the dictionary
    df.loc[new_row, 'y_outdoor'] = oa_temp
    
    df.loc[new_row, 'y_zone_temp_2001'] = zone_temp_2001
    df.loc[new_row, 'y_zone_temp_2002'] = zone_temp_2002
    df.loc[new_row, 'y_zone_temp_2003'] = zone_temp_2003
    df.loc[new_row, 'y_zone_temp_2004'] = zone_temp_2004
    df.loc[new_row, 'y_zone_temp_2005'] = zone_temp_2005
    df.loc[new_row, 'y_zone_temp_2006'] = zone_temp_2006
    
    df.loc[new_row, 'hvac_2001'] = hvac_2001
    df.loc[new_row, 'hvac_2002'] = hvac_2002
    df.loc[new_row, 'hvac_2003'] = hvac_2003
    df.loc[new_row, 'hvac_2004'] = hvac_2004
    df.loc[new_row, 'hvac_2005'] = hvac_2005
    df.loc[new_row, 'hvac_2006'] = hvac_2006
    
    
    T_list = (np.array([zone_temp_2001,
                        zone_temp_2002,
                        zone_temp_2003,
                        zone_temp_2004,
                        zone_temp_2005,
                        zone_temp_2006]))
        
    
    T_mean = np.mean(T_list)
    
    df.loc[new_row, 'T_mean'] = T_mean
    df.loc[new_row, 'T_diff'] = np.max(T_list)-np.min(T_list)
    df.loc[new_row, 'T_var']  = np.var(T_list)
    

    df.loc[new_row, 'year'] = year
    df.loc[new_row, 'month'] = month
    df.loc[new_row, 'day'] = day
    df.loc[new_row, 'hour'] = hour
    df.loc[new_row, 'minute'] = minute
    
    
    
    dt = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute)

    
    df.loc[new_row, 'time_line'] = dt
    
    if dt.weekday() > 4:
        # print 'Given date is weekend.'
        df.loc[new_row, 'weekday'] = dt.weekday()+1
        isweekday = 0
        isweekend = 1
        df.loc[new_row, 'isweekday'] = isweekday
        df.loc[new_row, 'isweekend'] = isweekend
    else:
        # print 'Given data is weekday.'
        df.loc[new_row, 'weekday'] = dt.weekday()+1
        isweekday = 1
        isweekend = 0
        df.loc[new_row, 'isweekday'] = isweekday
        df.loc[new_row, 'isweekend'] = isweekend
        

    if 6 < hour < 20:
        sun_is_up = 1
    else:
        sun_is_up = 0
        
    is_worktime = isweekday * sun_is_up
    df.loc[new_row, 'work_time'] = is_worktime

    
    # print('last data: ', df.iloc[-2])

    # print('current data: ', df.iloc[-1])
    
    
    
    ''' 
    DQN 
    
    '''
    if config.Train == False:
        df.loc[new_row, 'reward'] = 0

    if config.Train == True:
    
        if time_interval == 1:
            df.loc[1, 'reward'] = 0
            df.loc[1, 'action_list'] = 0
    
    
        '''
        Replay
        '''
        done = False
        
        # t_-1
        O0 = df.loc[new_row-1, 'y_outdoor']
        W0 = df.loc[new_row-1, 'work_time']

        T_10 = df.loc[new_row-1, 'y_zone_temp_2001']
        T_20 = df.loc[new_row-1, 'y_zone_temp_2002']
        T_30 = df.loc[new_row-1, 'y_zone_temp_2003']
        T_40 = df.loc[new_row-1, 'y_zone_temp_2004']
        T_50 = df.loc[new_row-1, 'y_zone_temp_2005']
        T_60 = df.loc[new_row-1, 'y_zone_temp_2006']
        
        H_10 = df.loc[new_row-1, 'hvac_2001']
        H_20 = df.loc[new_row-1, 'hvac_2002']
        H_30 = df.loc[new_row-1, 'hvac_2003']
        H_40 = df.loc[new_row-1, 'hvac_2004']
        H_50 = df.loc[new_row-1, 'hvac_2005']
        H_60 = df.loc[new_row-1, 'hvac_2006']
        
        state_0 = [O0/100,W0,
                   T_10/100,T_20/100,T_30/100,T_40/100,T_50/100,T_60/100,
                   H_10,H_20,H_30,H_40,H_50,H_60]
        
        print(f'State t_-1: {state_0}')

       
        action_0 = df.loc[new_row-1, 'action_list']

        
        # t_0
        O1 = oa_temp
        W1 = is_worktime

        T_11 = zone_temp_2001
        T_21 = zone_temp_2002
        T_31 = zone_temp_2003
        T_41 = zone_temp_2004
        T_51 = zone_temp_2005
        T_61 = zone_temp_2006
        
        H_11 = hvac_2001
        H_21 = hvac_2002
        H_31 = hvac_2003
        H_41 = hvac_2004
        H_51 = hvac_2005
        H_61 = hvac_2006
        
        
        # t_1
        state_1 = [O1/100,W1,
                   T_11/100,T_21/100,T_31/100,T_41/100,T_51/100,T_61/100,
                   H_11,H_21,H_31,H_41,H_51,H_61] 
        
        print(f'State t_0: {state_1}')

        action_1 = agent.take_action(state_1)
        action_map = HVAC_action_list[action_1]


        [hvac_2001_new, 
         hvac_2002_new, 
         hvac_2003_new,
         hvac_2004_new, 
         hvac_2005_new, 
         hvac_2006_new] = action_map
        
            
        
        
        
        
        # df.loc[new_row, 'hvac_2001'] = hvac_2001_new
        # df.loc[new_row, 'hvac_2002'] = hvac_2002_new
        # df.loc[new_row, 'hvac_2003'] = hvac_2003_new
        # df.loc[new_row, 'hvac_2004'] = hvac_2004_new
        # df.loc[new_row, 'hvac_2005'] = hvac_2005_new
        # df.loc[new_row, 'hvac_2006'] = hvac_2006_new
        

        df.loc[new_row, 'action_list'] = action_1
        
        
        print("Action for next step: ", action_map)
        
        
        
        '''
        calculating VAV energy
        '''
        airflow_rate = 0.5  # cubic meters per second (m³/s)
        specific_heat_capacity_air = 1005  # J/kg°C (at constant pressure)

        E_HVAC = []
        
        T_list = [T_11,T_21,T_31,T_41,T_51,T_61]
        
        for i in range(len(T_list)):
            # temperature_difference = T_i - O0  # °C
            
            '''
            building has no cooling coil
            '''
            # if T_i > O0:
            #     temperature_difference = T_i - O0  # °C
            #     cooling_energy = airflow_rate * specific_heat_capacity_air * temperature_difference 
            #     E_HVAC.append(cooling_energy)
                
            if T_list[i] < 71:
                temperature_difference = 71 - T_list[i]  # °C
                heating_energy = airflow_rate * specific_heat_capacity_air * temperature_difference * action_map[i]
                E_HVAC.append(heating_energy*3)
                
                
        E1 = np.sum(E_HVAC)

        

        ''' 
        reward define  
        
        '''
        if is_worktime:
            E_factor = config.E_factor_day
            T_factor = config.T_factor_day
            positive_reward = config.positive_reward
            
        else:
            E_factor = config.E_factor_night
            T_factor = config.T_factor_night
            positive_reward = 0

                
        # 1 Energy
        reward_E = -E1 * E_factor
        
        
        # 2 Temp
        reward_T = []
        
        for T_i in [T_11,T_21,T_31,T_41,T_51,T_61]:
            if 68<T_i<77:
                reward_T.append(1*positive_reward)
            else:
                reward_T.append( -(T_i-72)**2 * T_factor )
            
        reward_T = np.mean(reward_T)
        

        # 4 Smootheness
        current_action = HVAC_action_list[int(df.loc[new_row, 'action_list'])]
        last_action    = HVAC_action_list[int(df.loc[new_row-1, 'action_list'])]
        
        change_action = np.array(current_action) ^ np.array(last_action)
        num_unstable = len(change_action[change_action==1])
        reward_signal = -config.signal_factor * num_unstable
        
        
        if config.signal_loss == True:
            reward_1 = reward_T + reward_E + reward_signal
        else:
            reward_1 = reward_T + reward_E
    
        df.loc[new_row, 'reward'] = reward_1
        
        
        
        T_violation = []
        if is_worktime:
            for T_i in [T_11,T_21,T_31,T_41,T_51,T_61]:
                if T_i > 77:
                    T_violation.append(T_i-77)
                elif T_i < 68:
                    T_violation.append(68-T_i)
            
        T_violation = np.sum(T_violation)
        df.loc[new_row, 'T_violation'] = T_violation
        
        
        
        print('\n')
        print('ts / dt                  T_mean / 72     T   /    E   /    S')
        print('%d / %s   %.2f / 72   %.3f / %.3f / %.2f'%(time_interval,
                                                                  dt,
                                                                  T_mean,
                                                                  reward_T,
                                                                  reward_E,
                                                                  reward_signal))

    
        

        if done == True:
            print('Adjust HVAC setting...')
            
    
        '''
        replay buffer
        
        '''
        # add to experience
        replay_buffer.add(state_0, action_0, reward_1, state_1, done)
        
        with open(config.replay_buffer, 'wb') as file:
            pickle.dump(replay_buffer, file)
        

        '''
        training
        
        '''
        if replay_buffer.size() > config.minimal_size:
            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(config.batch_size)
            transition_dict = {
                                'states': b_s,
                                'actions': b_a,
                                'next_states': b_ns,
                                'rewards': b_r,
                                'dones': b_d
                                }
            agent.update(transition_dict)








    if config.HVAC_output == True:
        # Save DataFrame to a CSV file
        df.to_csv('./data/History.csv', index=False)
        df.to_csv('./data/History_backup.csv', index=False)
        
        # Save DataFrame to an Excel file
        # df.to_excel('./data/History.xlsx', index=False)
        # df.to_excel('./data/History_backup.xlsx', index=False)


        




        




    torch.save(agent.target_q_net.state_dict(), config.weight_file)

    time_end = time.time()
    time_round = time_end-time_start

    print(f'\n Training finished, time cost: {time_round} \n')
        
    


    






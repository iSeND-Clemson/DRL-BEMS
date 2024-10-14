# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 02:09:15 2023

@author: MaxGr
"""


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"



import time
import copy
import math
import shutil
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.pyplot import MultipleLocator

import matplotlib.dates as mdates

break_point




np.mean(df[:,9])



'''
Energy Saving
'''

df = np.load('Benchmark/DQN.npy', allow_pickle=True)

x = df[0,11]#[288:]
work_time = df[0,27]#[288:]
work_time_length = work_time.count(1)

E_HVAC_all_RBC = df[0,1]#[288:]
E_HVAC_all_DQN = df[10,2]#[288:]


plt.figure(figsize=(20,5), dpi=100)
plt.plot(E_HVAC_all_RBC, label='HVAC_RBC')
plt.plot(E_HVAC_all_DQN, label='HVAC_DQN')
plt.legend()


print(np.sum(E_HVAC_all_RBC))
print(np.sum(E_HVAC_all_DQN))

212,233 MJ
132,703 MJ


E_RBC_day = []
E_DQN_day = []
x_day = []

dt = 288

for i in range(0,len(x),dt):
    print(x[i])
    E_RBC_day.append(np.mean(E_HVAC_all_RBC[i:i+dt]))
    E_DQN_day.append(np.mean(E_HVAC_all_DQN[i:i+dt]))
    x_day.append(x[i])
    
plt.figure(figsize=(20,5), dpi=100)
plt.plot(x_day, E_RBC_day, label='HVAC_RBC')
plt.plot(x_day, E_DQN_day, label='HVAC_DQN')
plt.legend()


E_day = [x_day, E_RBC_day, E_DQN_day]

# Convert NumPy array to DataFrame
df = pd.DataFrame(E_day)

# Save DataFrame to Excel
file_path = 'Benchmark/DQN_RBC_E.xlsx'
df.to_excel(file_path, index=False)







# x = df[0,11][288:]
E = df[:,3]
T = df[:,4]


plt.figure(figsize=(20,5), dpi=100)
plt.plot(E, label='HVAC_RBC')
plt.plot(T, label='HVAC_DQN')
plt.legend()



E_T = [E,T]

# Convert NumPy array to DataFrame
df_xlsx = pd.DataFrame(E_T)

# Save DataFrame to Excel
file_path = 'Benchmark/DQN_E_T.xlsx'
df_xlsx.to_excel(file_path, index=False)







'''Day Average'''
E_RBC_day = E_HVAC_all_RBC[0:288]
E_DQN_day = E_HVAC_all_DQN[0:288]
x_day = x[0:288]

for i in range(0,len(x),288):
    print(x[i])
    E_RBC_1 = E_HVAC_all_RBC[i:i+288]
    E_DQN_2 = E_HVAC_all_DQN[i:i+288]

    E_RBC_day = (np.array(E_RBC_day)+np.array(E_RBC_1))/2
    E_DQN_day = (np.array(E_DQN_day)+np.array(E_DQN_2))/2
    


plt.plot(E_RBC_day)



plt.figure(figsize=(20,5), dpi=100)
plt.plot(x_day, E_RBC_day, label='HVAC_RBC')
plt.plot(x_day, E_DQN_day, label='HVAC_DQN')
plt.legend()


















HVAC_action_list = []
for HC_1 in [0,1]:
    for HC_2 in [0,1]:
        for HC_3 in [0,1]:
            for HC_4 in [0,1]:
                for HC_5 in [0,1]:
                    for HC_6 in [0,1]:
                        HVAC_action_list.append([HC_1,HC_2,HC_3,HC_4,HC_5,HC_6])
    



'''
Signal Smooth
'''
df1 = np.load('Benchmark/Signal/0.1.npy', allow_pickle=True)[-1]
df2 = np.load('Benchmark/Signal/0.5.npy', allow_pickle=True)[-1]
df3 = np.load('Benchmark/Signal/1.0.npy', allow_pickle=True)[6]
df4 = np.load('Benchmark/Signal/5.0.npy', allow_pickle=True)[8]
df5 = np.load('Benchmark/Signal/10.0.npy', allow_pickle=True)[-1]
df6 = np.load('Benchmark/Signal/20.0.npy', allow_pickle=True)[4]

df7 = np.load('Benchmark/Signal/10.0.npy', allow_pickle=True)[5]
df8 = np.load('Benchmark/Signal/10.0.npy', allow_pickle=True)[2]
df9 = np.load('Benchmark/Signal/10.0.npy', allow_pickle=True)[3]
df10 = np.load('Benchmark/Signal/10.0.npy', allow_pickle=True)[6]


signal_change = []



for df in [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10]:
    
    HVAC_action = df[8][1:]
    signal_total = [0, 0, 0, 0, 0, 0]
    
    E = df[3]
    T = df[4]
    
    for i in range(len(HVAC_action)):
        if work_time[i] == 1:
            signal = HVAC_action_list[HVAC_action[i]]
            signal_total = np.array(signal_total) + np.array(signal)
            
    signal_change.append(signal_total)
    print(signal_total)
    print(E,T)



for df in [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10]:
    
    HVAC_action = df[8]
    
    signal_change = []

    for i in range(1, len(HVAC_action)):
        current_action = HVAC_action_list[HVAC_action[i-1]]
        last_action    = HVAC_action_list[HVAC_action[i]]
        
        change_action = np.array(current_action) ^ np.array(last_action)
        num_unstable = len(change_action[change_action==1])            

        signal_change.append(num_unstable)
        
    signal_total = np.sum(signal_change)
    print(signal_total)
    # print(E,T)












plt.figure(figsize=(20,5), dpi=100)
plt.plot(E_HVAC_all_RBC, label='HVAC_RBC')
plt.plot(E_HVAC_all_DQN, label='HVAC_DQN')
plt.legend()













df1 = np.load('Benchmark/E_T/1-1.npy', allow_pickle=True)
df2 = np.load('Benchmark/E_T/1-2.npy', allow_pickle=True)
df3 = np.load('Benchmark/E_T/1-5.npy', allow_pickle=True)
df4 = np.load('Benchmark/E_T/1-10.npy', allow_pickle=True)
df5 = np.load('Benchmark/E_T/2-1.npy', allow_pickle=True)
df6 = np.load('Benchmark/E_T/2-2.npy', allow_pickle=True)
df7 = np.load('Benchmark/E_T/5-1.npy', allow_pickle=True)
df8 = np.load('Benchmark/E_T/5-5.npy', allow_pickle=True)
df9 = np.load('Benchmark/E_T/10-1.npy', allow_pickle=True)
df10 = np.load('Benchmark/E_T/10-10.npy', allow_pickle=True)


E = []
T = []

for df in [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10]:
    
    
    E_max = np.mean(df[:,3])
    T_max = np.mean(df[:,4])
    
    E_total = 212233225957*(1 - E_max)
    
    T_offset = np.mean(df[:,3])
    T_sigma  = np.mean(df[:,3])
    
    print(E_max, T_max, E_total)
    
    E_i = df[1:,3]
    T_i = df[1:,4]
    
    print(E_i, T_i)
    
    E.append(E_i)
    T.append(T_i)
    


E = np.array(E)
T = np.array(T)


E_T = [E,T]

# Convert NumPy array to DataFrame
E_xlsx = pd.DataFrame(E)
T_xlsx = pd.DataFrame(T)


# Save DataFrame to Excel
E_xlsx.to_excel('Benchmark/DQN_E_T_Trade_E.xlsx', index=False)
T_xlsx.to_excel('Benchmark/DQN_E_T_Trade_T.xlsx', index=False)






df1 = np.load('Benchmark/DQN.npy', allow_pickle=True)
df2 = np.load('Benchmark/reward_1.npy', allow_pickle=True)

x = df1[0,27][287:]
H_reward = []
D_reward = []


for i in range(1):
    h_reward = df1[i,7][288:]
    d_reward = df2[i,7][288:]
    
    h_reward = np.array(h_reward)*x
    d_reward = np.array(d_reward)*x
    
    
    # H_reward += h_reward
    # D_reward += d_reward
    
h_reward = np.delete(h_reward, np.where(h_reward == 0))+6
d_reward = np.delete(d_reward, np.where(d_reward == 0))
    

plt.figure(figsize=(20,5), dpi=100)
plt.plot(h_reward, label='D_reward')
plt.plot(d_reward, label='H_reward')
plt.legend()





h_reward_day = []
d_reward_day = []

interval = 200

for i in range(0,len(h_reward),interval):
    # print(x[i])
    h_reward_day_i = np.sum(h_reward[i:i+interval])
    d_reward_day_i = np.sum(d_reward[i:i+interval])

    h_reward_day.append(h_reward_day_i)
    d_reward_day.append(d_reward_day_i)




plt.figure(figsize=(20,5), dpi=100)
plt.plot(h_reward_day, label='D_reward')
plt.plot(d_reward_day, label='H_reward')
plt.legend()














df1 = np.load('Benchmark/h_reward.npy', allow_pickle=True)
df2 = np.load('Benchmark/d_reward.npy', allow_pickle=True)

h_reward = []
d_reward = []

for i in range(len(df1)):
    h_reward_i = np.sum(df1[i,7][288:])
    d_reward_i = np.sum(df2[i,7][288:])
    
    h_reward.append(h_reward_i)
    d_reward.append(d_reward_i)

    print(h_reward_i, d_reward_i)



h_reward = h_reward-np.max(h_reward)
d_reward = d_reward-np.max(d_reward)

plt.figure(figsize=(20,5), dpi=100)
plt.plot(h_reward, label='H_reward')
plt.plot(d_reward, label='D_reward')
plt.legend()


D_H = [h_reward, d_reward]

# Convert NumPy array to DataFrame
df_xlsx = pd.DataFrame(D_H)

# Save DataFrame to Excel
file_path = 'Benchmark/DQN_D_H.xlsx'
df_xlsx.to_excel(file_path, index=False)





    
    
index = []
h_reward_50 = []
d_reward_50 = []
    
i_count = 0
for i in range(0,len(h_reward), 2):
    print(i_count)
    
    index.append(i_count)
    h_reward_50.append((h_reward[i]+h_reward[i+1])/2)
    d_reward_50.append((d_reward[i]+d_reward[i+1])/2)

    print(h_reward_50[i_count], d_reward_50[i_count])
    i_count += 1




plt.figure(figsize=(20,5), dpi=100)
plt.plot(h_reward_50, label='H_reward')
plt.plot(d_reward_50, label='D_reward')
plt.legend()




D_H = [d_reward_50, h_reward_50]

# Convert NumPy array to DataFrame
df_xlsx = pd.DataFrame(D_H)

# Save DataFrame to Excel
file_path = 'Benchmark/DQN_D_H.xlsx'
df_xlsx.to_excel(file_path, index=False)

















df_MARL = np.load('./data/Benchmark_MARL_0-5.npy', allow_pickle=True)

signal_list = []

for k in range(len(df_MARL)):
    HVAC_action = df_MARL[k, 8][2:]
    
    signal_change = []
    
    for i in range(1, len(HVAC_action)):
        current_action = HVAC_action[i-1]
        last_action    = HVAC_action[i]
        
        change_action = np.array(current_action) ^ np.array(last_action)
        num_unstable = len(change_action[change_action==1])            
    
        signal_change.append(num_unstable)
        
    signal_total = np.sum(signal_change)
    print(signal_total)
    # print(E,T)
    
    signal_list.append(signal_total)
    
print(np.mean(signal_list))






df_DQN_new = np.load('./data/Benchmark_MARL_0-5.npy', allow_pickle=True)







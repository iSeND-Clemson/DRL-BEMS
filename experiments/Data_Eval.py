# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 08:48:11 2023

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


time_end = time.time()
time_round = time_end-time_start
print(time_round)
    
    

# plt.figure(figsize=(30,10), dpi=100)
# plt.plot(E_HVAC_all_RBC, label='Default')
# plt.plot(E_HVAC_all_DQN, label='DQN')
# plt.legend()


x_sum_1 = np.sum(E_HVAC_all_RBC)
x_sum_2 = np.sum(E_HVAC_all_DQN)

print(f'{x_sum_2}/{x_sum_1}')

E_save = (x_sum_1-x_sum_2)/x_sum_1
T_violation = len(EPLUS.T_Violation)/ len(E_HVAC_all_DQN)
T_violation_offset = np.mean(EPLUS.T_Violation)

print(E_save)
print(T_violation)
print(T_violation_offset)



Benchmark[epoch, 0] = epoch
Benchmark[epoch, 1] = E_HVAC_all_RBC
Benchmark[epoch, 2] = E_HVAC_all_DQN
Benchmark[epoch, 3] = E_save
Benchmark[epoch, 4] = T_violation
Benchmark[epoch, 5] = T_violation_offset
Benchmark[epoch, 6] = EPLUS.T_Violation
Benchmark[epoch, 7] = EPLUS.episode_reward
Benchmark[epoch, 8] = EPLUS.action_list
Benchmark[epoch, 9] = time_round
Benchmark[epoch, 10] =  EPLUS.score

np.save('Benchmark.npy', Benchmark, allow_pickle=True)




# plt.figure(figsize=(30,10), dpi=100)
# plt.plot(EPLUS.E_Heating, label='E_Heating')
# plt.plot(EPLUS.E_Cooling, label='E_Cooling')
# plt.plot(Benchmark[:, 3], label='E_save')
# plt.plot(Benchmark[:, 4], label='T_violation')
# plt.legend()


# E_sum_1 = np.sum(EPLUS.E_Heating)
# E_sum_2 = np.sum(EPLUS.E_Cooling)

# print((E_sum_1)/(E_sum_1+E_sum_2))
# print((E_sum_2)/(E_sum_1+E_sum_2))



import numpy as np

df1 = np.load('Benchmark_DQN_10_1zone.npy', allow_pickle=True)
df2 = np.load('Benchmark_DQN_10_20_1zone.npy', allow_pickle=True)

df3 = np.load('Benchmark_MADDPG_20.npy', allow_pickle=True)




epoch_reward = []
E_HVAC_all_DQN_list = []
for i in range(len(Benchmark[:,7])):
    epoch_reward.append(np.sum((Benchmark[i,7])))
    E_HVAC_all_DQN_list.append(np.sum((Benchmark[i,2])))


plt.figure(figsize=(20,5), dpi=100)
# plt.plot(Benchmark[:,3], label='Save')
# plt.plot(epoch_reward, label='Reward')
# plt.plot(E_HVAC_all_DQN_list, label='HVAC_DQN')
# plt.plot(EPLUS.score, label='EPLUS.score')
plt.plot(Benchmark[0,1], label='HVAC_RBC')
plt.plot(Benchmark[10,2], label='HVAC_DQN')
plt.legend()




plt.figure(figsize=(20,7), dpi=100)
plt.plot(df2[0,1], label='HVAC_RBC')
plt.plot(df2[4,2], label='HVAC_DQN')
plt.legend()




plt.figure(figsize=(20,7), dpi=100)
plt.plot(df2[4,6], label='T_Violation')
# plt.plot(df2[4,2], label='HVAC_DQN')
plt.legend()




plt.figure(figsize=(20,7), dpi=100)
plt.plot(df2[1,7], label='episode_reward')
# plt.plot(df2[4,2], label='HVAC_DQN')
plt.legend()





plt.figure(figsize=(20,7), dpi=100)
plt.plot(df2[:,4], label='E_save')
# plt.plot(df2[4,2], label='HVAC_DQN')
plt.legend()




'''
Action Space Test

30 vs 3

'''




# episode_reward_3 = EPLUS.episode_reward


episode_reward_no  = np.load('episode_reward_nonheuristic.npy', allow_pickle=True)


episode_reward_3  = np.load('episode_reward_3_act.npy', allow_pickle=True)
episode_reward_30 = np.load('episode_reward_30_act.npy', allow_pickle=True)

x = np.arange(len(episode_reward_3))




plt.figure(figsize=(10,5), dpi=100)
plt.plot(episode_reward_no[100:], label='Reward - Binary')
plt.plot(episode_reward_3[100:],  label='Reward - Heuristic')
plt.legend(fontsize=16)
plt.xlabel('Training iter', fontsize=16)
plt.ylabel('Reward', fontsize=16)



sample_rate = 10

new_action_3  = []
new_action_30 = []



for i in range(len(episode_reward_3)):
    if i % sample_rate == 0:
        mean_action_3  = np.min(episode_reward_3[i:i+sample_rate])
        mean_action_30 = np.min(episode_reward_30[i:i+sample_rate])
        
        new_action_3.append(mean_action_3)
        new_action_30.append(mean_action_30)

x = np.arange(len(episode_reward_3)/sample_rate)





plt.figure(figsize=(10,5), dpi=100)
plt.plot(episode_reward_30[100:], label='Action Space = 30')
plt.plot(episode_reward_3[100:],  label='Action Space = 3')
plt.legend(fontsize=16)
plt.xlabel('Training iter', fontsize=16)
plt.ylabel('Reward', fontsize=16)



plt.figure(figsize=(10,7), dpi=100)
plt.scatter(x,episode_reward_30, label='Action Space = 30')
plt.scatter(x,episode_reward_3,  label='Action Space = 3')
plt.legend(fontsize=16)








plt.figure(figsize=(10,7), dpi=100)
plt.plot(new_action_30, label='Action Space = 30')
plt.plot(new_action_3,  label='Action Space = 3')
plt.legend(fontsize=16)











'''
E-T Trade off

'''



df1 = np.load('Benchmark_DQN_noweekday.npy', allow_pickle=True)
df2 = np.load('Benchmark_DQN_10_1zone.npy', allow_pickle=True)
df3 = np.load('Benchmark_DQN_10_20_1zone.npy', allow_pickle=True)
df4 = np.load('Benchmark_DQN_20_40_1zone.npy', allow_pickle=True)






df_2 = np.vstack((df1,df2,df3,df4))


delet_list = []
for i in range(len(df_2)):
    # if len(df[i,1]) == 0:
        # if np.sum(df[i]) == 0:
    if df_2[i,3] == 0:
        delet_list.append(i)


df_2 = np.delete(df_2, delet_list, 0)






plt.figure(figsize=(10,7), dpi=100)
plt.scatter(df[:,4], df[:,3])
# plt.plot(df[:,3], label='E_save')
# plt.plot(df[:,4], label='T_violation')
plt.xlabel('T_violation')
plt.ylabel('E_save')
plt.legend(fontsize=16)


plt.figure(figsize=(10,7), dpi=100)
# plt.scatter(df[:,4], df[:,3])
plt.plot(np.sort(df[:,3]*100), label='E_save', linewidth=3)
plt.plot(np.sort(df[:,4]*100), label='T_violation', linewidth=3)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Ratio %', fontsize=16)
plt.ylim(-100,100)
plt.legend(fontsize=16)





plt.figure(figsize=(10,5), dpi=100)
# plt.scatter(df[:,4], df[:,3])
# plt.plot(np.sort(df[:,3]*100), label='E_save', linewidth=3)
plt.scatter(np.arange(len(df)),(df[:,5]), label='T_offset', linewidth=3, color='limegreen')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Ratio %', fontsize=16)
# plt.ylim(-100,100)
plt.legend(fontsize=16)




plt.figure(figsize=(10,7), dpi=100)
# plt.scatter(df[:,4], df[:,3])
plt.plot(df[0,11], (df[10,13]), label='T_outdoor', linewidth=2)
plt.plot(df[0,11], (df[10,14]), label='T_zone_temp_2003', linewidth=2)
plt.xlabel('Time', fontsize=16)
plt.ylabel('Temperature', fontsize=16)
# plt.ylim(-100,100)
plt.legend(fontsize=16)







# np.sum(Benchmark[:,9])/3600






'''
Reward Test

Heuristic vs Binary

'''


episode_reward_no  = np.load('episode_reward_nonheuristic.npy', allow_pickle=True)
episode_reward_hu  = np.load('Benchmark_ONOFF_E_T_10x.npy', allow_pickle=True)

episode_reward_hu[19,7][0] = -33


plt.figure(figsize=(10,5), dpi=100)
plt.plot(episode_reward_no*5, label='Reward - Binary', color="teal")
plt.plot(episode_reward_hu[19,7],  label='Reward - Heuristic', color="lightcoral")
plt.legend(fontsize=16)
plt.xlabel('Training iter', fontsize=16)
plt.ylabel('Reward', fontsize=16)
plt.tight_layout()





df = np.load('Benchmark_E_T_5.npy', allow_pickle=True)


plt.figure(figsize=(10,5), dpi=100)
# plt.plot(df[:], label='Reward')
plt.plot(df[1,7],  label='Reward - Heuristic')
plt.legend(fontsize=16)
plt.xlabel('Training iter', fontsize=16)
plt.ylabel('Reward', fontsize=16)






df1 = np.load('Benchmark_E_T_1.npy', allow_pickle=True)

# if is_worktime:
#     E_factor = 1e-6
#     T_factor = 0.1
#     work_flag = 0
#     reward_signal = 0

#     E_save = E_factor
#     T_save = T_factor

# else:
#     E_factor = 1e-5/2
#     T_factor = 0.01
#     work_flag = 0
#     reward_signal = 0



df2 = np.load('Benchmark_E_T_2.npy', allow_pickle=True)


        # if is_worktime:
        #     E_factor = 1e-6
        #     T_factor = 0.1
        #     work_flag = 0
        #     reward_signal = 0
            
        #     E_save = E_factor
        #     T_save = T_factor
    
        # else:
        #     E_factor = 1e-5
        #     T_factor = 0.01
        #     work_flag = 0
        #     reward_signal = 0




df3 = np.load('Benchmark_E_T_3.npy', allow_pickle=True)


        # if is_worktime:
        #     E_factor = 1e-6
        #     T_factor = 0.1
        #     work_flag = 0
        #     reward_signal = 0
            
        #     E_save = E_factor
        #     T_save = T_factor
    
        # else:
        #     E_factor = 1e-4
        #     T_factor = 0.01
        #     work_flag = 0
        #     reward_signal = 0








df4 = np.load('Benchmark_E_T_4.npy', allow_pickle=True)


        # if is_worktime:
        #     E_factor = 1e-6
        #     T_factor = 0.1
        #     work_flag = 0
        #     reward_signal = 0
            
        #     E_save = E_factor
        #     T_save = T_factor
    
        # else:
        #     E_factor = 1e-6
        #     T_factor = 0.01
        #     work_flag = 0
        #     reward_signal = 0










df5 = np.load('Benchmark_E_T_5.npy', allow_pickle=True)

        # if is_worktime:
        #     E_factor = 1e-6
        #     T_factor = 0.1
        #     work_flag = 0
        #     reward_signal = 0
            
        #     E_save = E_factor
        #     T_save = T_factor
    
        # else:
        #     E_factor = 1e-5
        #     T_factor = 0.1
        #     work_flag = 0
        #     reward_signal = 0









df_1 = np.vstack((df1,df2,df3,df4,df5))


delet_list = []
for i in range(len(df_1)):
    # if len(df[i,1]) == 0:
        # if np.sum(df[i]) == 0:
    if df_1[i,3] == 0:
        delet_list.append(i)


df_1zone = np.delete(df_1, delet_list, 0)







x = df[0,11]



plt.figure(figsize=(10,5), dpi=100)
# plt.plot(df[:], label='Reward')
plt.scatter(df[:,4], df[:,3],  label='E_T')
plt.legend(fontsize=16)
plt.xlabel('T', fontsize=16)
plt.ylabel('E', fontsize=16)
plt.ylim(-0.5,0.5)








df_all = np.vstack((df_1,df_2))



plt.figure(figsize=(10,7), dpi=100)
# plt.scatter(df[:,4], df[:,3])
plt.plot(np.sort(df_all[:,3]*100), label='E_save', linewidth=3)
plt.plot(np.sort(df_all[:,4]*100), label='T_violation', linewidth=3)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Ratio %', fontsize=16)
plt.ylim(-100,100)
plt.legend(fontsize=16)






plt.figure(figsize=(10,5), dpi=100)
# plt.plot(df[:], label='Reward')
plt.scatter(df_all[:,4], df_all[:,3],  label='E_T')
plt.legend(fontsize=16)
plt.xlabel('T', fontsize=16)
plt.ylabel('E', fontsize=16)
plt.ylim(-0.5,0.5)













df1 = np.load('Benchmark_DQN_20_6zone.npy', allow_pickle=True)[:,:10]
df2 = np.load('Benchmark_weekday_40_60.npy', allow_pickle=True)[:,:10]
df3 = np.load('Benchmark_weekday_20_40.npy', allow_pickle=True)[:,:10]
df4 = np.load('Benchmark_weekday_40_60.npy', allow_pickle=True)[:,:10]
df5 = np.load('Benchmark_weekday_60_80.npy', allow_pickle=True)[:,:10]



df_3 = np.vstack((df1,df2,df3,df4,df5))


delet_list = []
for i in range(len(df_3)):
    # if len(df[i,1]) == 0:
        # if np.sum(df[i]) == 0:
    if df_3[i,3] == 0:
        delet_list.append(i)


df_6zone = np.delete(df_3, delet_list, 0)




plt.figure(figsize=(10,5), dpi=100)
# plt.plot(df[:], label='Reward')
plt.scatter(df_3[:,4], df_3[:,3],  label='E_T')
plt.legend(fontsize=16)
plt.xlabel('T', fontsize=16)
plt.ylabel('E', fontsize=16)
plt.ylim(-0.5,0.5)






plt.figure(figsize=(10,7), dpi=100)
# plt.scatter(df[:,4], df[:,3])
plt.plot(np.sort(df_3[:,3]*100), label='E_save', linewidth=3)
plt.plot(np.sort(df_3[:,4]*100), label='T_violation', linewidth=3)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Ratio %', fontsize=16)
plt.ylim(-100,100)
plt.legend(fontsize=16)








df = np.load('Benchmark.npy', allow_pickle=True)






plt.figure(figsize=(10,5), dpi=100)
# plt.plot(df[:], label='Reward')
plt.scatter(df[:,4], df[:,3],  label='E_T')
plt.legend(fontsize=16)
plt.xlabel('T', fontsize=16)
plt.ylabel('E', fontsize=16)
plt.ylim(-0.5,0.5)






plt.figure(figsize=(10,7), dpi=100)
# plt.scatter(df[:,4], df[:,3])
plt.plot(np.sort(df[:,3]*100), label='E_save', linewidth=3)
plt.plot(np.sort(df[:,4]*100), label='T_violation', linewidth=3)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Ratio %', fontsize=16)
plt.ylim(-100,100)
plt.legend(fontsize=16)



'''
E_T ONOFF
'''




'''ONOFF Model'''

df1 = np.load('Benchmark_ONOFF_20.npy', allow_pickle=True)
df2 = np.load('Benchmark_ONOFF_Best.npy', allow_pickle=True)
        # if is_worktime:
        #     E_factor = 1e-6
        #     T_factor = 0.01
        #     work_flag = 0
        #     reward_signal = 0
            
        #     E_save = E_factor
        #     T_save = T_factor
            
        # else:
        #     E_factor = 1e-6
        #     T_factor = 0
        #     work_flag = 0
        #     reward_signal = 0
df3 = np.load('Benchmark_ONOFF_E_T.npy', allow_pickle=True)
        # if is_worktime:
        #     E_factor = 1e-6
        #     T_factor = 0.01 *2
        #     work_flag = 0
        #     reward_signal = 0
            
        #     E_save = E_factor
        #     T_save = T_factor
            
        # else:
        #     E_factor = 1e-6 *2
        #     T_factor = 0
        #     work_flag = 0
        #     reward_signal = 0
        
df4 = np.load('Benchmark_ONOFF_E_T_5x.npy', allow_pickle=True)
        # if is_worktime:
        #     E_factor = 1e-6
        #     T_factor = 0.01 *5
        #     work_flag = 0
        #     reward_signal = 0
            
        #     E_save = E_factor
        #     T_save = T_factor
            
        # else:
        #     E_factor = 1e-6 *5
        #     T_factor = 0
        #     work_flag = 0
        #     reward_signal = 0
        
df5 = np.load('Benchmark_ONOFF_E_T_10x.npy', allow_pickle=True)
        # if is_worktime:
        #     E_factor = 1e-6
        #     T_factor = 0.001
        #     work_flag = 0
        #     reward_signal = 0
            
        #     E_save = E_factor
        #     T_save = T_factor
            
        # else:
        #     E_factor = 1e-6
        #     T_factor = 0
        #     work_flag = 0
        #     reward_signal = 0
        
        
df6 = np.load('Benchmark_ONOFF_air_infiltration_off.npy', allow_pickle=True)


df = np.vstack((df1,df2,df3,df4,df5,df6[:,:20]))


df_mean = np.load('Benchmark_MEANHVAC_E_T.npy', allow_pickle=True)




''' Energy '''

x = df5[2,11][288::]

plt.figure(figsize=(10,5), dpi=100)
plt.plot(x, df5[13,1][288::], label='HVAC_RBC')
plt.plot(x, df5[13,2][288::], label='HVAC_DQN')
plt.xlabel('Date', fontsize=16)
plt.ylabel('HVAC Electricity (J)', fontsize=16)
plt.legend(fontsize=16)
plt.tight_layout()  # otherwise the right y-label is slightly clipped




'''E_T grid'''


fig = plt.figure(figsize=(7,7), dpi=100)
ax = plt.subplot(111)


ax.set_ylim(25,45,1)
ax.set_xlim(0,14,1)
ax.set_xlabel('Temperature Violation (%)', fontsize=18)
ax.set_ylabel('Energy Saving (%)', fontsize=18)
ax.tick_params(labelsize=16)

# ax.spines['left'].set_position('zero')
# # ax.spines['right'].set_color('none')
# ax.spines['bottom'].set_position('zero')
# # ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')


# plt.plot(df[:], label='Reward')
ax.axvline(0, zorder=0)
ax.axhline(0, zorder=0)

# ax.xaxis.set_major_locator(MultipleLocator(1))
# ax.yaxis.set_major_locator(MultipleLocator(1))

ax.scatter(df2[:,4]*100, df2[:,3]*100,  label='E:T = 1:1', marker='o', color='C0', zorder=3)
ax.scatter(df3[:,4]*100, df3[:,3]*100,  label='E:T = 1:2', marker='o', color='C1', zorder=1)
ax.scatter(df4[:,4]*100, df4[:,3]*100,  label='E:T = 1:5', marker='o', color='C2', zorder=2)
ax.scatter(df1[:,4]*100, df1[:,3]*100,  label='E:T = 2:1', marker='o', color='C3')
ax.scatter(df5[:,4]*100, df5[:,3]*100,  label='E:T = 10:1', marker='o', color='C4')
# ax.scatter(df6[:,4], df6[:,3],  label='E:T = 0.1:0.1', marker='o', color='C5')


# ax.scatter(df_6zone[:,4], df_6zone[:,3],  label='DQN_6_Zone', marker='o', color='r')
# ax.scatter(df_1zone[:,4], df_1zone[:,3],  label='DQN_Single_Zone', marker='o', color='b')
# ax.scatter(df_mean[:,4], df_mean[:,3],  label='DQN_Mean_HVAC', marker='o', color='limegreen')
# ax.scatter(df[:,4], df[:,3], label='DQN_Cycle_AHU', marker='o', color='violet')

ax.legend(loc='lower right',fontsize=16)

ax.grid()
ax.set_axisbelow(True)

plt.title('Energy-Temperature Violation', fontsize=20)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()











for df in [df2,df3,df4,df1,df5,df6]:
    mean_E = np.mean(df[:,3])
    mean_T = np.mean(df[:,4])
    
    mean_offset = np.mean(df[:,5])
    
    
    sum_E = []
    for i in range(len(df)):
        E_i = np.sum(df[i,2])
        sum_E.append(E_i)
        
    sum_E = np.mean(sum_E)
    
    
    
    sum_Tvar = []
    for i in range(len(df)):
        Tvar_i = np.mean(df[i,18])
        sum_Tvar.append(Tvar_i)
        
    sum_Tvar = np.mean(sum_Tvar)
    
    
    print(sum_E, mean_E, mean_T, mean_offset, sum_Tvar)
    





# plt.figure(figsize=(10,7), dpi=100)
# # plt.scatter(df[:,4], df[:,3])
# plt.plot((df[:,3]*100), label='E_save', linewidth=3)
# plt.plot((df[:,4]*100), label='T_violation', linewidth=3)
# plt.xlabel('Epoch', fontsize=16)
# plt.ylabel('Ratio %', fontsize=16)
# # plt.ylim(-100,100)
# plt.legend(fontsize=16)




# Row 1 & 195204162760.59616 & 0.3389 & 0.0093 & 2.3811 & 1.9708 \
# Row 2 & 198393469346.52612 & 0.3281 & 0.0057 & 3.1960 & 1.4045 \
# Row 3 & 198222357010.09830 & 0.3287 & 0.0056 & 3.3570 & 1.3030 \
# Row 4 & 191584681284.83780 & 0.3512 & 0.0452 & 1.5236 & 2.9042 \
# Row 5 & 173230445554.96744 & 0.4133 & 0.1061 & 2.4719 & 1.6917 \
# Row 6 & 171060114159.77990 & 0.2883 & 0.0254 & 1.2598 & 2.1065 \




df2 = np.load('Benchmark_ONOFF_air_infiltration_off.npy', allow_pickle=True)

df = np.load('Benchmark_ONOFF_Best.npy', allow_pickle=True)



fig, ax1 = plt.subplots()
fontsize = 16

ax1.set_xlabel('Epoch', fontsize=fontsize)
ax1.tick_params(labelsize=fontsize)

color='C0'
p1, = ax1.plot(df[:,0].astype(int), (df[:,3]*100), label='E_save', linewidth=5, color=color)
ax1.set_ylabel('Energy Saving Ratio (%)', color=color, fontsize=fontsize)
ax1.set_ylim(0,50)
ax1.tick_params(axis='y', labelcolor=color, labelsize=fontsize)


ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color='C1'
p2, = ax2.plot(df[:,0].astype(int), (df[:,4]*100), label='T_violation', linewidth=5, color=color)
ax2.set_ylabel('Temperature Violation Ratio (%)', color=color, fontsize=fontsize)  # we already handled the x-label with ax1
ax2.set_ylim(0,10)
ax2.tick_params(axis='y', labelcolor=color, labelsize=fontsize)


# ax1.set_xticks(range(0, len(df), 1))  # set x-axis tick positions to 0-20
ax2.set_xticks(range(0, len(df), 1))
ax2.set_yticks(range(0, 10, 1))


ax1.set_title("Energy Saving - Temperature Violation", fontsize=20)

ax1.grid()
ax1.set_axisbelow(True)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig('./weather_data/'+str(weather_file[:-19])+'.png')
plt.show()





fig, ax1 = plt.subplots()
fontsize = 12

ax1.set_xlabel('Epoch', fontsize=fontsize)
ax1.tick_params(labelsize=fontsize)

color='C0'
p1, = ax1.plot(df2[:,0].astype(int), (df2[:,3]*100), label='E_save', linewidth=2, color=color)
ax1.set_ylabel('Energy SavingRatio %', color=color, fontsize=fontsize)
ax1.set_ylim(0,50)
ax1.tick_params(axis='y', labelcolor=color, labelsize=fontsize)


ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color='C1'
p2, = ax2.plot(df2[:,0].astype(int), (df2[:,4]*100), label='T_violation', linewidth=2, color=color)
ax2.set_ylabel('Temperature Violation Ratio %', color=color, fontsize=fontsize)  # we already handled the x-label with ax1
ax2.set_ylim(0,10)
ax2.tick_params(axis='y', labelcolor=color, labelsize=fontsize)


ax1.set_xticks(range(0, 20, 1))  # set x-axis tick positions to 0-20
ax2.set_xticks(range(0, 20, 1))

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()




















import pandas as pd


'''HVAC Action'''
action_list = df2[-1,8]

x = df2[0,11]



HVAC_action_list = []
for HC_1 in [0,1]:
    for HC_2 in [0,1]:
        for HC_3 in [0,1]:
            for HC_4 in [0,1]:
                for HC_5 in [0,1]:
                    for HC_6 in [0,1]:
                        HVAC_action_list.append([HC_1,HC_2,HC_3,HC_4,HC_5,HC_6])
    
    
    



AHU_1_list = []
AHU_2_list = []
AHU_3_list = []
AHU_4_list = []
AHU_5_list = []
AHU_6_list = []


for i in range(1,len(action_list)):
    HVAC_action = HVAC_action_list[action_list[i]]
    AHU_1,AHU_2,AHU_3,AHU_4,AHU_5,AHU_6 = HVAC_action
    AHU_1_list.append(AHU_1)
    AHU_2_list.append(AHU_2)
    AHU_3_list.append(AHU_3)
    AHU_4_list.append(AHU_4)
    AHU_5_list.append(AHU_5)
    AHU_6_list.append(AHU_6)





# plt.figure(figsize=(10,5), dpi=100)
# plt.plot(AHU_1_list, label='VAV_1_list')
# # plt.scatter(df[:,4], df[:,3],  label='E_T')
# plt.legend(fontsize=16)
# # plt.xlabel('T', fontsize=16)
# # plt.ylabel('E', fontsize=16)
# # plt.ylim(-0.5,0.5)



AHU_1_list.count(0), AHU_1_list.count(1)




df_AHU = pd.DataFrame(data=[x,
                            AHU_1_list,AHU_2_list,AHU_3_list,
                            AHU_4_list,AHU_5_list,AHU_6_list], 
                  index=['Time','VAV_1', 'VAV_2', 'VAV_3', 'VAV_4', 'VAV_5', 'VAV_6'])




Day_Load_Map = []
for i in range(288,len(x),288):
    AHU_1_load = df_AHU.loc['VAV_1'][i:i+288].tolist().count(1)
    AHU_2_load = df_AHU.loc['VAV_2'][i:i+288].tolist().count(1)
    AHU_3_load = df_AHU.loc['VAV_3'][i:i+288].tolist().count(1)
    AHU_4_load = df_AHU.loc['VAV_4'][i:i+288].tolist().count(1)
    AHU_5_load = df_AHU.loc['VAV_5'][i:i+288].tolist().count(1)
    AHU_6_load = df_AHU.loc['VAV_6'][i:i+288].tolist().count(1)
    
    Day_Load_Map.append([AHU_1_load,AHU_2_load,AHU_3_load,
                         AHU_4_load,AHU_5_load,AHU_6_load])
    if i%288==0: 
        print(df_AHU.loc['Time'][i])

Day_Load_Map = np.array(Day_Load_Map)
# Day_Load_Map = np.array(Day_Load_Map) /288 *100




Occ = np.sum(Day_Load_Map, axis=0)
Occ_all = np.sum(Occ)

O_1 = Occ[0]/Occ_all
O_2 = Occ[1]/Occ_all
O_3 = Occ[2]/Occ_all
O_4 = Occ[3]/Occ_all
O_5 = Occ[4]/Occ_all
O_6 = Occ[5]/Occ_all

print(O_1, O_2, O_3, O_4, O_5, O_6)
plt.plot(Occ)

# plt.figure(figsize=(10,5), dpi=100)
# plt.plot(Day_Load_Map)
# plt.show()


# c = plt.pcolormesh(Day_Load_Map[:], y_r, z, cmap='viridis_r', shading='gouraud')# 彩虹热力图
# c = plt.pcolormesh(x_r, y_r, z, cmap='viridis_r')# 普通热力图
# plt.colorbar(c, label='AUPR')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.savefig('heatmap.tif', dpi=300)
# plt.show()





'''Day load bar plot'''
fig = plt.figure(figsize = (10, 20))
ax1 = plt.subplot(6,1,1)
ax2 = plt.subplot(6,1,2)
ax3 = plt.subplot(6,1,3)
ax4 = plt.subplot(6,1,4)
ax5 = plt.subplot(6,1,5)
ax6 = plt.subplot(6,1,6)


# ax1.set_box_aspect((3,1,1)) 

alpha = 1
fontsize=20

ax1.bar(np.arange(364), Day_Load_Map[:,0], color='C0', alpha=alpha, label='VAV_1')
ax2.bar(np.arange(364), Day_Load_Map[:,1], color='C1', alpha=alpha, label='VAV_2')
ax3.bar(np.arange(364), Day_Load_Map[:,2], color='C2', alpha=alpha, label='VAV_3')
ax4.bar(np.arange(364), Day_Load_Map[:,3], color='C3', alpha=alpha, label='VAV_4')
ax5.bar(np.arange(364), Day_Load_Map[:,4], color='C4', alpha=alpha, label='VAV_5')
ax6.bar(np.arange(364), Day_Load_Map[:,5], color='C5', alpha=alpha, label='VAV_6')

for ax in [ax1,ax2,ax3,ax4,ax5,ax6]:
    ax.set_xlabel('Day', fontsize=fontsize)
    ax.set_ylabel('VAV Load (%)', fontsize=fontsize)
    ax.set_ylim(0,100)
    ax.tick_params(labelsize=16)
    ax.legend(loc='upper right',fontsize=fontsize)

fig.tight_layout()
# plt.subplots_adjust(wspace=0, hspace=0)
plt.show()






VAV_all = 1
np.sum(Day_Load_Map, 1)




fontsize = 16

fig = plt.figure(figsize=(10,5), dpi=100)
ax = plt.subplot(111)
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

colors = ['C0','C1','C2','C3','C4','C5']

ax.stackplot(np.arange(364), 
             Day_Load_Map[:,0]/VAV_all,
             Day_Load_Map[:,1]/VAV_all,
             Day_Load_Map[:,2]/VAV_all,
             Day_Load_Map[:,3]/VAV_all,
             Day_Load_Map[:,4]/VAV_all,
             Day_Load_Map[:,5]/VAV_all,
             colors=colors)

ax.legend(labels=['VAV_1', 'VAV_2', 'VAV_3', 'VAV_4', 'VAV_5', 'VAV_6'], 
          fontsize = fontsize)

ax.tick_params(labelsize=fontsize)
ax.set_xlabel('Date', fontsize=fontsize)
ax.set_ylabel('Heating-Cooling Energy Ratio (%)', fontsize=fontsize)


plt.title('Heating-Cooling Energy Comparison', fontsize=20)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()




















'''3D Bar'''

fig = plt.figure(figsize = (20, 20))
ax1 = fig.add_subplot(projection='3d')
# ax1.set_box_aspect((3,1,1)) 

alpha = 1
fontsize=20

ax1.bar(np.arange(364), Day_Load_Map[:,0], zs=1, zdir='y', color='C0', alpha=alpha, label='AHU_1')
ax1.bar(np.arange(364), Day_Load_Map[:,1], zs=2, zdir='y', color='C1', alpha=alpha, label='AHU_2')
ax1.bar(np.arange(364), Day_Load_Map[:,2], zs=3, zdir='y', color='C2', alpha=alpha, label='AHU_3')
ax1.bar(np.arange(364), Day_Load_Map[:,3], zs=4, zdir='y', color='C3', alpha=alpha, label='AHU_4')
ax1.bar(np.arange(364), Day_Load_Map[:,4], zs=5, zdir='y', color='C4', alpha=alpha, label='AHU_5')
ax1.bar(np.arange(364), Day_Load_Map[:,5], zs=6, zdir='y', color='C5', alpha=alpha, label='AHU_6')

ax1.set_zlim(0,100)
ax1.set_xlabel('Day', fontsize=fontsize)
ax1.set_ylabel('AHU', fontsize=fontsize)
ax1.set_zlabel('Load (%)', fontsize=fontsize)
ax1.tick_params(labelsize=12)

ax1.legend(fontsize=fontsize)

# fig.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()






















df = pd.DataFrame(data=Day_Load_Map, 
                  columns=['VAV_1', 'VAV_2', 'VAV_3', 'VAV_4', 'VAV_5', 'VAV_6'])


df.plot.barh(stacked=True, edgecolor='none')
plt.legend(df.columns)
plt.legend(loc="best")




# 变量
labels = ['VAV_1', 'VAV_2', 'VAV_3', 'VAV_4', 'VAV_5', 'VAV_6']
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
title = 'Day Load Map \n'
subtitle = 'Proportion of AHU load'


# def plot_stackedbar_p(df, labels, colors, title, subtitle):
    
fields = df.columns.tolist()

fig, ax = plt.subplots(1, figsize=(12, 10))

left = len(df) * [0]
for idx, name in enumerate(fields):
    plt.barh(df.index, df[name], left = left, color=colors[idx])
    left = left + df[name]


plt.title(title, loc='left')
plt.text(0, ax.get_yticks()[-1] + 0.75, subtitle)


plt.legend(labels, bbox_to_anchor=([0.58, 1, 0, 0]), ncol=4, frameon=False)


ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)


xticks = np.arange(0,1.1,0.1)
xlabels = ['{}%'.format(i) for i in np.arange(0,101,10)]
plt.xticks(xticks, xlabels)


plt.ylim(-0.5, ax.get_yticks()[-1] + 0.5)
ax.xaxis.grid(color='gray', linestyle='dashed')
plt.show()

# plot_stackedbar_p(df_filter, labels, colors, title, subtitle)














import seaborn as sns




Hour_Load_Map = np.zeros((len(Day_Load_Map),24))

day=0

for i in range(288,len(x),288):
    hour=0
    Day_load_i = df_AHU.iloc[1:,i:i+288]
    
    for j in range(0,288,12):
        Hour_load_j = Day_load_i.iloc[:, j:j+12]
        Hour_Load_Map[day,hour] = Hour_load_j.sum().sum()
        
        hour = hour+1
        
    day = day+1
    if i%288==0: 
        print(df_AHU.loc['Time'][i])

Hour_Load_Map = np.array(Hour_Load_Map)







df_heatmap_hour = pd.DataFrame(data=Hour_Load_Map/(12*6)*100)


df_heatmap = pd.DataFrame(data=Day_Load_Map/288*100, 
                  columns=['VAV_1', 'VAV_2', 'VAV_3', 'VAV_4', 'VAV_5', 'VAV_6'])







sns.set(font_scale=1.5)
sns.set_context({"figure.figsize":(8,8)})


fig = plt.figure(figsize = (20, 10))

# fig, axes = plt.subplots(1, 2)


ax1 = sns.heatmap(df_heatmap,square=False, vmin=0, vmax=100,cmap='viridis',
            cbar_kws={'label': 'VAV Load (%)'}) 
# ax1 = sns.heatmap(hmap, cbar=0, cmap="YlGnBu",linewidths=2, ax=ax0,vmax=3000, vmin=0, square=True)
ax1.set_ylabel('Day')




fig = plt.figure(figsize = (20, 10))

ax2 = sns.heatmap(df_heatmap_hour,square=False, vmin=0, vmax=100, cmap='jet',
            cbar_kws={'label': 'VAV Load (%)'},
            xticklabels=2) 
ax2.set_xlabel('Hour', fontsize=24)
ax2.set_ylabel('Day', fontsize=24)





plt.show()








'''
Thermal Comfort & Temperature Violation
'''
EPLUS.work_time
work_time = np.tile(EPLUS.work_time, (6, 1)).T

df6 = np.load('Benchmark_ONOFF_air_infiltration_off.npy', allow_pickle=True)

df6[:,3]
df6[:,4]

zone_temp = np.array(df6[19,16])

zone_temp_work = [zone_temp*work_time]
non_zero = np.any(zone_temp_work, axis=0)

zone_temp_work = zone_temp_work[0]
zone_temp_work = zone_temp_work[non_zero]


plt.figure(figsize=(20,10), dpi=100)
plt.axhline(y=77, label='Upper comfort line', color='g')
plt.axhline(y=68, label='Lower comfort line', color='g')
        
# plt.plot(EPLUS.E_Heating, label='E_Heating')
# plt.plot(EPLUS.E_Cooling, label='E_Cooling')
# plt.plot(Benchmark[:, 3], label='E_save')
# plt.plot(Benchmark[:, 4], label='T_violation')
plt.plot(zone_temp_work, label='y_zone_temp')
plt.legend()

















'''
Solar vs Load
'''

df = np.load('Benchmark_ONOFF_air_infiltration_off.npy', allow_pickle=True)
# df = np.load('Benchmark_ONOFF_DRL_OFF.npy', allow_pickle=True)



import pandas as pd


'''HVAC Action'''
action_list = df[-1,8]

x = df[0,11]


HVAC_action_list = []
for HC_1 in [0,1]:
    for HC_2 in [0,1]:
        for HC_3 in [0,1]:
            for HC_4 in [0,1]:
                for HC_5 in [0,1]:
                    for HC_6 in [0,1]:
                        HVAC_action_list.append([HC_1,HC_2,HC_3,HC_4,HC_5,HC_6])
    
VAV_1_list = []
VAV_2_list = []
VAV_3_list = []
VAV_4_list = []
VAV_5_list = []
VAV_6_list = []


for i in range(1,len(action_list)):
    HVAC_action = HVAC_action_list[action_list[i]]
    VAV_1,VAV_2,VAV_3,VAV_4,VAV_5,VAV_6 = HVAC_action
    VAV_1_list.append(VAV_1)
    VAV_2_list.append(VAV_2)
    VAV_3_list.append(VAV_3)
    VAV_4_list.append(VAV_4)
    VAV_5_list.append(VAV_5)
    VAV_6_list.append(VAV_6)





plt.figure(figsize=(10,5), dpi=100)
plt.plot(VAV_1_list, label='VAV_1_list')
# plt.scatter(df[:,4], df[:,3],  label='E_T')
plt.legend(fontsize=16)
# plt.xlabel('T', fontsize=16)
# plt.ylabel('E', fontsize=16)
# plt.ylim(-0.5,0.5)



VAV_1_list.count(0), VAV_1_list.count(1)




df_VAV = pd.DataFrame(data=[x,
                            VAV_1_list,VAV_2_list,VAV_3_list,
                            VAV_4_list,VAV_5_list,VAV_6_list], 
                  index=['Time','VAV_1','VAV_2','VAV_3','VAV_4','VAV_5','VAV_6'])




Day_Load_Map = []
for i in range(288,len(x),288):
    VAV_1_load = df_VAV.loc['VAV_1'][i:i+288].tolist().count(1)
    VAV_2_load = df_VAV.loc['VAV_2'][i:i+288].tolist().count(1)
    VAV_3_load = df_VAV.loc['VAV_3'][i:i+288].tolist().count(1)
    VAV_4_load = df_VAV.loc['VAV_4'][i:i+288].tolist().count(1)
    VAV_5_load = df_VAV.loc['VAV_5'][i:i+288].tolist().count(1)
    VAV_6_load = df_VAV.loc['VAV_6'][i:i+288].tolist().count(1)
    
    Day_Load_Map.append([VAV_1_load,VAV_2_load,VAV_3_load,
                         VAV_4_load,VAV_5_load,VAV_6_load])
    if i%288==0: 
        print(df_VAV.loc['Time'][i])

Day_Load_Map = np.array(Day_Load_Map)





plt.figure(figsize=(10,5), dpi=100)
plt.plot(Day_Load_Map)
plt.show()






df = np.load('Benchmark_ONOFF_air_infiltration_off.npy', allow_pickle=True)
# df = np.load('Benchmark_ONOFF_DRL_OFF.npy', allow_pickle=True)

df2 = np.load('Benchmark_VAV_2_OFF.npy', allow_pickle=True)
    # Benchmark[0, 27] = EPLUS.work_time


work_time = df2[0, 27]

# non_zero = np.where(np.array(work_time))

time_line   = np.array(df[0,11])
HVAC_energy = np.array(df[-1,15])
window_heat = np.array(df[-1,25])

VAV_1_list = np.array(VAV_1_list)
VAV_2_list = np.array(VAV_2_list)
VAV_3_list = np.array(VAV_3_list)
VAV_4_list = np.array(VAV_4_list)
VAV_5_list = np.array(VAV_5_list)
VAV_6_list = np.array(VAV_6_list)


VAV_1_energy = VAV_1_list * HVAC_energy / 6
VAV_2_energy = VAV_2_list * HVAC_energy / 6
VAV_3_energy = VAV_3_list * HVAC_energy / 6
VAV_4_energy = VAV_4_list * HVAC_energy / 6
VAV_5_energy = VAV_5_list * HVAC_energy / 6
VAV_6_energy = VAV_6_list * HVAC_energy / 6





plt.figure(figsize=(20,10), dpi=100)
window_heat_1 = window_heat[:,0]/np.max(window_heat)
VAV_1_energy_1 = VAV_1_energy/np.max(VAV_1_energy)
plt.plot(time_line, VAV_1_energy_1, label='VAV_1_list')
plt.plot(time_line, window_heat_1, label='window_heat')
plt.legend()
plt.show()

plt.figure(figsize=(20,10), dpi=100)
window_heat_2 = window_heat[:,1]/np.max(window_heat)
VAV_2_energy_2 = VAV_2_energy/np.max(VAV_2_energy)
plt.plot(time_line, VAV_2_energy_2, label='VAV_2_list')
plt.plot(time_line, window_heat_2, label='window_heat')
plt.legend()
plt.show()

plt.figure(figsize=(20,10), dpi=100)
window_heat_3 = window_heat[:,2]/np.max(window_heat)
VAV_3_energy_3 = VAV_3_energy/np.max(VAV_3_energy)
plt.plot(time_line, VAV_3_energy_3, label='VAV_3_list')
plt.plot(time_line, window_heat_3, label='window_heat')
plt.legend()
plt.show()

plt.figure(figsize=(20,10), dpi=100)
window_heat_4 = window_heat[:,3]/np.max(window_heat)
VAV_4_energy_4 = VAV_4_energy/np.max(VAV_4_energy)
plt.plot(time_line, VAV_4_energy_4, label='VAV_4_list')
plt.plot(time_line, window_heat_4, label='window_heat')
plt.legend()
plt.show()

plt.figure(figsize=(20,10), dpi=100)
window_heat_5 = window_heat[:,4]/np.max(window_heat)
VAV_5_energy_5 = VAV_5_energy/np.max(VAV_5_energy)
plt.plot(time_line, VAV_5_energy_5, label='VAV_5_list')
plt.plot(time_line, window_heat_5, label='window_heat')
plt.legend()
plt.show()

plt.figure(figsize=(20,10), dpi=100)
window_heat_6 = window_heat[:,5]/np.max(window_heat)
VAV_6_energy_6 = VAV_6_energy/np.max(VAV_6_energy)
plt.plot(time_line, VAV_6_energy_6, label='VAV_6_list')
plt.plot(time_line, window_heat_6, label='window_heat')
plt.legend()
plt.show()













# summer
day = 180
length = 60

summer = time_line[288*day:288*(day+1)]
summer_VAV_energy = []
summer_window_heat = []
for i in range(288*day,288*(day+length),288):
    summer_VAV_energy.append(VAV_4_energy_4[i:i+288])
    summer_window_heat.append(window_heat_4[i:i+288])

summer_VAV_energy = np.mean(summer_VAV_energy, axis=0)
summer_window_heat = np.mean(summer_window_heat, axis=0)


# winter
day = 300
length = 60

winter = time_line[288*day:288*(day+1)]
winter_VAV_energy = []
winter_window_heat = []
for i in range(288*day,288*(day+length),288):
    winter_VAV_energy.append(VAV_4_energy_4[i:i+288])
    winter_window_heat.append(window_heat_4[i:i+288])

winter_VAV_energy = np.mean(winter_VAV_energy, axis=0)
winter_window_heat = np.mean(winter_window_heat, axis=0)





fig = plt.figure(figsize=(10,6), dpi=100)
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)


ax1.plot(summer,summer_VAV_energy, label='VAV_4_Load', linewidth=3)
ax1.plot(summer,summer_window_heat, label='Solar Energy Gain', linewidth=3)
ax1.set_ylim(0,0.5)
ax1.legend(fontsize=16)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
ax1.set_title('July Solar Energy vs VAV load', fontsize=16)
# ax1.grid()
# ax1.set_axisbelow(True)


ax2.plot(winter,winter_VAV_energy, label='VAV_4_Load', color='C0', linewidth=3)
ax2.plot(winter,winter_window_heat, label='Solar Energy Gain', color='C1', linewidth=3)
ax2.set_ylim(0,0.5)
ax2.legend(fontsize=16)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
ax2.set_title('December Solar Energy vs VAV load', fontsize=16)


fig.tight_layout()  
plt.show()










# summer
day = 180
length = 60

summer = time_line[288*day:288*(day+1)]
summer_VAV_energy = []
summer_window_heat = []
for i in range(288*day,288*(day+length),288):
    summer_VAV_energy.append(VAV_6_energy_6[i:i+288])
    summer_window_heat.append(window_heat_6[i:i+288])

summer_VAV_energy = np.mean(summer_VAV_energy, axis=0)
summer_window_heat = np.mean(summer_window_heat, axis=0)


# winter
day = 300
length = 60

winter = time_line[288*day:288*(day+1)]
winter_VAV_energy = []
winter_window_heat = []
for i in range(288*day,288*(day+length),288):
    winter_VAV_energy.append(VAV_6_energy_6[i:i+288])
    winter_window_heat.append(window_heat_6[i:i+288])

winter_VAV_energy = np.mean(winter_VAV_energy, axis=0)
winter_window_heat = np.mean(winter_window_heat, axis=0)





fig = plt.figure(figsize=(10,6), dpi=100)
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)


ax1.plot(summer,summer_VAV_energy, label='VAV_6_Load', linewidth=3)
ax1.plot(summer,summer_window_heat, label='Solar Energy Gain', linewidth=3)
ax1.set_ylim(0,0.5)
ax1.legend(fontsize=16)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
ax1.set_title('July Solar Energy vs VAV load', fontsize=16)
# ax1.grid()
# ax1.set_axisbelow(True)


ax2.plot(winter,winter_VAV_energy, label='VAV_6_Load', color='C0', linewidth=3)
ax2.plot(winter,winter_window_heat, label='Solar Energy Gain', color='C1', linewidth=3)
ax2.set_ylim(0,0.5)
ax2.legend(fontsize=16)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
ax2.set_title('December Solar Energy vs VAV load', fontsize=16)


fig.tight_layout()  
plt.show()












# summer
day = 180
length = 30

summer = time_line[288*day:288*(day+1)]
summer_VAV_energy = []
# summer_window_heat = []
for i in range(288*day,288*(day+length),288):
    summer_VAV_energy.append(VAV_3_energy_3[i:i+288])
    # summer_window_heat.append(window_heat_6[i:i+288])

summer_VAV_energy = np.mean(summer_VAV_energy, axis=0)
# summer_window_heat = np.mean(summer_window_heat, axis=0)


# winter
day = 300
length = 30

winter = time_line[288*day:288*(day+1)]
winter_VAV_energy = []
# winter_window_heat = []
for i in range(288*day,288*(day+length),288):
    winter_VAV_energy.append(VAV_3_energy_3[i:i+288])
    # winter_window_heat.append(window_heat_6[i:i+288])

winter_VAV_energy = np.mean(winter_VAV_energy, axis=0)
# winter_window_heat = np.mean(winter_window_heat, axis=0)





fig = plt.figure(figsize=(10,6), dpi=100)
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)


ax1.plot(summer,summer_VAV_energy, label='VAV_3_Load', linewidth=3)
# ax1.plot(summer,summer_window_heat, label='Solar Energy Gain', linewidth=3)
ax1.set_ylim(0,0.5)
ax1.legend(fontsize=16)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
ax1.set_title('July Solar Energy vs VAV load', fontsize=16)
# ax1.grid()
# ax1.set_axisbelow(True)


ax2.plot(winter,winter_VAV_energy, label='VAV_3_Load', color='C0', linewidth=3)
# ax2.plot(winter,winter_window_heat, label='Solar Energy Gain', color='C1', linewidth=3)
ax2.set_ylim(0,0.5)
ax2.legend(fontsize=16)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
ax2.set_title('December Solar Energy vs VAV load', fontsize=16)


fig.tight_layout()  
plt.show()








# summer
day = 180
length = 60

summer = time_line[288*day:288*(day+1)]
summer_VAV_energy = []
summer_window_heat = []
for i in range(288*day,288*(day+length),288):
    summer_VAV_energy.append(VAV_2_energy_2[i:i+288])
    summer_window_heat.append(window_heat_2[i:i+288])

summer_VAV_energy = np.mean(summer_VAV_energy, axis=0)
summer_window_heat = np.mean(summer_window_heat, axis=0)


# winter
day = 300
length = 60

winter = time_line[288*day:288*(day+1)]
winter_VAV_energy = []
winter_window_heat = []
for i in range(288*day,288*(day+length),288):
    winter_VAV_energy.append(VAV_2_energy_2[i:i+288])
    winter_window_heat.append(window_heat_2[i:i+288])

winter_VAV_energy = np.mean(winter_VAV_energy, axis=0)
winter_window_heat = np.mean(winter_window_heat, axis=0)





fig = plt.figure(figsize=(10,6), dpi=100)
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)


ax1.plot(summer,summer_VAV_energy, label='VAV_2_Load', linewidth=3)
ax1.plot(summer,summer_window_heat, label='Solar Energy Gain', linewidth=3)
ax1.set_ylim(0,0.5)
ax1.legend(fontsize=16)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
ax1.set_title('July Solar Energy vs VAV load', fontsize=16)
# ax1.grid()
# ax1.set_axisbelow(True)


ax2.plot(winter,winter_VAV_energy, label='VAV_2_Load', color='C0', linewidth=3)
ax2.plot(winter,winter_window_heat, label='Solar Energy Gain', color='C1', linewidth=3)
ax2.set_ylim(0,0.5)
ax2.legend(fontsize=16)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
ax2.set_title('December Solar Energy vs VAV load', fontsize=16)


fig.tight_layout()  
plt.show()







indices = [1, 2, 3, 4, 5, 6]
for i in indices:
    VAV_energy = eval(f"VAV_{i}_energy_{i}")
    window_heat = eval(f"window_heat_{i}")
    
    
    # summer
    day = 180
    length = 30
    
    summer_VAV_energy = []
    summer_window_heat = []
    for j in range(288*day, 288*(day+length), 288):
        summer_VAV_energy.append(VAV_energy[j:j+288])
        summer_window_heat.append(window_heat[j:j+288])

    summer_VAV_energy = np.mean(summer_VAV_energy, axis=0)
    summer_window_heat = np.mean(summer_window_heat, axis=0)


    # winter
    day = 300
    length = 30

    winter_VAV_energy = []
    winter_window_heat = []
    for j in range(288*day, 288*(day+length), 288):
        winter_VAV_energy.append(VAV_energy[j:j+288])
        winter_window_heat.append(window_heat[j:j+288])

    winter_VAV_energy = np.mean(winter_VAV_energy, axis=0)
    winter_window_heat = np.mean(winter_window_heat, axis=0)

    # Plot figures here using summer_VAV_energy, summer_window_heat, winter_VAV_energy, winter_window_heat
    
    
    
    fig = plt.figure(figsize=(7,4), dpi=100)
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    
    
    ax1.plot(summer,summer_VAV_energy, label=f'VAV_{i}_Load', linewidth=3)
    ax1.plot(summer,summer_window_heat, label='Solar Energy Gain', linewidth=3)
    ax1.set_ylim(0,0.5)
    ax1.legend(fontsize=16)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax1.set_title('July Solar Energy vs VAV load', fontsize=16)
    # ax1.grid()
    # ax1.set_axisbelow(True)
    
    
    ax2.plot(winter,winter_VAV_energy, label=f'VAV_{i}_Load', color='C0', linewidth=3)
    ax2.plot(winter,winter_window_heat, label='Solar Energy Gain', color='C1', linewidth=3)
    ax2.set_ylim(0,0.5)
    ax2.legend(fontsize=16)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax2.set_title('December Solar Energy vs VAV load', fontsize=16)
    
    
    fig.tight_layout()  
    plt.show()
    







df = np.load('Benchmark_ONOFF_DRL_OFF.npy', allow_pickle=True)

VAV_energy_1 = np.array(df[0,1]) / 7
VAV_energy_2 = np.array(df[0,2]) / 7

time_line   = np.array(df[0,11])
window_heat = np.array(df[-1,25])

plt.plot(time_line, VAV_energy)
plt.plot(time_line, window_heat[:,3])


plt.plot(time_line, VAV_energy_1)
plt.plot(time_line, VAV_energy_2)







work_time = [np.array(EPLUS.work_time)]
non_zero = np.any(work_time, axis=0)

HVAC_action_list = []
for HC_1 in [0,1]:
    for HC_2 in [0,1]:
        for HC_3 in [0,1]:
            for HC_4 in [0,1]:
                for HC_5 in [0,1]:
                    for HC_6 in [0,1]:
                        HVAC_action_list.append([HC_1,HC_2,HC_3,HC_4,HC_5,HC_6])
    


E_T = []

'''
6zone_2 vs Weather Location
'''

for weather in [
                'USA_SC_Greenville-Spartanburg.Intl.AP.723120_TMY3.epw',
                'USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3.epw',
                'USA_CA_Los.Angeles.722950_TMY2.epw',
                'USA_FL_Miami.Intl.AP.722020_TMY3.epw',
                'USA_MA_Boston-Logan.Intl.AP.725090_TMY3.epw',
                'USA_MN_International.Falls.Intl.AP.727470_TMY3.epw',
                'USA_TX_Houston-William.P.Hobby.AP.722435_TMY3.epw'
                ]:
    
    
    weather_file = weather
    print(weather_file)
        


    df = np.load('./weather_data/'+weather_file+'.npy', allow_pickle=True)
    
    
    E_mean = np.mean(df[:,3])
    T_mean = np.mean(df[:,4])
    
    E_T.append([E_mean, T_mean])
    
    
    
    
    fig, ax1 = plt.subplots()
    fontsize = 16
    
    ax1.set_xlabel('Epoch', fontsize=fontsize)
    ax1.tick_params(labelsize=fontsize)
    
    color='C0'
    p1, = ax1.plot(df[:,0].astype(int), (df[:,3]*100), label='E_save', linewidth=5, color=color)
    ax1.set_ylabel('Energy SavingRatio %', color=color, fontsize=fontsize)
    ax1.set_ylim(0,50)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=fontsize)
    
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color='C1'
    p2, = ax2.plot(df[:,0].astype(int), (df[:,4]*100), label='T_violation', linewidth=5, color=color)
    ax2.set_ylabel('Temperature Violation Ratio %', color=color, fontsize=fontsize)  # we already handled the x-label with ax1
    ax2.set_ylim(0,10)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=fontsize)
    
    
    ax1.set_xticks(range(0, len(df), 1))  # set x-axis tick positions to 0-20
    ax2.set_xticks(range(0, len(df), 1))
    
    ax1.set_title(str(weather_file[4:-15]), fontsize=20)
    
    ax1.grid()
    ax1.set_axisbelow(True)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('./weather_data/'+str(weather_file[:-19])+'.png')
    plt.show()


    mean_E = np.max(df[:,3])
    mean_T = np.min(df[:,4])
    mean_offset = np.min(df[:,5])
    
    
    oa = df[0,13][288:]
    humd = df[0,21][288:]
    
    max_oa = np.max(oa)
    min_oa = np.min(oa)
    mean_oa = np.mean(oa)
    
    max_humd = np.max(humd)
    min_humd = np.min(humd)
    mean_humd = np.mean(humd)


    sum_E = []
    for i in range(len(df)):
        E_i = np.sum(df[i,2][288:])
        sum_E.append(E_i)
        
    sum_E = np.mean(sum_E)
    
    
    
    mean_Tvar = []
    for i in range(len(df)):
        Tvar_i = np.mean(df[i,18][288:])
        mean_Tvar.append(Tvar_i)
        
    mean_Tvar = np.mean(mean_Tvar)
    
    
    HVAC_load = []
    # Benchmark[epoch, 8] = EPLUS.action_list
    # Benchmark[0, 27] = EPLUS.work_time
    
    action_list = np.array(df[-1,8][288:-1])
    work_time = np.array(df[0,27][288:])
    
    action_list = action_list[work_time==1]
    for i in range(len(action_list)):
        action = action_list[i]
        HVAC_load_i = HVAC_action_list[action]
        HVAC_load_i = HVAC_load_i.count(1)/len(HVAC_load_i)
        HVAC_load.append(HVAC_load_i)
        
    mean_HVAC_load = np.mean(HVAC_load)
        
    
    # print(weather_file)
    print(mean_E,mean_T,mean_offset,sum_E,mean_Tvar)
    print(mean_oa,max_oa,min_oa)
    print(mean_humd, max_humd,min_humd)
    print(mean_humd, max_humd,min_humd)
    print(mean_HVAC_load)


















df = np.load('./weather_data/Weather_Data.npy', allow_pickle=True)


T_H = []

for i in range(len(df)):
    weather_name = df[i,0]
    print(weather_name)
    
    time_line = df[i,1][144:]
    time_line = time_line[::144]
    
    
    
    # for j in range(len(time_line)):
    #     time_line[j] = time_line[j].strftime('%m-%d')

    
    oa_temp = df[i,2][144:]    
    oa_humd = df[i,3][144:]
    
    T_H.append([np.mean(oa_temp),np.mean(oa_humd)])


    oa_temp = oa_temp[::144]
    oa_humd = oa_humd[::144]

    fig, ax1 = plt.subplots()
    fontsize = 16
    
    ax1.set_xlabel('Date', fontsize=fontsize)
    ax1.tick_params(labelsize=fontsize)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'seagreen'
    p2, = ax1.plot(time_line, oa_humd, label='Outdoor Relative Humidity', linewidth=2, color=color)
    ax2.set_ylabel('Outdoor Relative Humidity (%)', color=color, fontsize=fontsize)  # we already handled the x-label with ax1
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=fontsize)
    
    color = 'C1'
    p1, = ax2.plot(time_line, oa_temp, label='Outdoor Temperature', linewidth=2, color=color)
    ax1.set_ylabel('Outdoor Temperature (F)', color=color, fontsize=fontsize)
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=fontsize)
    
    
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax1.grid()
    ax1.set_axisbelow(True)
    
    plt.title(str(weather_name[4:-15]), fontsize=20)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('./weather_data/'+str(i)+'.png')
    plt.show()
    
    
    # Heat & Cool
    # Weather_Data[loc, 5] = EPLUS.E_Heating
    # Weather_Data[loc, 6] = EPLUS.E_Cooling
    
    E_Heating = df[i,5][144:]
    E_Cooling = df[i,6][144:]
    
    E_Heating = np.sum(E_Heating)
    E_Cooling = np.sum(E_Cooling)
    
    E_Heating_ratio = E_Heating/(E_Heating+E_Cooling)
    E_Cooling_ratio = E_Cooling/(E_Heating+E_Cooling)

    print(E_Heating_ratio, E_Cooling_ratio)






'''E_T grid'''

E_T = np.array(E_T)

fontsize = 16

fig = plt.figure(figsize=(6,6), dpi=100)
ax = plt.subplot(111)


ax.set_ylim(-10,50,1)
ax.set_xlim(-5,10,1)
ax.set_xlabel('Temperature Violation (%)', fontsize=fontsize)
ax.set_ylabel('Energy Saving (%)', fontsize=fontsize)
ax.tick_params(labelsize=fontsize)

# ax.spines['left'].set_position('zero')
# # ax.spines['right'].set_color('none')
# ax.spines['bottom'].set_position('zero')
# # ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')


# plt.plot(df[:], label='Reward')
ax.axvline(0, zorder=0)
ax.axhline(0, zorder=0)

# ax.xaxis.set_major_locator(MultipleLocator(0.1))
# ax.yaxis.set_major_locator(MultipleLocator(0.1))

ax.scatter(E_T[0,1]*100, E_T[0,0]*100,  label='SC', marker='o', s=100)
ax.scatter(E_T[1,1]*100, E_T[1,0]*100,  label='AZ', marker='o', s=100)
ax.scatter(E_T[2,1]*100, E_T[2,0]*100,  label='CA', marker='o', s=100)
ax.scatter(E_T[3,1]*100, E_T[3,0]*100,  label='FL', marker='o', s=100)
ax.scatter(E_T[4,1]*100, E_T[4,0]*100,  label='MA', marker='o', s=100)
ax.scatter(E_T[5,1]*100, E_T[5,0]*100,  label='MN', marker='o', s=100)
ax.scatter(E_T[6,1]*100, E_T[6,0]*100,  label='TX', marker='o', s=100)



# ax.scatter(df_6zone[:,4], df_6zone[:,3],  label='DQN_6_Zone', marker='o', color='r')
# ax.scatter(df_1zone[:,4], df_1zone[:,3],  label='DQN_Single_Zone', marker='o', color='b')
# ax.scatter(df_mean[:,4], df_mean[:,3],  label='DQN_Mean_HVAC', marker='o', color='limegreen')
# ax.scatter(df[:,4], df[:,3], label='DQN_Cycle_AHU', marker='o', color='violet')

ax.legend(loc='upper left', fontsize=fontsize)

ax.grid()
ax.set_axisbelow(True)

plt.title('Energy-Temperature', fontsize=20)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()










'''Temp_Humidity grid'''

T_H = np.array(T_H)

fontsize = 16

fig = plt.figure(figsize=(6,6), dpi=100)
ax = plt.subplot(111)


ax.set_ylim(30,80,1)
ax.set_xlim(0,100,1)
ax.set_xlabel('Outdoor Relative Humidity (%)', fontsize=fontsize)
ax.set_ylabel('Outdoor Temperature (F)', fontsize=fontsize)
ax.tick_params(labelsize=fontsize)

# ax.spines['left'].set_position('zero')
# # ax.spines['right'].set_color('none')
# ax.spines['bottom'].set_position('zero')
# # ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')


# plt.plot(df[:], label='Reward')
ax.axvline(0, zorder=0)
ax.axhline(0, zorder=0)

# ax.xaxis.set_major_locator(MultipleLocator(0.1))
# ax.yaxis.set_major_locator(MultipleLocator(0.1))

ax.scatter(T_H[0,1], T_H[0,0],  label='SC', marker='o', s=100)
ax.scatter(T_H[1,1], T_H[1,0],  label='AZ', marker='o', s=100)
ax.scatter(T_H[2,1], T_H[2,0],  label='CA', marker='o', s=100)
ax.scatter(T_H[3,1], T_H[3,0],  label='FL', marker='o', s=100)
ax.scatter(T_H[4,1], T_H[4,0],  label='MA', marker='o', s=100)
ax.scatter(T_H[5,1], T_H[5,0],  label='MN', marker='o', s=100)
ax.scatter(T_H[6,1], T_H[6,0],  label='TX', marker='o', s=100)


# ax.scatter(df_6zone[:,4], df_6zone[:,3],  label='DQN_6_Zone', marker='o', color='r')
# ax.scatter(df_1zone[:,4], df_1zone[:,3],  label='DQN_Single_Zone', marker='o', color='b')
# ax.scatter(df_mean[:,4], df_mean[:,3],  label='DQN_Mean_HVAC', marker='o', color='limegreen')
# ax.scatter(df[:,4], df[:,3], label='DQN_Cycle_AHU', marker='o', color='violet')

ax.legend(loc='lower left', fontsize=fontsize)

ax.grid()
ax.set_axisbelow(True)

plt.title('Temperature-Humidity', fontsize=20)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()






'''
Heat - Cool 
Area map
'''

df = np.load('./Benchmark.npy', allow_pickle=True)

# Benchmark[0, 11] = EPLUS.time_line
# Benchmark[0, 27] = EPLUS.work_time
# Benchmark[epoch, 28] = EPLUS.E_Heating
# Benchmark[epoch, 29] = EPLUS.E_Cooling



step = 1
fontsize = 16

fig = plt.figure(figsize=(10,5), dpi=100)
ax = plt.subplot(111)

# plt.figure(figsize=(10,6), dpi=100)

x = np.array(df[0,11][288:])
work_time = np.array(df[0,27][288:])
E_Heating = np.array(df[-1,28][288:])
E_Cooling = np.array(df[-1,29][288:])


# Find the indices where work_time is zero
zero_indices = np.where(work_time == 0)[0]

# Remove the corresponding elements from all arrays
x = np.delete(x, zero_indices)[::step]
work_time = np.delete(work_time, zero_indices)[::step]
E_Heating = np.delete(E_Heating, zero_indices)[::step]
E_Cooling = np.delete(E_Cooling, zero_indices)[::step]


x_new = []
work_time_new = []
E_Heating_new = [] 
E_Cooling_new = []

interval = 288*7

for i in range(0,len(x),interval):
    x_new.append((x[i]))
    work_time_new.append(np.mean(work_time[i:i+interval]))
    E_Heating_new.append(np.mean(E_Heating[i:i+interval]))
    E_Cooling_new.append(np.mean(E_Cooling[i:i+interval]))



x = np.array(x_new)
work_time = np.array(work_time_new)
E_Heating = np.array(E_Heating_new)
E_Cooling = np.array(E_Cooling_new)


E_total = E_Heating + E_Cooling

E_Heating = E_Heating/E_total * 100
E_Cooling = E_Cooling/E_total * 100

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

colors = ['tomato','cyan']
ax.stackplot(x, E_Heating, E_Cooling, colors=colors)
ax.legend(labels=['E_Heating','E_Cooling'], fontsize = fontsize)
ax.tick_params(labelsize=fontsize)
ax.set_xlabel('Date', fontsize=fontsize)
ax.set_ylabel('Heating-Cooling Energy Ratio (%)', fontsize=fontsize)


plt.title('Heating-Cooling Energy Comparison', fontsize=20)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()












'''
Commercial HVAC
'''

fontsize = 20

plt.figure(figsize=(10,10), dpi=100)
# plt.rcParams.update({'font.size': fontsize})

# Set the font name and weight
fontname = 'Arial'
fontweight = 'bold'
# plt.figure(figsize=(10,6), dpi=100)


y = np.array([33, 7, 12, 16, 32])

plt.pie(y,
        labels=['Space Heating','Cooling','Water Heating','Lighting','Others'], # 设置饼图标签
        colors=['tomato', 'cyan', 'teal', 'gold', 'gray'], # 设置饼图颜色
        explode=(0.1, 0.1, 0, 0, 0), # 第二部分突出显示，值越大，距离中心越远
        autopct='%d%%', # 格式化输出百分比
        textprops={'fontsize': fontsize}
        # 'weight': fontweight, 
       )
plt.title("Commercial Building Energy Consumption in 2010", fontsize=25,
          fontname=fontname, fontweight=fontweight)
# plt.show()


# plt.legend(fontsize = fontsize)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
















'''
Off VAV test
'''

df1 = np.load('./Benchmark_ONOFF_air_infiltration_off.npy', allow_pickle=True)


df2 = np.load('./Benchmark_VAV_2_OFF.npy', allow_pickle=True)[:5]


df3 = np.load('./Benchmark_VAV_2_5_OFF.npy', allow_pickle=True)[:3]


df4 = np.load('./Benchmark_VAV_2_5_6_OFF.npy', allow_pickle=True)[:2]


df5 = np.load('./Benchmark_VAV_2_5_ON.npy', allow_pickle=True)
df6 = np.load('./Benchmark_VAV_5_ON.npy', allow_pickle=True)
df7 = np.load('./Benchmark_VAV_ALL_OFF.npy', allow_pickle=True)


np.sum(df7[0,1])
np.sum(df7[0,2])

plt.plot(df7[0,11], df7[0,1])
plt.plot(df7[0,11], df7[-1,2])




work_time = df2[0,27]

work_time.count(0)
work_time.count(1)

ratio = work_time.count(1)/(work_time.count(0)+work_time.count(1))






'''E_T grid'''

E_T_1 = np.array(df1[:,3:5])
E_T_2 = np.array(df2[:,3:5])
E_T_3 = np.array(df3[:,3:5])
E_T_4 = np.array(df4[:,3:5])

E_T_5 = np.array(df5[:,3:5])
E_T_6 = np.array(df6[:,3:5])
E_T_7 = np.array(df7[:,3:5])



E_T_2[:,1] = E_T_2[:,1]/ratio
E_T_3[:,1] = E_T_3[:,1]/ratio
E_T_4[:,1] = E_T_4[:,1]/ratio

E_T_5[:,1] = E_T_5[:,1]
E_T_6[:,1] = E_T_6[:,1]
E_T_7[:,1] = E_T_7[:,1]


fontsize = 16

fig = plt.figure(figsize=(6,6), dpi=100)
ax = plt.subplot(111)


ax.set_ylim(-10,50,1)
ax.set_xlim(-5,10,1)
ax.set_xlabel('Temperature Violation (%)', fontsize=fontsize)
ax.set_ylabel('Energy Saving (%)', fontsize=fontsize)
ax.tick_params(labelsize=fontsize)

# ax.spines['left'].set_position('zero')
# # ax.spines['right'].set_color('none')
# ax.spines['bottom'].set_position('zero')
# # ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')


# plt.plot(df[:], label='Reward')
ax.axvline(0, zorder=0)
ax.axhline(0, zorder=0)

# ax.xaxis.set_major_locator(MultipleLocator(0.1))
# ax.yaxis.set_major_locator(MultipleLocator(0.1))

ax.scatter(E_T_1[:,1]*100, E_T_1[:,0]*100,  label='VAV_All', marker='o', s=100)
ax.scatter(E_T_2[:,1]*100, E_T_2[:,0]*100,  label='VAV_5_OFF', marker='o', s=100)
ax.scatter(E_T_3[:,1]*100, E_T_3[:,0]*100,  label='VAV_2_5_OFF', marker='o', s=100)
ax.scatter(E_T_4[:,1]*100, E_T_4[:,0]*100,  label='VAV_2_5_6_OFF', marker='o', s=100)
# ax.scatter(E_T[4,1]*100, E_T[4,0]*100,  label='MA', marker='o', s=100)
# ax.scatter(E_T[5,1]*100, E_T[5,0]*100,  label='MN', marker='o', s=100)
# ax.scatter(E_T[6,1]*100, E_T[6,0]*100,  label='TX', marker='o', s=100)



# ax.scatter(df_6zone[:,4], df_6zone[:,3],  label='DQN_6_Zone', marker='o', color='r')
# ax.scatter(df_1zone[:,4], df_1zone[:,3],  label='DQN_Single_Zone', marker='o', color='b')
# ax.scatter(df_mean[:,4], df_mean[:,3],  label='DQN_Mean_HVAC', marker='o', color='limegreen')
# ax.scatter(df[:,4], df[:,3], label='DQN_Cycle_AHU', marker='o', color='violet')

ax.legend(loc='upper left', fontsize=fontsize)

ax.grid()
ax.set_axisbelow(True)

plt.title('Energy-Temperature', fontsize=20)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()






'''
E-T timeline

'''


E_T = np.array([np.mean(E_T_1, axis=0),
               np.mean(E_T_2, axis=0),
               np.mean(E_T_3, axis=0),
               np.mean(E_T_4, axis=0),
               np.mean(E_T_5, axis=0),
               np.mean(E_T_6, axis=0),
               np.mean(E_T_7, axis=0)])

x = [6,5,4,3,2,1,0]
# x = list(reversed(x))

fig, ax1 = plt.subplots(figsize=(7,5), dpi=100)
fontsize = 16

ax1.set_xlabel('Number of VAV', fontsize=fontsize)
ax1.tick_params(labelsize=fontsize)

color='g'
E_T[:-1,0] += 0.05
p1, = ax1.plot(x, (E_T[:,0]*100), label='E_save', linewidth=3, color=color, marker='o', markersize=10)
ax1.set_ylabel('Energy SavingRatio %', color=color, fontsize=fontsize)
ax1.set_ylim(0,60)
ax1.tick_params(axis='y', labelcolor=color, labelsize=fontsize)


ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color='r'
p2, = ax2.plot(x, (E_T[:,1]*100), label='T_violation', linewidth=3, color=color, marker='s', markersize=10)
ax2.set_ylabel('Temperature Violation Ratio %', color=color, fontsize=fontsize)  # we already handled the x-label with ax1
ax2.set_ylim(0,60)
ax2.tick_params(axis='y', labelcolor=color, labelsize=fontsize)


ax1.set_xticks(x)  # set x-axis tick positions to 0-20
ax2.set_xticks(x)

ax1.set_title('Energy-Temperature to Number of VAV', fontsize=20)

ax1.grid()
ax1.set_axisbelow(True)

plt.gca().invert_xaxis()

fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig('./weather_data/'+str(weather_file[:-19])+'.png')
plt.show()













import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


# Define the data as a pandas DataFrame
data = pd.DataFrame({
    'Q1': [ 0. ,  0.5,  0. , 0. , 0. ],
    'Q2': [ 0.2,  0.4,  0. , 0. , 0. ],
    'Q3': [ 0.2,  0.3,  0. , -0.1, 0. ],
    'Q4': [ 0.2,  0.3,  0. , 0. , 0. ],
    'Q5': [ 0.2,  0.4,  0. , 0. , 0. ],
    # 'Q7': [0.2 , 0.45, 0.1 , 0.  , 0.  ],
    # 'Q8': [0.8 , 0.15, 0.  , 0.  , 0.  ],
    # 'Q9': [0.2 , 0.45, 0.1 , 0.  , 0.  ],
    # 'Q10':[0.  , 0.6 , 0.1 , 0.  , 0. ],
    
    'willingness': [ 0.6,  0.1,  0. , 0. , 0. ],


})

# Separate the independent and dependent variables
X = data.iloc[:, :-1]  # all columns except last one
y = data.iloc[:, -1]   # last column

# Add a constant term to the independent variables (for the intercept term)
X = sm.add_constant(X)

# Perform the multiple linear regression using Ordinary Least Squares (OLS)
model = sm.OLS(y, X).fit()

# Print the results summary
print(model.summary())


# create a LinearRegression object
reg = LinearRegression()

# fit the model
reg.fit(X, y)

# print intercept and coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)




# Define the data as a pandas DataFrame
data = pd.DataFrame({

    'Q7': [0.2 , 0.45, 0.1 , 0.  , 0.  ],
    'Q8': [0.8 , 0.15, 0.  , 0.  , 0.  ],
    'Q9': [0.2 , 0.45, 0.1 , 0.  , 0.  ],
    'Q10':[0.  , 0.6 , 0.1 , 0.  , 0. ],
    
    'willingness': [ 0.6,  0.1,  0. , 0. , 0. ],


})

# Separate the independent and dependent variables
X = data.iloc[:, :-1]  # all columns except last one
y = data.iloc[:, -1]   # last column

# Add a constant term to the independent variables (for the intercept term)
X = sm.add_constant(X)

# Perform the multiple linear regression using Ordinary Least Squares (OLS)
model = sm.OLS(y, X).fit()

# Print the results summary
print(model.summary())





np.array([0.6, 0.2, 0.2, 0, 0])*np.array([1,0.5,0,-0.5,-1])

np.array([0, 1, 0, 0, 0])*np.array([1,0.5,0,-0.5,-1])
np.array([0.2, 0.8, 0, 0, 0])*np.array([1,0.5,0,-0.5,-1])
np.array([0.2, 0.6, 0, 0.2, 0])*np.array([1,0.5,0,-0.5,-1])
np.array([0.2, 0.6, 0.2, 0, 0])*np.array([1,0.5,0,-0.5,-1])
np.array([0.2, 0.8, 0, 0, 0])*np.array([1,0.5,0,-0.5,-1])


np.array([0.2, 0.6, 0.2, 0, 0])*np.array([1,0.75,0.5,0.25,0])
np.array([0.8, 0.2, 0, 0, 0])*np.array([1,0.75,0.5,0.25,0])
np.array([0.2, 0.6, 0.2, 0, 0])*np.array([1,0.75,0.5,0.25,0])
np.array([0, 0.8, 0.2, 0, 0])*np.array([1,0.75,0.5,0.25,0])








noise = np.random.normal(loc=0, scale=0.05, size=data.shape)


data = data + np.abs(noise)





data = pd.DataFrame({
    'P1': [0.5, 0.5,-0.5, 0.5, 0.5,   0, 2, 4, 4, 4],
    'P2': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 3, 4, 4, 3],
    'P3': [0.5,   1, 0.5,   1, 0.5,   1, 4, 4, 3, 3],
    'P4': [0.5, 0.5, 0.5, 0.5, 0.5,   1, 3, 3, 3, 3],
    'P5': [0.5, 0.5,   1,   0,   1,   1, 3, 4, 3, 3],
    'P6': [ -1,-0.5,-0.5,  -1,  -1,-0.5, 2, 2, 2, 1],
    'P7': [  1,   1,   0,   1,   1, 0.5, 4, 4, 3, 3],
    'P8': [  0, 0.5,   0,   0,   0,   0, 3, 3, 2, 2],
    'P9': [0.5, 0.5,   0, 0.5, 0.5,   0, 2, 3, 3, 3],
    'P10':[  0,-0.5,   0,   0,   0, 0.5, 2, 2, 2, 2],
})    

# Convert the dictionary of lists into a NumPy array
data = data.values


print(data)






np.mean(data[6:], 0)




data = pd.DataFrame({
    'D1': [ 0.5 ,  0.5 ,  0.75,  0.5 ,  0.5 , -0.75,  1.  ,  0.25,  0.5 ,  -0.25],
    
    'D2': [ 0.16,  0.5 ,  0.66,  0.5 ,  0.66, -0.83,  0.66,  0.  ,  0.33,  0.   ],
    
    'D3': [ 0. ,  0.5,  1. ,  1. ,  1. , -0.5,  0.5,  0. ,  0. ,  0.5],
    
    'D4': [3.5 , 3.5 , 3.5 , 3.  , 3.25, 1.75, 3.5 , 2.5 , 2.75, 2.  ]
})    

# Convert the dictionary of lists into a NumPy array




data = pd.DataFrame({
    'D1': [ 0.5 ,  0.5 ,  0.75,  0.5 , -0.75,  1.  ,  0.25],
    
    'D2': [ 0.16,  0.5 ,  0.66,  0.66, -0.83,  0.66,  0.  ],
    
    'D3': [ 0. ,    0.5,   1. ,   1. ,  -0.5,   0.5,  0. ],
    
    'D4': [3.5 ,    3.5,  3.5 ,  3.25,  1.75,  3.5 ,  2.5 ]
})    



print(data)





# Separate the independent and dependent variables
X = data[['D1', 'D2', 'D4']] # all columns except last one
y = data['D3']   # last column

# Add a constant term to the independent variables (for the intercept term)
X = sm.add_constant(X)

# Perform the multiple linear regression using Ordinary Least Squares (OLS)
model = sm.OLS(y, X).fit()

# Print the results summary
print(model.summary())




import seaborn as sns

sns.regplot(x='D1', y='D3', data=data)



sns.pairplot(data, x_vars=['D1', 'D2', 'D4'], y_vars=['D3'], kind='reg')






import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(ncols=3, figsize=(20, 7))

sns.set(font_scale=2) # Set font size

sns.regplot(ax=axes[0], x='D1', y='D3', data=data, color='red')
axes[0].set(
    # title='Regression of D1 on D3', 
            xlabel='Expectation of AI', ylabel='Willingness to use AI')
axes[0].set_ylim([-2, 2])

sns.regplot(ax=axes[1], x='D2', y='D3', data=data, color='green')
axes[1].set(
    # title='Regression of D2 on D3', 
    xlabel='Team performance of AI', ylabel='Willingness to use AI')
axes[1].set_ylim([-2, 2])

sns.regplot(ax=axes[2], x='D4', y='D3', data=data, color='blue')
axes[2].set(
    # title='Regression of D4 on D3', 
    xlabel='AI Collaboration Feedbacks', ylabel='Willingness to use AI')
axes[2].set_ylim([-2, 2])

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()










df = np.load('./Benchmark/DQN.npy', allow_pickle=True)
































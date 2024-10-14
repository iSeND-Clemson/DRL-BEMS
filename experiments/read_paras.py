# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 12:20:00 2023

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



def save_parameters_to_txt(parameters, file_path):
    with open(file_path, 'w') as file:
        for key, value in parameters.items():
            file.write(f'{key}: {value}\n')
            print(f"{key}: {value}")
    # print(f'Parameters saved to {file_path}')


osm_name_box = 'ITRC_2nd_6zone_OPEN.osm'
timestep_per_hour = 12
begin_month = 1
begin_day_of_month =1
end_month = 12
end_day_of_month = 31
save_idf = 'test.idf'
AirWall_Switch = 'on'
Roof_Switch = 'off'
RL_flag = False

epochs = 10
lr = 1e-3
gamma = 0.9
epsilon = 0
target_update = 100
buffer_size = 10000
minimal_size = 200
batch_size = 128

FPS = 10000
signal_loss = False
signal_factor = 0.1
T_factor_day = 0.01 *5
E_factor_day = 1e-6
T_factor_night = 0
E_factor_night = 1e-6 *5
F_bottom = 0
F_top = 120

    
parameters = {
    'osm_name_box': osm_name_box,
    'timestep_per_hour': timestep_per_hour,
    'begin_month': begin_month,
    'begin_day_of_month': begin_day_of_month,
    'end_month' : end_month,
    'end_day_of_month' : end_day_of_month,
    'save_idf' : save_idf,
    'AirWall_Switch' : AirWall_Switch,
    'Roof_Switch' : Roof_Switch,
    'RL_flag' : RL_flag,
    
    'epochs' : epochs,
    'lr' : lr,
    'gamma' : gamma,
    'epsilon' : epsilon,
    'target_update' : target_update,
    'buffer_size' : buffer_size,
    'minimal_size' : minimal_size,
    'batch_size' : batch_size,
    
    'FPS' : FPS,
    'signal_loss' : signal_loss,
    'signal_factor' : signal_factor,
    'T_factor_day' : T_factor_day,
    'E_factor_day' : E_factor_day,
    'T_factor_night' : T_factor_night,
    'E_factor_night' : E_factor_night,
    'F_bottom' : F_bottom,
    'F_top' : F_top
    
}



save_parameters_to_txt(parameters, 'parameters.txt')
print('\n save parameters successful...\n')
print(parameters)








import eppy
from eppy import modeleditor
from eppy.modeleditor import IDF


# # building = idf1.idfobjects['BUILDING'][0]
# # hvac_cool_setpoint = idf1.idfobjects['Schedule:Day:Interval'][0]
# # hvac_heat_setpoint = idf1.idfobjects['Schedule:Day:Interval'][1]
# # hvac_cool_setpoint.Value_Until_Time_1

# # idf1.save()

# Building_Surfaces = idf1.idfobjects['BuildingSurface:Detailed']

# len(Building_Surfaces)

import random

class Building(object):
    
    '''
    Air Wall List:
        Face ['2','12','3','19','11','20','17','28','22','35','29','34','21','42','36','41','10','43']
        Surface Type: Wall
        Construction Name: AirWall
    '''

    def __init__(self, filename_to_run):
        iddfile = "Energy+.idd"
        IDF.setiddname(iddfile)    
        idf1 = IDF(filename_to_run)
        # print(idf1.idfobjects['BUILDING']) 
        
        self.idf = idf1
        
    def AirWall_Switch(self, switch):
        Building_Surfaces = self.idf.idfobjects['BuildingSurface:Detailed']

        if switch == 'on':
            for i in range(len(Building_Surfaces)): #print(Building_Surfaces[i])
                surface_i = Building_Surfaces[i]
                ''' AirWall '''
                if surface_i['Name'][5::] in ['2','12',
                                              '3','19',
                                              '11','20',
                                              '17','28',
                                              '22','35',
                                              '29','34',
                                              '21','42',
                                              '36','41',
                                              '10','43']:
                    surface_i['Construction_Name'] = 'AirWall'
                    # print(surface_i)
        return (print('AirWall set to True'))
    
    def Roof_Switch(self, switch):
        Building_Surfaces = self.idf.idfobjects['BuildingSurface:Detailed']

        if switch == 'off':
            for i in range(len(Building_Surfaces)): #print(Building_Surfaces[i])
                surface_i = Building_Surfaces[i]
                ''' Roof '''
                if surface_i['Surface_Type'] == 'Roof':
                    surface_i['Sun_Exposure'] = 'NoSun'
                    surface_i['Wind_Exposure'] = 'NoWind'
                    # print(surface_i)
        return (print('Roof set to False'))

    def init_map_2D(self, scale=10):
        self.scale = scale
        self.building_floor = []
        self.floor = []
        self.x_list = []
        self.y_list = []
        self.X = 0
        self.Y = 0
        
        self.zone_center_xy = []
        
        Building_Surfaces = self.idf.idfobjects['BuildingSurface:Detailed']

        for i in range(len(Building_Surfaces)): #print(Building_Surfaces[i])
            surface_i = Building_Surfaces[i]
            
            ''' Floor Map '''
            if surface_i['Surface_Type'] == 'Floor':
                self.building_floor.append([surface_i])

                x1, y1 = surface_i['Vertex_1_Xcoordinate'], surface_i['Vertex_1_Ycoordinate']
                x2, y2 = surface_i['Vertex_2_Xcoordinate'], surface_i['Vertex_2_Ycoordinate']
                x3, y3 = surface_i['Vertex_3_Xcoordinate'], surface_i['Vertex_3_Ycoordinate']
                x4, y4 = surface_i['Vertex_4_Xcoordinate'], surface_i['Vertex_4_Ycoordinate']
                # x5, y5 = surface_i['Vertex_5_Xcoordinate'], surface_i['Vertex_5_Ycoordinate']
                # x6, y6 = surface_i['Vertex_6_Xcoordinate'], surface_i['Vertex_6_Ycoordinate']

                # self.floor.append([[x1,x2,x3,x4], [y1,y2,y3,y4]])
                self.x_list.append([x1,x2,x3,x4])
                self.y_list.append([y1,y2,y3,y4])
                
                
                if surface_i['Vertex_6_Xcoordinate']:
                    x5, y5 = surface_i['Vertex_5_Xcoordinate'], surface_i['Vertex_5_Ycoordinate']
                    x6, y6 = surface_i['Vertex_6_Xcoordinate'], surface_i['Vertex_6_Ycoordinate']
    
                    self.x_list[-1].append(x5)
                    self.x_list[-1].append(x6)

                    self.y_list[-1].append(y5)
                    self.y_list[-1].append(y6)
                    
                    
                    
        for i in range(len(self.x_list)):
            self.x_list[i] = [np.min(self.x_list[i]), np.max(self.x_list[i])]
            self.y_list[i] = [np.min(self.y_list[i]), np.max(self.y_list[i])]

                    
        self.min_X, self.max_X = np.min(self.x_list), np.max(self.x_list) 
        self.min_Y, self.max_Y = np.min(self.y_list), np.max(self.y_list) 
        
        self.x_list = (np.array(self.x_list)-self.min_X) * self.scale
        self.y_list = (np.array(self.y_list)-self.min_Y) * self.scale

        for i in range(len(self.building_floor)):
            self.building_floor[i].append(self.x_list[i])
            self.building_floor[i].append(self.y_list[i])



        self.min_X, self.max_X = np.min(self.x_list), np.max(self.x_list) 
        self.min_Y, self.max_Y = np.min(self.y_list), np.max(self.y_list) 
        
        self.X = round(self.max_X-self.min_X) + 100 
        self.Y = round(self.max_Y-self.min_Y) + 100 
    
        '''Build grid'''
        self.map_2D = np.zeros((self.Y, self.X))
        for i in self.building_floor:
            temp = random.randint(0, 255)
            x1, x2 = np.int32([np.min(i[1]), np.max(i[1])])
            y1, y2 = np.int32([np.min(i[2]), np.max(i[2])])
            
            self.map_2D[y1:y2, x1:x2] = round(temp)
            
        # plt.figure(figsize=(10,10))
        # plt.imshow(self.map_2D)
        return self.map_2D
    

    def draw_map(self, temp):
        self.map_2D[:,:] = temp['Outdoor Temp']
        for i in self.building_floor:
            surface_i = i[0]
            if temp[surface_i['Zone_Name']]:
                zone_temp = temp[surface_i['Zone_Name']]
            else:
                zone_temp = 0

            x1, x2 = np.int32([np.min(i[1]), np.max(i[1])]) +50
            y1, y2 = np.int32([np.min(i[2]), np.max(i[2])]) +50
            
            self.map_2D[y1:y2, x1:x2] = round(zone_temp)
            self.map_2D[y1:y2, x1] = 0
            self.map_2D[y1:y2, x2] = 0
            self.map_2D[y1, x1:x2] = 0
            self.map_2D[y2, x1:x2] = 0
            
            self.zone_center_xy.append([(x1+x2)/2, self.Y-(y1+y2)/2])
            
        # plt.figure(figsize=(10,10))
        # plt.imshow(self.map_2D)
        self.map_2D[0,0] = 50
        self.map_2D[0,1] = 0
        self.map_2D[:,:] = self.map_2D[::-1,:]

        return self.map_2D, self.zone_center_xy

        


# temp = {}

# temp['Thermal Zone 1'] = 10
# temp['Thermal Zone 2'] = 15
# temp['Thermal Zone 3'] = 100
# temp['Thermal Zone 4'] = 25
# temp['Thermal Zone 5'] = 30
# temp['Thermal Zone 6'] = 35
# temp['Thermal Zone 7'] = 40


# map_2D = ITRC_2.draw_map(temp)

###############################################################################




filename_to_run = save_idf

iddfile = "Energy+.idd"
IDF.setiddname(iddfile)


IDF.getiddname()

idf1 = IDF(filename_to_run)

idf1.printidf()
# print(idf1.idfobjects['BUILDING']) # put the name of the object you'd like to look at in brackets



'''
Building Model
'''

filename_to_run = save_idf

ITRC_2 = Building(save_idf)
ITRC_2.idf.idfobjects['BuildingSurface:Detailed']
ITRC_2.AirWall_Switch(AirWall_Switch)
ITRC_2.Roof_Switch(Roof_Switch)
ITRC_2.idf.saveas(save_idf)

THERMAL_MAP_2D = ITRC_2.init_map_2D()
building_floor = ITRC_2.building_floor


print('save to:', save_idf)














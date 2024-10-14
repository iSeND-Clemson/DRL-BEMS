# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 16:29:04 2022

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


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transformtransforms

from tqdm import tqdm
from torchvision import models
from torchsummary import summary
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage

print('torch.version: ',torch. __version__)
print('torch.version.cuda: ',torch.version.cuda)
print('torch.cuda.is_available: ',torch.cuda.is_available())
print('torch.cuda.device_count: ',torch.cuda.device_count())
print('torch.cuda.current_device: ',torch.cuda.current_device())
device_default = torch.cuda.current_device()
torch.cuda.device(device_default)
print('torch.cuda.get_device_name: ',torch.cuda.get_device_name(device_default))
device = torch.device("cuda")

    




# Use OpenStudio to create a Model
import openstudio
'''
For reproducibility, here are the versions I used to create and run this notebook
_s = !pip list
print(f"Pip package used initially: {[x for x in _s if 'openstudio' in x][0]}")
print(f"OpenStudio Long Version:    {openstudio.openStudioLongVersion()}")    
'''

# Run EnergyPlus in API mode
# insert the repo build tree or install path into the search Path, then import the EnergyPlus API
import sys
sys.path.insert(0, './openstudioapplication-1.4.0/EnergyPlus')


# from pyenergyplus import api
from pyenergyplus.api import EnergyPlusAPI
import pyenergyplus
pyenergyplus.api.EnergyPlusAPI.api_version()


###############################################################################

def temp_c_to_f(temp_c):#: float, arbitrary_arg=None):
    """Convert temp from C to F. Test function with arbitrary argument, for example."""
    # x = arbitrary_arg
    return 1.8 * temp_c + 32

def temp_f_to_c(temp_f):
    return (temp_f-32)/1.8




###############################################################################


'''
Openstudio setup
define date, save model

'''
# test_name = "Room Air Zone Vertical Temperature Gradient"
# osm_name_box = 'bmw_f2.osm'
# osm_name_box = 'test_new.osm'
osm_name_box = 'ITRC_2nd_6zone_OPEN.osm'

current_dir = os.getcwd()
osm_path = os.path.join(current_dir,osm_name_box)
osm_path = openstudio.path(osm_path) # I guess this is how it wants the path for the translator
print(osm_path)


translator = openstudio.osversion.VersionTranslator()
osm = translator.loadModel(osm_path).get()

# Create an example model
# m = openstudio.model.exampleModel()
m = translator.loadModel(osm_path).get()

zones = [zone for zone in openstudio.model.getThermalZones(m)]



# Set output variables
[x.remove() for x in m.getOutputVariables()]

o = openstudio.model.OutputVariable("Site Outdoor Air Drybulb Temperature", m)
o.setKeyValue("Environment")
o.setReportingFrequency("Timestep")


for var in ["Site Outdoor Air Drybulb Temperature",
            "Site Wind Speed",
            "Site Wind Direction",
            "Site Solar Azimuth Angle",
            "Site Solar Altitude Angle",
            "Site Solar Hour Angle"]:
    o = openstudio.model.OutputVariable(var, m)
    o.setKeyValue('Environment')
    o.setReportingFrequency("Timestep")

# o = openstudio.model.OutputVariable("Zone Thermal Comfort Fanger Model PPD", m)
# o.setKeyValue('*')
# o.setReportingFrequency("Timestep")





for var in ["Zone Air Relative Humidity",
            "Zone Windows Total Heat Gain Energy",
            "Zone Infiltration Mass",
            "Zone Mechanical Ventilation Mass",
            "Zone Mechanical Ventilation Mass Flow Rate",
            "Zone Air Temperature",
            "Zone Mean Radiant Temperature",
            "Zone Thermostat Heating Setpoint Temperature",
            "Zone Thermostat Cooling Setpoint Temperature"]:
    o = openstudio.model.OutputVariable(var, m)
    o.setKeyValue('Thermal Zone 1')
    o.setReportingFrequency("Timestep")
    
for var in ["Zone Air Relative Humidity",
            "Zone Windows Total Heat Gain Energy",
            "Zone Infiltration Mass",
            "Zone Mechanical Ventilation Mass",
            "Zone Mechanical Ventilation Mass Flow Rate",
            "Zone Air Temperature",
            "Zone Mean Radiant Temperature",
            "Zone Thermostat Heating Setpoint Temperature",
            "Zone Thermostat Cooling Setpoint Temperature"]:
    o = openstudio.model.OutputVariable(var, m)
    o.setKeyValue('Thermal Zone 2')
    o.setReportingFrequency("Timestep")
    
for var in ["Zone Air Relative Humidity",
            "Zone Windows Total Heat Gain Energy",
            "Zone Infiltration Mass",
            "Zone Mechanical Ventilation Mass",
            "Zone Mechanical Ventilation Mass Flow Rate",
            "Zone Air Temperature",
            "Zone Mean Radiant Temperature",
            "Zone Thermostat Heating Setpoint Temperature",
            "Zone Thermostat Cooling Setpoint Temperature"]:
    o = openstudio.model.OutputVariable(var, m)
    o.setKeyValue('Thermal Zone 3')
    o.setReportingFrequency("Timestep")
    
for var in ["Zone Air Relative Humidity",
            "Zone Windows Total Heat Gain Energy",
            "Zone Infiltration Mass",
            "Zone Mechanical Ventilation Mass",
            "Zone Mechanical Ventilation Mass Flow Rate",
            "Zone Air Temperature",
            "Zone Mean Radiant Temperature",
            "Zone Thermostat Heating Setpoint Temperature",
            "Zone Thermostat Cooling Setpoint Temperature"]:
    o = openstudio.model.OutputVariable(var, m)
    o.setKeyValue('Thermal Zone 4')
    o.setReportingFrequency("Timestep")
    
for var in ["Zone Air Relative Humidity",
            "Zone Windows Total Heat Gain Energy",
            "Zone Infiltration Mass",
            "Zone Mechanical Ventilation Mass",
            "Zone Mechanical Ventilation Mass Flow Rate",
            "Zone Air Temperature",
            "Zone Mean Radiant Temperature",
            "Zone Thermostat Heating Setpoint Temperature",
            "Zone Thermostat Cooling Setpoint Temperature"]:
    o = openstudio.model.OutputVariable(var, m)
    o.setKeyValue('Thermal Zone 5')
    o.setReportingFrequency("Timestep")
    
for var in ["Zone Air Relative Humidity",
            "Zone Windows Total Heat Gain Energy",
            "Zone Infiltration Mass",
            "Zone Mechanical Ventilation Mass",
            "Zone Mechanical Ventilation Mass Flow Rate",
            "Zone Air Temperature",
            "Zone Mean Radiant Temperature",
            "Zone Thermostat Heating Setpoint Temperature",
            "Zone Thermostat Cooling Setpoint Temperature"]:
    o = openstudio.model.OutputVariable(var, m)
    o.setKeyValue('Thermal Zone 6')
    o.setReportingFrequency("Timestep")
    
# for var in ["Zone Air Relative Humidity",
#             "Zone Windows Total Heat Gain Energy",
#             "Zone Infiltration Mass",
#             "Zone Mechanical Ventilation Mass",
#             "Zone Mechanical Ventilation Mass Flow Rate",
#             "Zone Air Temperature",
#             "Zone Mean Radiant Temperature",
#             "Zone Thermostat Heating Setpoint Temperature",
#             "Zone Thermostat Cooling Setpoint Temperature"]:
#     o = openstudio.model.OutputVariable(var, m)
#     o.setKeyValue('Thermal Zone 7')
#     o.setReportingFrequency("Timestep")
    
      

# Set timestep
timestep = m.getTimestep()
timestep.setNumberOfTimestepsPerHour(12)

# Check the heating thermostat schedule
# z = m.getThermalZones()[2]
# print(z)
# t = z.thermostatSetpointDualSetpoint().get()
# heating_sch = t.heatingSetpointTemperatureSchedule().get()
# o = heating_sch.to_ScheduleRuleset()


# Restrict to one month of simulation
r = m.getRunPeriod()
# print(r)

r.setBeginMonth(1)
r.setBeginDayOfMonth(1)

r.setEndMonth(12)
r.setEndDayOfMonth(31)


ft = openstudio.energyplus.ForwardTranslator()
w = ft.translateModel(m)
w.save(openstudio.path('test.idf'), True)



###############################################################################

import eppy
from eppy import modeleditor
from eppy.modeleditor import IDF

filename_to_run = "test.idf"

iddfile = "Energy+.idd"
IDF.setiddname(iddfile)


IDF.getiddname()

idf1 = IDF(filename_to_run)

idf1.printidf()
print(idf1.idfobjects['BUILDING']) # put the name of the object you'd like to look at in brackets


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
        # plt.show()
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
            self.map_2D[y1:y2, x1] = 20
            self.map_2D[y1:y2, x2] = 20
            self.map_2D[y1, x1:x2] = 20
            self.map_2D[y2, x1:x2] = 20
            
            self.zone_center_xy.append([(x1+x2)/2, self.Y-(y1+y2)/2])
            
        # plt.figure(figsize=(10,10))
        # plt.imshow(self.map_2D)
        self.map_2D[0,0] = 110
        self.map_2D[0,1] = 20
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
class Data_Bank():
    
    def __init__(self):
        self.view_distance  = 2000
        self.NUM_HVAC       = 7
        self.FPS            = 144
        
        self.E_factor       = 0
        self.T_factor       = 0
        
        self.episode_reward = 0
        self.episode_return = 0

        self.RL_flag        = True
        self.time_interval  = 0
        self.time_line      = []
        self.T_Violation    = []
        self.score          = []
        
        self.H_Violation    = []
        self.Humd_mean      = []
        self.signal         = []

        
        self.T_diff         = []
        self.T_maen         = []
        self.T_var          = []
        
        self.T_map          = {}

        
        ''' handles '''
        self.got_handles           = False

        self.oa_temp_handle        = -1
        self.oa_humd_handle        = -1
        self.oa_windspeed_handle   = -1
        self.oa_winddirct_handle   = -1
        self.oa_solar_azi_handle   = -1
        self.oa_solar_alt_handle   = -1
        self.oa_solar_ang_handle   = -1
        
        # self.oa_temp_handle        = -1
        self.zone_temp_handle      = -1
        self.zone_htg_tstat_handle = -1
        self.zone_clg_tstat_handle = -1
        
        self.zone_humd_handle_2001 = -1
        self.zone_humd_handle_2002 = -1
        self.zone_humd_handle_2003 = -1
        self.zone_humd_handle_2004 = -1
        self.zone_humd_handle_2005 = -1
        self.zone_humd_handle_2006 = -1
        # self.zone_humd_handle_2007 = -1
        
        self.zone_window_handle_2001 = -1
        self.zone_window_handle_2002 = -1
        self.zone_window_handle_2003 = -1
        self.zone_window_handle_2004 = -1
        self.zone_window_handle_2005 = -1
        self.zone_window_handle_2006 = -1
        # self.zone_window_handle_2007 = -1
        
        self.zone_ventmass_handle_2001 = -1
        self.zone_ventmass_handle_2002 = -1
        self.zone_ventmass_handle_2003 = -1
        self.zone_ventmass_handle_2004 = -1
        self.zone_ventmass_handle_2005 = -1
        self.zone_ventmass_handle_2006 = -1
        # self.zone_ventmass_handle_2007 = -1
        
        self.zone_temp_handle_2001 = -1
        self.zone_temp_handle_2002 = -1
        self.zone_temp_handle_2003 = -1
        self.zone_temp_handle_2004 = -1
        self.zone_temp_handle_2005 = -1
        self.zone_temp_handle_2006 = -1
        # self.zone_temp_handle_2007 = -1
        
        # self.zone_PPD_handle_2003 = -1
        
        self.hvac_htg_2001_handle  = -1
        self.hvac_clg_2001_handle  = -1
        self.hvac_htg_2002_handle  = -1
        self.hvac_clg_2002_handle  = -1
        self.hvac_htg_2003_handle  = -1
        self.hvac_clg_2003_handle  = -1
        self.hvac_htg_2004_handle  = -1
        self.hvac_clg_2004_handle  = -1
        self.hvac_htg_2005_handle  = -1
        self.hvac_clg_2005_handle  = -1
        self.hvac_htg_2006_handle  = -1
        self.hvac_clg_2006_handle  = -1
        # self.hvac_htg_2007_handle  = -1
        # self.hvac_clg_2007_handle  = -1
        
        self.E_Facility_handle     = -1
        self.E_HVAC_handle         = -1
        self.E_Heating_handle      = -1
        self.E_Cooling_handle      = -1
        

        ''' time '''
        self.x = []
        
        self.years = []
        self.months = []
        self.days = []
        self.hours = []
        self.minutes = []
        self.current_times = []
        self.actual_date_times = []
        self.actual_times = []
        
        self.weekday = []
        self.isweekday = []
        self.isweekend = []
        self.work_time = []
        
        ''' building parameters '''
        # self.oa_temp_handle        = -1
        self.y_humd = []
        self.y_wind = []
        self.y_solar = []
        
        self.y_zone_humd = []
        self.y_zone_window = []
        self.y_zone_ventmass = []
        
        self.y_zone_temp = []
        
        self.y_outdoor = []
        self.y_zone = []
        self.y_htg = []
        self.y_clg = []
        
        self.y_zone_temp_2001 = []
        self.y_zone_temp_2002 = []
        self.y_zone_temp_2003 = []
        self.y_zone_temp_2004 = []
        self.y_zone_temp_2005 = []
        self.y_zone_temp_2006 = []
        # self.y_zone_temp_2007 = []
        
        self.y_zone_humd_2001 = []
        self.y_zone_humd_2002 = []
        self.y_zone_humd_2003 = []
        self.y_zone_humd_2004 = []
        self.y_zone_humd_2005 = []
        self.y_zone_humd_2006 = []
        
        self.sun_is_up = []
        self.is_raining = []
        self.outdoor_humidity = []
        self.wind_speed = []
        self.diffuse_solar = []
        
        self.E_Facility = []
        self.E_HVAC = []
        self.E_Heating = []
        self.E_Cooling = []
        
        self.E_HVAC_all = []
        
        ''' DQN '''
        self.action_list = []
        self.action_map  = []
        self.episode_reward = []
        
        self.hvac_htg_2001 = []
        self.hvac_clg_2001 = []
        self.hvac_htg_2002 = []
        self.hvac_clg_2002 = []
        self.hvac_htg_2003 = []
        self.hvac_clg_2003 = []
        self.hvac_htg_2004 = []
        self.hvac_clg_2004 = []
        self.hvac_htg_2005 = []
        self.hvac_clg_2005 = []
        self.hvac_htg_2006 = []
        self.hvac_clg_2006 = []
        # self.hvac_htg_2007 = []
        # self.hvac_clg_2007 = []
        return
    
    def handle_availability(self):
        ''' check handle_availability '''
        self.handle_list = [self.oa_humd_handle,
                            self.oa_windspeed_handle,
                            self.oa_winddirct_handle,
                            self.oa_solar_azi_handle,
                            self.oa_solar_alt_handle,
                            self.oa_solar_ang_handle,
                            self.oa_temp_handle,
                            self.zone_temp_handle,
                            # self.zone_htg_tstat_handle,
                            # self.zone_clg_tstat_handle,
                            self.zone_temp_handle_2001, self.zone_temp_handle_2002,
                            self.zone_temp_handle_2003, self.zone_temp_handle_2004,
                            self.zone_temp_handle_2005, self.zone_temp_handle_2006,
                            # self.zone_temp_handle_2007,
                            # self.zone_PPD_handle_2003,
                            self.hvac_htg_2001_handle, self.hvac_clg_2001_handle,
                            self.hvac_htg_2002_handle, self.hvac_clg_2002_handle,
                            self.hvac_htg_2003_handle, self.hvac_clg_2003_handle,
                            self.hvac_htg_2004_handle, self.hvac_clg_2004_handle,
                            self.hvac_htg_2005_handle, self.hvac_clg_2005_handle,
                            self.hvac_htg_2006_handle, self.hvac_clg_2006_handle,
                            # self.hvac_htg_2007_handle, self.hvac_clg_2007_handle,
                            self.E_Facility_handle,
                            self.E_HVAC_handle,
                            self.E_Heating_handle,
                            self.E_Cooling_handle]
        return self.handle_list
        
        


# "Zone Air Relative Humidity"
# "Zone Windows Total Heat Gain Energy"
# "Zone Infiltration Mass"
# "Zone Mechanical Ventilation Mass"
# "Zone Mechanical Ventilation Mass Flow Rate"
# "Zone Air Temperature"
# "Zone Mean Radiant Temperature"
# "Zone Thermostat Heating Setpoint Temperature"
# "Zone Thermostat Cooling Setpoint Temperature"

# "Site Outdoor Air Drybulb Temperature"
# "Site Wind Speed"
# "Site Wind Direction"
# "Site Solar Azimuth Angle"
# "Site Solar Altitude Angle"
# "Site Solar Hour Angle"





###############################################################################




HVAC_action_list = []
for HC_1 in [0,1]:
    for HC_2 in [0,1]:
        for HC_3 in [0,1]:
            for HC_4 in [0,1]:
                for HC_5 in [0,1]:
                    for HC_6 in [0,1]:
                        HVAC_action_list.append([HC_1,HC_2,HC_3,HC_4,HC_5,HC_6])
    
    
def HVAC_action(action, temp):
    action = int(action)

    if action == 0:
        H_new = 60
        C_new = 90
        
    elif action == 1:
        H_new = temp[0]
        C_new = temp[1]
        
    else:
        H_new = 60
        C_new = 90
        
    return int(H_new), int(C_new)




def callback_function_DQN(state_argument):
    RL_flag = EPLUS.RL_flag
    view_distance = EPLUS.view_distance
    time_interval = EPLUS.time_interval
    NUM_HVAC = EPLUS.NUM_HVAC
    FPS = EPLUS.FPS
    
    
    '''
    Read data
    '''
    if not EPLUS.got_handles:
        if not api.exchange.api_data_fully_ready(state_argument):
            return
        
        EPLUS.oa_temp_handle        = api.exchange.get_variable_handle(state_argument, u"SITE OUTDOOR AIR DRYBULB TEMPERATURE", u"ENVIRONMENT")
        EPLUS.oa_humd_handle        = api.exchange.get_variable_handle(state_argument, u"Site Outdoor Air Drybulb Temperature", u"ENVIRONMENT")
        EPLUS.oa_windspeed_handle   = api.exchange.get_variable_handle(state_argument, u"Site Wind Speed", u"ENVIRONMENT")
        EPLUS.oa_winddirct_handle   = api.exchange.get_variable_handle(state_argument, u"Site Wind Direction", u"ENVIRONMENT")
        EPLUS.oa_solar_azi_handle   = api.exchange.get_variable_handle(state_argument, u"Site Solar Azimuth Angle", u"ENVIRONMENT")
        EPLUS.oa_solar_alt_handle   = api.exchange.get_variable_handle(state_argument, u"Site Solar Altitude Angle", u"ENVIRONMENT")
        EPLUS.oa_solar_ang_handle   = api.exchange.get_variable_handle(state_argument, u"Site Solar Hour Angle", u"ENVIRONMENT")
        
        EPLUS.zone_temp_handle      = api.exchange.get_variable_handle(state_argument, "Zone Air Temperature", 'Thermal Zone 2')
        # EPLUS.zone_htg_tstat_handle = api.exchange.get_variable_handle(state_argument, "Zone Thermostat Heating Setpoint Temperature", 'Thermal Zone 7')
        # EPLUS.zone_clg_tstat_handle = api.exchange.get_variable_handle(state_argument, "Zone Thermostat Cooling Setpoint Temperature", 'Thermal Zone 7')
        # EPLUS.zone_PPD_handle_2003 = api.exchange.get_variable_handle(state_argument, "Zone Thermal Comfort Fanger Model PPD", '*')
        
        EPLUS.zone_humd_handle_2001 = api.exchange.get_variable_handle(state_argument, "Zone Air Relative Humidity", 'Thermal Zone 1')
        EPLUS.zone_humd_handle_2002 = api.exchange.get_variable_handle(state_argument, "Zone Air Relative Humidity", 'Thermal Zone 2')
        EPLUS.zone_humd_handle_2003 = api.exchange.get_variable_handle(state_argument, "Zone Air Relative Humidity", 'Thermal Zone 3')
        EPLUS.zone_humd_handle_2004 = api.exchange.get_variable_handle(state_argument, "Zone Air Relative Humidity", 'Thermal Zone 4')
        EPLUS.zone_humd_handle_2005 = api.exchange.get_variable_handle(state_argument, "Zone Air Relative Humidity", 'Thermal Zone 5')
        EPLUS.zone_humd_handle_2006 = api.exchange.get_variable_handle(state_argument, "Zone Air Relative Humidity", 'Thermal Zone 6')
        # EPLUS.zone_humd_handle_2007 = api.exchange.get_variable_handle(state_argument, "Zone Air Relative Humidity", 'Thermal Zone 7')
        
        # EPLUS.zone_window_handle_2001 = api.exchange.get_variable_handle(state_argument, "Zone Windows Total Heat Gain Energy", 'Thermal Zone 1')
        # EPLUS.zone_window_handle_2002 = api.exchange.get_variable_handle(state_argument, "Zone Windows Total Heat Gain Energy", 'Thermal Zone 2')
        # EPLUS.zone_window_handle_2003 = api.exchange.get_variable_handle(state_argument, "Zone Windows Total Heat Gain Energy", 'Thermal Zone 3')
        # EPLUS.zone_window_handle_2004 = api.exchange.get_variable_handle(state_argument, "Zone Windows Total Heat Gain Energy", 'Thermal Zone 4')
        # EPLUS.zone_window_handle_2005 = api.exchange.get_variable_handle(state_argument, "Zone Windows Total Heat Gain Energy", 'Thermal Zone 5')
        # EPLUS.zone_window_handle_2006 = api.exchange.get_variable_handle(state_argument, "Zone Windows Total Heat Gain Energy", 'Thermal Zone 6')
        # EPLUS.zone_window_handle_2007 = api.exchange.get_variable_handle(state_argument, "Zone Windows Total Heat Gain Energy", 'Thermal Zone 7')
        
        # EPLUS.zone_ventmass_handle_2001 = api.exchange.get_variable_handle(state_argument, "Zone Mechanical Ventilation Mass", 'Thermal Zone 1')
        # EPLUS.zone_ventmass_handle_2002 = api.exchange.get_variable_handle(state_argument, "Zone Mechanical Ventilation Mass", 'Thermal Zone 2')
        # EPLUS.zone_ventmass_handle_2003 = api.exchange.get_variable_handle(state_argument, "Zone Mechanical Ventilation Mass", 'Thermal Zone 3')
        # EPLUS.zone_ventmass_handle_2004 = api.exchange.get_variable_handle(state_argument, "Zone Mechanical Ventilation Mass", 'Thermal Zone 4')
        # EPLUS.zone_ventmass_handle_2005 = api.exchange.get_variable_handle(state_argument, "Zone Mechanical Ventilation Mass", 'Thermal Zone 5')
        # EPLUS.zone_ventmass_handle_2006 = api.exchange.get_variable_handle(state_argument, "Zone Mechanical Ventilation Mass", 'Thermal Zone 6')
        # EPLUS.zone_ventmass_handle_2007 = api.exchange.get_variable_handle(state_argument, "Zone Mechanical Ventilation Mass", 'Thermal Zone 7')

        EPLUS.zone_temp_handle_2001 = api.exchange.get_variable_handle(state_argument, "Zone Air Temperature", 'Thermal Zone 1')
        EPLUS.zone_temp_handle_2002 = api.exchange.get_variable_handle(state_argument, "Zone Air Temperature", 'Thermal Zone 2')
        EPLUS.zone_temp_handle_2003 = api.exchange.get_variable_handle(state_argument, "Zone Air Temperature", 'Thermal Zone 3')
        EPLUS.zone_temp_handle_2004 = api.exchange.get_variable_handle(state_argument, "Zone Air Temperature", 'Thermal Zone 4')
        EPLUS.zone_temp_handle_2005 = api.exchange.get_variable_handle(state_argument, "Zone Air Temperature", 'Thermal Zone 5')
        EPLUS.zone_temp_handle_2006 = api.exchange.get_variable_handle(state_argument, "Zone Air Temperature", 'Thermal Zone 6')
        # EPLUS.zone_temp_handle_2007 = api.exchange.get_variable_handle(state_argument, "Zone Air Temperature", 'Thermal Zone 7')
        
        EPLUS.hvac_htg_2001_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control', 'Heating Setpoint', 'Thermal Zone 1')
        EPLUS.hvac_clg_2001_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control', 'Cooling Setpoint', 'Thermal Zone 1')
        EPLUS.hvac_htg_2002_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control', 'Heating Setpoint', 'Thermal Zone 2')
        EPLUS.hvac_clg_2002_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control', 'Cooling Setpoint', 'Thermal Zone 2')
        EPLUS.hvac_htg_2003_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control', 'Heating Setpoint', 'Thermal Zone 3')
        EPLUS.hvac_clg_2003_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control', 'Cooling Setpoint', 'Thermal Zone 3')
        EPLUS.hvac_htg_2004_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control', 'Heating Setpoint', 'Thermal Zone 4')
        EPLUS.hvac_clg_2004_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control', 'Cooling Setpoint', 'Thermal Zone 4')
        EPLUS.hvac_htg_2005_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control', 'Heating Setpoint', 'Thermal Zone 5')
        EPLUS.hvac_clg_2005_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control', 'Cooling Setpoint', 'Thermal Zone 5')
        EPLUS.hvac_htg_2006_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control', 'Heating Setpoint', 'Thermal Zone 6')
        EPLUS.hvac_clg_2006_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control', 'Cooling Setpoint', 'Thermal Zone 6')
        # EPLUS.hvac_htg_2007_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control', 'Heating Setpoint', 'Thermal Zone 7')
        # EPLUS.hvac_clg_2007_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control', 'Cooling Setpoint', 'Thermal Zone 7')
        
        EPLUS.E_Facility_handle = api.exchange.get_meter_handle(state_argument, 'Electricity:Facility')
        EPLUS.E_HVAC_handle     = api.exchange.get_meter_handle(state_argument, 'Electricity:HVAC')
        EPLUS.E_Heating_handle  = api.exchange.get_meter_handle(state_argument, 'Heating:Electricity')
        EPLUS.E_Cooling_handle  = api.exchange.get_meter_handle(state_argument, 'Cooling:Electricity')
        
        handle_list = EPLUS.handle_availability()
        if -1 in handle_list:
            print("***Invalid handles, check spelling and sensor/actuator availability")
            sys.exit(1)
        EPLUS.got_handles = True
    if api.exchange.warmup_flag(state_argument):
        return
    
    ''' Time '''
    year = api.exchange.year(state_argument)
    month = api.exchange.month(state_argument)
    day = api.exchange.day_of_month(state_argument)
    hour = api.exchange.hour(state_argument)
    minute = api.exchange.minutes(state_argument)
    current_time = api.exchange.current_time(state_argument)
    actual_date_time = api.exchange.actual_date_time(state_argument)
    actual_time = api.exchange.actual_time(state_argument)
    time_step = api.exchange.zone_time_step_number(state_argument)
    
    '''Temperature'''
    
    oa_humd      = api.exchange.get_variable_value(state_argument, EPLUS.oa_humd_handle)
    # oa_windspeed = api.exchange.get_variable_value(state_argument, EPLUS.oa_windspeed_handle)
    # oa_winddirct = api.exchange.get_variable_value(state_argument, EPLUS.oa_winddirct_handle)
    # oa_solar_azi = api.exchange.get_variable_value(state_argument, EPLUS.oa_solar_azi_handle)
    # oa_solar_alt = api.exchange.get_variable_value(state_argument, EPLUS.oa_solar_alt_handle)
    # oa_solar_ang = api.exchange.get_variable_value(state_argument, EPLUS.oa_solar_ang_handle)
        
    oa_temp = api.exchange.get_variable_value(state_argument,        EPLUS.oa_temp_handle)
    # zone_temp = api.exchange.get_variable_value(state_argument,      EPLUS.zone_temp_handle)
    # zone_htg_tstat = api.exchange.get_variable_value(state_argument, EPLUS.zone_htg_tstat_handle)
    # zone_clg_tstat = api.exchange.get_variable_value(state_argument, EPLUS.zone_clg_tstat_handle)
    
    zone_temp_2001 = api.exchange.get_variable_value(state_argument, EPLUS.zone_temp_handle_2001)
    zone_temp_2002 = api.exchange.get_variable_value(state_argument, EPLUS.zone_temp_handle_2002)
    zone_temp_2003 = api.exchange.get_variable_value(state_argument, EPLUS.zone_temp_handle_2003)
    zone_temp_2004 = api.exchange.get_variable_value(state_argument, EPLUS.zone_temp_handle_2004)
    zone_temp_2005 = api.exchange.get_variable_value(state_argument, EPLUS.zone_temp_handle_2005)
    zone_temp_2006 = api.exchange.get_variable_value(state_argument, EPLUS.zone_temp_handle_2006)
    # zone_temp_2007 = api.exchange.get_variable_value(state_argument, EPLUS.zone_temp_handle_2007)
    
    hvac_htg_2001 =  api.exchange.get_actuator_value(state_argument, EPLUS.hvac_htg_2001_handle)
    hvac_clg_2001 =  api.exchange.get_actuator_value(state_argument, EPLUS.hvac_clg_2001_handle)
    hvac_htg_2002 =  api.exchange.get_actuator_value(state_argument, EPLUS.hvac_htg_2002_handle)
    hvac_clg_2002 =  api.exchange.get_actuator_value(state_argument, EPLUS.hvac_clg_2002_handle)

    hvac_htg_2003 =  api.exchange.get_actuator_value(state_argument, EPLUS.hvac_htg_2003_handle)
    hvac_clg_2003 =  api.exchange.get_actuator_value(state_argument, EPLUS.hvac_clg_2003_handle)

    hvac_htg_2004 =  api.exchange.get_actuator_value(state_argument, EPLUS.hvac_htg_2004_handle)
    hvac_clg_2004 =  api.exchange.get_actuator_value(state_argument, EPLUS.hvac_clg_2004_handle)
    hvac_htg_2005 =  api.exchange.get_actuator_value(state_argument, EPLUS.hvac_htg_2005_handle)
    hvac_clg_2005 =  api.exchange.get_actuator_value(state_argument, EPLUS.hvac_clg_2005_handle)
    hvac_htg_2006 =  api.exchange.get_actuator_value(state_argument, EPLUS.hvac_htg_2006_handle)
    hvac_clg_2006 =  api.exchange.get_actuator_value(state_argument, EPLUS.hvac_clg_2006_handle)
    
    # zone_PPD_2003 = api.exchange.get_variable_value(state_argument, EPLUS.zone_PPD_handle_2003)
    # print(zone_PPD_2003)
    
    zone_humd_2001 = api.exchange.get_variable_value(state_argument, EPLUS.zone_humd_handle_2001)
    zone_humd_2002 = api.exchange.get_variable_value(state_argument, EPLUS.zone_humd_handle_2002)
    zone_humd_2003 = api.exchange.get_variable_value(state_argument, EPLUS.zone_humd_handle_2003)
    zone_humd_2004 = api.exchange.get_variable_value(state_argument, EPLUS.zone_humd_handle_2004)
    zone_humd_2005 = api.exchange.get_variable_value(state_argument, EPLUS.zone_humd_handle_2005)
    zone_humd_2006 = api.exchange.get_variable_value(state_argument, EPLUS.zone_humd_handle_2006)
    # zone_humd_2007 = api.exchange.get_variable_value(state_argument, EPLUS.zone_humd_handle_2007)

    # zone_window_2001 = api.exchange.get_variable_value(state_argument, EPLUS.zone_window_handle_2001)
    # zone_window_2002 = api.exchange.get_variable_value(state_argument, EPLUS.zone_window_handle_2002)
    # zone_window_2003 = api.exchange.get_variable_value(state_argument, EPLUS.zone_window_handle_2003)
    # zone_window_2004 = api.exchange.get_variable_value(state_argument, EPLUS.zone_window_handle_2004)
    # zone_window_2005 = api.exchange.get_variable_value(state_argument, EPLUS.zone_window_handle_2005)
    # zone_window_2006 = api.exchange.get_variable_value(state_argument, EPLUS.zone_window_handle_2006)
    # zone_window_2007 = api.exchange.get_variable_value(state_argument, EPLUS.zone_window_handle_2007)

    # zone_ventmass_2001 = api.exchange.get_variable_value(state_argument, EPLUS.zone_temp_handle_2001)
    # zone_ventmass_2002 = api.exchange.get_variable_value(state_argument, EPLUS.zone_temp_handle_2002)
    # zone_ventmass_2003 = api.exchange.get_variable_value(state_argument, EPLUS.zone_temp_handle_2003)
    # zone_ventmass_2004 = api.exchange.get_variable_value(state_argument, EPLUS.zone_temp_handle_2004)
    # zone_ventmass_2005 = api.exchange.get_variable_value(state_argument, EPLUS.zone_temp_handle_2005)
    # zone_ventmass_2006 = api.exchange.get_variable_value(state_argument, EPLUS.zone_temp_handle_2006)
    # zone_ventmass_2007 = api.exchange.get_variable_value(state_argument, EPLUS.zone_temp_handle_2007)



    '''
    Store data
    '''
    EPLUS.y_humd.append(oa_humd)
    # EPLUS.y_wind.append([oa_windspeed,oa_winddirct])
    # EPLUS.y_solar.append([oa_solar_azi, oa_solar_alt, oa_solar_ang])
    # EPLUS.y_zone_humd.append([zone_humd_2001,zone_humd_2002,zone_humd_2003,
    #                           zone_humd_2004,zone_humd_2005,zone_humd_2006
    #                           # ,zone_humd_2007
    #                           ])
    
    # EPLUS.y_zone_window.append([zone_window_2001,zone_window_2002,zone_window_2003,
    #                             zone_window_2004,zone_window_2005,zone_window_2006
    #                             # ,zone_window_2007
    #                             ])
    
    # EPLUS.y_zone_ventmass.append([zone_ventmass_2001,zone_ventmass_2002,zone_ventmass_2003,
    #                               zone_ventmass_2004,zone_ventmass_2005,zone_ventmass_2006
    #                               # ,zone_ventmass_2007
    #                               ])

    EPLUS.y_zone_humd_2001.append(zone_humd_2001)
    EPLUS.y_zone_humd_2002.append(zone_humd_2002)
    EPLUS.y_zone_humd_2003.append(zone_humd_2003)
    EPLUS.y_zone_humd_2004.append(zone_humd_2004)
    EPLUS.y_zone_humd_2005.append(zone_humd_2005)
    EPLUS.y_zone_humd_2006.append(zone_humd_2006)
    
    Humd_list = [zone_humd_2001,zone_humd_2002,zone_humd_2003,
             zone_humd_2004,zone_humd_2005,zone_humd_2006]
    Humd_mean = np.mean(Humd_list)
    
    EPLUS.Humd_mean.append(Humd_mean)


    EPLUS.y_outdoor.append(temp_c_to_f(oa_temp))
    # EPLUS.y_htg.append(temp_c_to_f(    zone_htg_tstat))
    # EPLUS.y_clg.append(temp_c_to_f(    zone_clg_tstat))
    # EPLUS.y_zone.append(temp_c_to_f(   zone_temp))
    
    EPLUS.y_zone_temp_2001.append(temp_c_to_f(zone_temp_2001))
    EPLUS.y_zone_temp_2002.append(temp_c_to_f(zone_temp_2002))
    EPLUS.y_zone_temp_2003.append(temp_c_to_f(zone_temp_2003))
    EPLUS.y_zone_temp_2004.append(temp_c_to_f(zone_temp_2004))
    EPLUS.y_zone_temp_2005.append(temp_c_to_f(zone_temp_2005))
    EPLUS.y_zone_temp_2006.append(temp_c_to_f(zone_temp_2006))
    # EPLUS.y_zone_temp_2007.append(temp_c_to_f(zone_temp_2007))
    
    EPLUS.hvac_htg_2001.append(temp_c_to_f(hvac_htg_2001))
    EPLUS.hvac_clg_2001.append(temp_c_to_f(hvac_clg_2001))
    EPLUS.hvac_htg_2002.append(temp_c_to_f(hvac_htg_2002))
    EPLUS.hvac_clg_2002.append(temp_c_to_f(hvac_clg_2002))
    EPLUS.hvac_htg_2003.append(temp_c_to_f(hvac_htg_2003))
    EPLUS.hvac_clg_2003.append(temp_c_to_f(hvac_clg_2003))
    EPLUS.hvac_htg_2004.append(temp_c_to_f(hvac_htg_2004))
    EPLUS.hvac_clg_2004.append(temp_c_to_f(hvac_clg_2004))
    EPLUS.hvac_htg_2005.append(temp_c_to_f(hvac_htg_2005))
    EPLUS.hvac_clg_2005.append(temp_c_to_f(hvac_clg_2005))
    EPLUS.hvac_htg_2006.append(temp_c_to_f(hvac_htg_2006))
    EPLUS.hvac_clg_2006.append(temp_c_to_f(hvac_clg_2006))
    

    T_list = temp_c_to_f(np.array([zone_temp_2001,
                                   zone_temp_2002,
                                   zone_temp_2003,
                                   zone_temp_2004,
                                   zone_temp_2005,
                                   zone_temp_2006]))
    
    EPLUS.y_zone_temp.append(T_list)
    
    
    T_mean = np.mean(T_list)
    
    EPLUS.T_maen.append(T_mean)
    EPLUS.T_diff.append(np.max(T_list)-np.min(T_list))
    EPLUS.T_var.append(np.var(T_list))
        
    EPLUS.E_Facility.append(api.exchange.get_meter_value(state_argument, EPLUS.E_Facility_handle))
    EPLUS.E_HVAC.append(api.exchange.get_meter_value(state_argument,     EPLUS.E_HVAC_handle))
    EPLUS.E_Heating.append(api.exchange.get_meter_value(state_argument,  EPLUS.E_Heating_handle))
    EPLUS.E_Cooling.append(api.exchange.get_meter_value(state_argument,  EPLUS.E_Cooling_handle))
    
    EPLUS.E_HVAC_all.append(api.exchange.get_meter_value(state_argument, EPLUS.E_HVAC_handle))
    
    EPLUS.sun_is_up.append(api.exchange.sun_is_up(state_argument))
    EPLUS.is_raining.append(api.exchange.today_weather_is_raining_at_time(state_argument,                       hour, time_step))
    EPLUS.outdoor_humidity.append(api.exchange.today_weather_outdoor_relative_humidity_at_time(state_argument,  hour, time_step))
    EPLUS.wind_speed.append(api.exchange.today_weather_wind_speed_at_time(state_argument,                       hour, time_step))
    EPLUS.diffuse_solar.append(api.exchange.today_weather_diffuse_solar_at_time(state_argument,                 hour, time_step))

    # api.exchange.today_outdoor_relative_humidity_at_time(state_argument, 19, 2)
    # print(api.exchange.today_weather_wind_direction_at_time(state_argument, hour, time_step))
    
    # Year is bogus, seems to be reading the weather file year instead...         
    # So harcode it to 2022
    year = 2022
    EPLUS.years.append(year)
    EPLUS.months.append(month)
    EPLUS.days.append(day)
    EPLUS.hours.append(hour)
    EPLUS.minutes.append(minute)
    
    EPLUS.current_times.append(current_time)
    EPLUS.actual_date_times.append(actual_date_time)
    EPLUS.actual_times.append(actual_time)
    
    timedelta = datetime.timedelta()
    if hour >= 24.0:
        hour = 23.0
        timedelta += datetime.timedelta(hours=1)
    if minute >= 60.0:
        minute = 59
        timedelta += datetime.timedelta(minutes=1)
    
    dt = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute)
    dt += timedelta
    EPLUS.x.append(dt)
    EPLUS.time_line.append(dt)
    
    if dt.weekday() > 4:
        # print 'Given date is weekend.'
        EPLUS.weekday.append(dt.weekday())
        EPLUS.isweekday.append(0)
        EPLUS.isweekend.append(1)
    else:
        # print 'Given data is weekday.'
        EPLUS.weekday.append(dt.weekday())
        EPLUS.isweekday.append(1)
        EPLUS.isweekend.append(0)
        
    EPLUS.work_time.append(EPLUS.isweekday[-1] * EPLUS.sun_is_up[-1])
    
    

    
    
    ''' 
    DQN 
    
    '''
    if EPLUS.RL_flag == False:
        EPLUS.episode_reward.append(0)

    if EPLUS.RL_flag == True:
    
        if time_interval == 0:
            EPLUS.episode_reward.append(0)
            EPLUS.action_list.append([0,0])
    
        '''
        training
        '''
        
        done = False
        is_worktime = EPLUS.work_time[-1]
        
        # t_-1
        O0 = EPLUS.y_outdoor[-2]
        E0 = EPLUS.E_HVAC[-2]
        W0 = EPLUS.work_time[-2]
        # D0 = EPLUS.weekday[-2]
        # M0 = EPLUS.months[-2]
        # H0 = EPLUS.hours[-2]
        # S0 = EPLUS.sun_is_up[-2]

        # T_10 = EPLUS.y_zone_temp_2001[-2] 
        # T_20 = EPLUS.y_zone_temp_2002[-2] 
        # T_30 = EPLUS.y_zone_temp_2003[-2] 
        # T_40 = EPLUS.y_zone_temp_2004[-2] 
        # T_50 = EPLUS.y_zone_temp_2005[-2] 
        # T_60 = EPLUS.y_zone_temp_2006[-2] 
        
        # H_10 = EPLUS.hvac_htg_2001[-2] 
        # H_20 = EPLUS.hvac_htg_2002[-2] 
        # H_30 = EPLUS.hvac_htg_2003[-2] 
        # H_40 = EPLUS.hvac_htg_2004[-2] 
        # H_50 = EPLUS.hvac_htg_2005[-2] 
        # H_60 = EPLUS.hvac_htg_2006[-2] 
        
        # state_0 = [O0/100,W0,
        #             T_10/100,T_20/100,T_30/100,T_40/100,T_50/100,T_60/100,
        #             H_10/100,H_20/100,H_30/100,H_40/100,H_50/100,H_60/100]
    
        
        action_0 = EPLUS.action_list[-1]
        
        # t_0
        O1 = EPLUS.y_outdoor[-1] 
        E1 = EPLUS.E_HVAC[-1]
        W1 = EPLUS.work_time[-1]
        # D1 = EPLUS.weekday[-1]
        # M1 = EPLUS.months[-1]
        # H1 = EPLUS.hours[-1]
        # S1 = EPLUS.sun_is_up[-1]

        T_11 = EPLUS.y_zone_temp_2001[-1] 
        T_21 = EPLUS.y_zone_temp_2002[-1] 
        T_31 = EPLUS.y_zone_temp_2003[-1] 
        T_41 = EPLUS.y_zone_temp_2004[-1] 
        T_51 = EPLUS.y_zone_temp_2005[-1] 
        T_61 = EPLUS.y_zone_temp_2006[-1] 
        
        # H_11 = EPLUS.hvac_htg_2001[-1]
        # H_21 = EPLUS.hvac_htg_2002[-1] 
        # H_31 = EPLUS.hvac_htg_2003[-1] 
        # H_41 = EPLUS.hvac_htg_2004[-1] 
        # H_51 = EPLUS.hvac_htg_2005[-1]
        # H_61 = EPLUS.hvac_htg_2006[-1]
        
        
        # t_1
        # state_1 = [O1/100,W1,
        #             T_11/100,T_21/100,T_31/100,T_41/100,T_51/100,T_61/100,
        #             H_11/100,H_21/100,H_31/100,H_41/100,H_51/100,H_61/100] 
                   
        
        state_1 = {
                    'T_outdoor': O1,
                    'E_cost': E1,
                    'Work': W1,
                    'T_1':  [T_11,
                             T_21,
                             T_31,
                             T_41,
                             T_51,
                             T_61]
            }
        
        # print(EPLUS.time_interval)

        
        E_pred, action_1 = mpc_controller.optimize(state_1)
        # action_1 = [0,1,0,1,0,1]


        # print(action_1)
        
        
        

        # action_1 = [agent.select_action(state_1[i]).argmax().item() for i, agent in enumerate(agents)]

        
        
        # action_1 = agent.take_action(state_1)
        # state_1 = torch.tensor([state_1], dtype=torch.float).to(device)
        # action_1 = model(state_1).argmax().item()
        # action_map = HVAC_action_list[action_1]

        action_map = np.array(action_1).astype(int)
        # print(action_map)
        
        for i in range(len(action_map)):
            if 68<action_map[i]<77:
                action_map[i] = 0
            else:
                action_map[i] = 1
                

        set_temp = [71,74]

        # Take action
        H_new_1, C_new_1 = HVAC_action(action_map[0], set_temp)
        H_new_2, C_new_2 = HVAC_action(action_map[1], set_temp)
        H_new_3, C_new_3 = HVAC_action(action_map[2], set_temp)
        H_new_4, C_new_4 = HVAC_action(action_map[3], set_temp)
        H_new_5, C_new_5 = HVAC_action(action_map[4], set_temp)
        H_new_6, C_new_6 = HVAC_action(action_map[5], set_temp)
        # api.exchange.set_actuator_value(state_argument, EPLUS.hvac_htg_2001_handle, temp_f_to_c(H_new))
        # api.exchange.set_actuator_value(state_argument, EPLUS.hvac_clg_2001_handle, temp_f_to_c(C_new))
        # api.exchange.set_actuator_value(state_argument, EPLUS.hvac_htg_2003_handle, temp_f_to_c(120))
        # api.exchange.set_actuator_value(state_argument, EPLUS.hvac_clg_2003_handle, temp_f_to_c(120))

        
        
        api.exchange.set_actuator_value(state_argument, EPLUS.hvac_htg_2001_handle, temp_f_to_c(H_new_1))
        api.exchange.set_actuator_value(state_argument, EPLUS.hvac_clg_2001_handle, temp_f_to_c(C_new_1))
        api.exchange.set_actuator_value(state_argument, EPLUS.hvac_htg_2002_handle, temp_f_to_c(H_new_2))
        api.exchange.set_actuator_value(state_argument, EPLUS.hvac_clg_2002_handle, temp_f_to_c(C_new_2))
        api.exchange.set_actuator_value(state_argument, EPLUS.hvac_htg_2003_handle, temp_f_to_c(H_new_3))
        api.exchange.set_actuator_value(state_argument, EPLUS.hvac_clg_2003_handle, temp_f_to_c(C_new_3))
        api.exchange.set_actuator_value(state_argument, EPLUS.hvac_htg_2004_handle, temp_f_to_c(H_new_4))
        api.exchange.set_actuator_value(state_argument, EPLUS.hvac_clg_2004_handle, temp_f_to_c(C_new_4))
        api.exchange.set_actuator_value(state_argument, EPLUS.hvac_htg_2005_handle, temp_f_to_c(H_new_5))
        api.exchange.set_actuator_value(state_argument, EPLUS.hvac_clg_2005_handle, temp_f_to_c(C_new_5))
        api.exchange.set_actuator_value(state_argument, EPLUS.hvac_htg_2006_handle, temp_f_to_c(H_new_6))
        api.exchange.set_actuator_value(state_argument, EPLUS.hvac_clg_2006_handle, temp_f_to_c(C_new_6))
        # api.exchange.set_actuator_value(state_argument, EPLUS.hvac_htg_2007_handle, temp_f_to_c(0))
        # api.exchange.set_actuator_value(state_argument, EPLUS.hvac_clg_2007_handle, temp_f_to_c(120))
        

        EPLUS.action_map.append(action_map)



        # H_list = [H_new_1, H_new_2, H_new_3, H_new_4, H_new_5, H_new_6]
        # C_list = [C_new_1, C_new_2, C_new_3, C_new_4, C_new_5, C_new_6]
        # print(time_interval)
        # print(T_list)
        # print(H_list)
        # print(C_list)
        # print(action_map)
        
        ''' 
        reward define 
        
        '''
        if is_worktime:
            E_factor = 1e-6
            T_factor = 0.01 *5
            work_flag = 0
            reward_signal = 0
            
            E_save = E_factor
            T_save = T_factor
            
        else:
            E_factor = 1e-6 *5
            T_factor = 0
            work_flag = 0
            reward_signal = 0

                
        # 1 Energy
        reward_E = -E1 * E_factor
        
        # 2 Temp
        if 68<T_11<77:
            reward_T1 = 1*work_flag
            # reward_T1 = 0
        else:
            reward_T1 = -(T_11-72)**2 * T_factor 
            # reward_T1 = -((T_11)-72)**2 * T_factor
            # reward_T1 = -1
            
        if 68<T_21<77:
            reward_T2 = 1*work_flag
            # reward_T2 = 0
        else:
            reward_T2 = -(T_21-72)**2 * T_factor 
            # reward_T2 = -((T_21)-72)**2 * T_factor
            # reward_T2 = -1
            
        if 68<T_31<77:
            reward_T3 = 1*work_flag
            # reward_T3 = 0
        else:
            reward_T3 = -(T_31-72)**2 * T_factor
            # reward_T3 = -((T_31)-72)**2 * T_factor
            # reward_T3 = -1

        if 68<T_41<77:
            reward_T4 = 1*work_flag
            # reward_T4 = 0
        else:
            reward_T4 = -(T_41-72)**2 * T_factor
            # reward_T4 = -((T_41)-72)**2 * T_factor
            # reward_T4 = -1

        if 68<T_51<77:
            reward_T5 = 1*work_flag
            # reward_T5 = 0
        else:
            reward_T5 = -(T_51-72)**2 * T_factor 
            # reward_T5 = -((T_51)-72)**2 * T_factor
            # reward_T5 = -1

        if 68<T_61<77:
            reward_T6 = 1*work_flag
            # reward_T6 = 0
        else:
            reward_T6 = -(T_61-72)**2 * T_factor    
            # reward_T6 = -((T_61)-72)**2 * T_factor
            # reward_T6 = -1
        
        # reward_T = (-np.sum((np.array(H_list)-72))**2) * T_factor
        # reward_T = -((np.array(H_new_3)-72))**2 * T_factor
        
        # if 68<H_new_3<77:
        #     reward_T = 1*work_flag
        # else:
        #     reward_T = (-np.abs(H_new_3-72)) * T_factor

        # reward_T = np.mean([reward_T1,reward_T2,reward_T3,reward_T4,reward_T5,reward_T6])
        # reward_T = np.sum([reward_T1,reward_T2,reward_T3,reward_T4,reward_T5,reward_T6])
        reward_T = reward_T1+reward_T2+reward_T3+reward_T4+reward_T5+reward_T6
        
            
        # # 3 Control Signal
        # if np.abs(H1-C1)<10:
        #     reward_HC = 0
        # else:
        #     reward_HC = -1
            
        # # 4 Smootheness
        # if action_list[-1] != 0:
        #     if action_list[-1]==1 and action_list[-2]==2:
        #         reward_signal = -1
        #     if action_list[-1]==2 and action_list[-2]==1:
        #         reward_signal = -1
    
        # 4 Smootheness
        # if action_1 != 0:
            # reward_signal = -1
        
        # # if 1 in action_1:
        # action_map = np.array(action_map)
        # num_unstable = 6-len(action_map[action_map==0])
        # reward_signal = -0.1 * num_unstable
        
        # 4 Smootheness
        current_action = (EPLUS.action_map[-1]).astype(int)
        last_action    = (EPLUS.action_map[-2]).astype(int)
        # print(np.array(current_action) , np.array(last_action))
        # print(action_0, action_1)
        
        change_action = np.array(current_action) ^ np.array(last_action)
        num_unstable = len(change_action[change_action==1])
        reward_signal = -5 * num_unstable
        
        EPLUS.signal.append(num_unstable)

        
        
        # print(num_unstable)
        
        # Total reward
        reward_1 = reward_T + reward_E #+ reward_signal
        # print(reward_E, reward_T)
        
        EPLUS.episode_reward.append(reward_1)
        EPLUS.episode_return = EPLUS.episode_return + reward_1
        

        if is_worktime:
            if T_mean > 77:
                EPLUS.T_Violation.append(T_mean-77)
            elif T_mean < 68:
                EPLUS.T_Violation.append(68-T_mean)
          
        
        if is_worktime:
            if Humd_mean > 80:
                EPLUS.H_Violation.append(Humd_mean-80)
            elif Humd_mean < 20:
                EPLUS.H_Violation.append(20-Humd_mean)
                
                
        # T_violation = []
        # if is_worktime:
        #     for T_i in [T_11,T_21,T_31,T_41,T_51,T_61]:
        #         if T_i > 77:
        #             T_violation.append(T_i-77)
        #         elif T_i < 68:
        #             T_violation.append(68-T_i)
            
        # T_violation = np.sum(T_violation)
        # EPLUS.T_Violation.append(T_violation)
        
        # print(T_violation)
        
        

        if done == True:
            EPLUS.score.append(EPLUS.episode_return)
            EPLUS.episode_return = 0
            
    
                

    '''
    adjust view distance
    
    '''
    if EPLUS.time_interval > view_distance:
        y_outdoor = EPLUS.y_outdoor[-view_distance::]
        y_zone = EPLUS.y_zone[-view_distance::]
        y_htg = EPLUS.y_htg[-view_distance::]
        y_clg = EPLUS.y_clg[-view_distance::]
        
        hvac_htg_2001 = EPLUS.hvac_htg_2001[-view_distance::]        
        hvac_htg_2002 = EPLUS.hvac_htg_2002[-view_distance::]
        hvac_htg_2003 = EPLUS.hvac_htg_2003[-view_distance::]
        hvac_htg_2004 = EPLUS.hvac_htg_2004[-view_distance::]
        hvac_htg_2005 = EPLUS.hvac_htg_2005[-view_distance::]
        hvac_htg_2006 = EPLUS.hvac_htg_2006[-view_distance::]
        # hvac_htg_2007 = EPLUS.hvac_htg_2007[-view_distance::]
        
        hvac_htg_2001 = [1 if value > 70 else -1 for value in hvac_htg_2001]
        hvac_htg_2002 = [1 if value > 70 else -1 for value in hvac_htg_2002]
        hvac_htg_2003 = [1 if value > 70 else -1 for value in hvac_htg_2003]
        hvac_htg_2004 = [1 if value > 70 else -1 for value in hvac_htg_2004]
        hvac_htg_2005 = [1 if value > 70 else -1 for value in hvac_htg_2005]
        hvac_htg_2006 = [1 if value > 70 else -1 for value in hvac_htg_2006]

        y_zone_temp_2001 = EPLUS.y_zone_temp_2001[-view_distance::]
        y_zone_temp_2002 = EPLUS.y_zone_temp_2002[-view_distance::]
        y_zone_temp_2003 = EPLUS.y_zone_temp_2003[-view_distance::]
        y_zone_temp_2004 = EPLUS.y_zone_temp_2004[-view_distance::]
        y_zone_temp_2005 = EPLUS.y_zone_temp_2005[-view_distance::]
        y_zone_temp_2006 = EPLUS.y_zone_temp_2006[-view_distance::]
        T_mean = EPLUS.T_maen[-view_distance::]
        
        y_zone_humd_2001 = EPLUS.y_zone_humd_2001[-view_distance::]
        y_zone_humd_2002 = EPLUS.y_zone_humd_2002[-view_distance::]
        y_zone_humd_2003 = EPLUS.y_zone_humd_2003[-view_distance::]
        y_zone_humd_2004 = EPLUS.y_zone_humd_2004[-view_distance::]
        y_zone_humd_2005 = EPLUS.y_zone_humd_2005[-view_distance::]
        y_zone_humd_2006 = EPLUS.y_zone_humd_2006[-view_distance::]
        H_mean = EPLUS.Humd_mean[-view_distance::]

        # E_Facility = E_Facility[-view_distance::]
        E_HVAC = EPLUS.E_HVAC[-view_distance::]
        # E_Heating = E_Heating[-view_distance::]
        # E_Cooling = E_Cooling[-view_distance::]
        
        episode_reward = EPLUS.episode_reward[-view_distance::]
        
        # sun_is_up = sun_is_up[-view_distance::]
        # is_raining = is_raining[-view_distance::]
        # outdoor_humidity = outdoor_humidity[-view_distance::]
        # wind_speed = wind_speed[-view_distance::]
        # diffuse_solar = diffuse_solar[-view_distance::]
        
        # years = years[-view_distance::]
        # months = months[-view_distance::]
        # days = days[-view_distance::]
        # hours = hours[-view_distance::]
        # minutes = minutes[-view_distance::]
        
        # current_times = current_times[-view_distance::]
        # actual_date_times = actual_date_times[-view_distance::]
        # actual_times = actual_times[-view_distance::]
        
        x = EPLUS.x[-view_distance::]

        # weekday = weekday[-view_distance::]
        # isweekday = isweekday[-view_distance::]
        # isweekend = isweekend[-view_distance::]
        work_time = EPLUS.work_time[-view_distance::]
        
        
    EPLUS.time_interval = EPLUS.time_interval+1
    
    
    # if EPLUS.RL_flag == False:
    #     api.exchange.set_actuator_value(state_argument, EPLUS.hvac_htg_2003_handle, temp_f_to_c(65))
    #     api.exchange.set_actuator_value(state_argument, EPLUS.hvac_clg_2003_handle, temp_f_to_c(65))
    

    
    '''
    Plot
    
    '''
    if EPLUS.time_interval > 2050 and time_interval % EPLUS.FPS == 0:
        # print(
        # oa_humd     ,
        # oa_windspeed,
        # oa_winddirct,
        # oa_solar_azi,
        # oa_solar_alt,
        # oa_solar_ang)
        
        # EPLUS.y_humd
        # EPLUS.y_wind
        # EPLUS.y_solar
        # print(EPLUS.y_zone_humd[-1])
        # print(EPLUS.y_zone_window[-1])
        # print(EPLUS.y_zone_ventmass[-1])

          
        '''Draw Thermal Map'''
        EPLUS.T_map = {}
        
        EPLUS.T_map['Thermal Zone 1'] = temp_c_to_f(zone_temp_2001)
        EPLUS.T_map['Thermal Zone 2'] = temp_c_to_f(zone_temp_2002)
        EPLUS.T_map['Thermal Zone 3'] = temp_c_to_f(zone_temp_2003)
        EPLUS.T_map['Thermal Zone 4'] = temp_c_to_f(zone_temp_2004)
        EPLUS.T_map['Thermal Zone 5'] = temp_c_to_f(zone_temp_2005)
        EPLUS.T_map['Thermal Zone 6'] = temp_c_to_f(zone_temp_2006)
        # EPLUS.T_map['Thermal Zone 7'] = zone_temp_2007
        EPLUS.T_map['Outdoor Temp'] = temp_c_to_f(oa_temp)

        THERMAL_MAP_2D, Zone_Center_XY = ITRC_2.draw_map(EPLUS.T_map)
        
        
        if EPLUS.RL_flag == True:
            # print(f'{time_interval}   {T_31} | {70}   {reward_T} | {reward_E} | {reward_signal}   {action_1} | {action_map}')
            print('%d / %s   %.2f / 72   %.3f / %.3f / %.2f'%(time_interval,dt,T_31,reward_T,reward_E,reward_signal))
            print(action_map, Humd_mean)
            # print(all_critic_losses)
            # print(all_actor_losses)
        
        # plt.clf()
        # plt.clf()

        ax1.cla()
        ax3.cla()

        # twin1.cla()
        # twin2.cla()
    
        ax1.set(ylim=(-20, 100))#, xlabel='Time', ylabel='Temperature' ,fontsize=16)
        # ax1.set_xlabel('Time',fontsize=16)
        ax1.set_ylabel('Temperature', fontsize=16)
        ax1.plot(x,np.array(episode_reward), label='Reward', color='gray', linewidth=2)
        ax1.plot(x,y_outdoor, label='Outdoor_temp', color='orange', linewidth=3)
        # ax1.plot(x,y_zone, label='Zone_temp', color='deepskyblue', linewidth=2)
        ax1.plot(x,np.array(E_HVAC)*1e-6, label='Electricity_HVAC', color='limegreen', linewidth=2)
        
        ax3.set(ylim=(-20*1e4, 100*1e4), ylabel='Electricity_HVAC')
        ax3.plot(x, np.array(E_HVAC)/100, label='Electricity_HVAC', color='limegreen', linewidth=2)
        ax3.set_ylabel('HVAC Electricity (MJ)', fontsize=16)
        # print(np.array(E_HVAC)[-1])
        
        ax1.axhline(y=77, label='Comfort Zone', color='b', linewidth=2)
        ax1.axhline(y=68, label='Comfort Zone', color='b', linewidth=2)
        ax1.axhline(y=90, label='Upper Temperature Constraint', color='r', linewidth=2)
        ax1.axhline(y=60, label='Lower Temperature Constraint', color='r', linewidth=2)
        ax1.axhline(y=80, label='Upper Humidity Constraint', color='teal', linewidth=2)
        ax1.axhline(y=20, label='Lower Humidity Constraint', color='teal', linewidth=2)
        
        
        ax1.plot(x,T_mean, label='Zone_temp_mean', color='violet', linewidth=7)
        ax1.plot(x,y_zone_temp_2001, label='Zone_1_temp', color='C0', linewidth=1)
        ax1.plot(x,y_zone_temp_2002, label='Zone_2_temp', color='C1', linewidth=1)
        ax1.plot(x,y_zone_temp_2003, label='Zone_3_temp', color='C2', linewidth=1)
        ax1.plot(x,y_zone_temp_2004, label='Zone_4_temp', color='C3', linewidth=1)
        ax1.plot(x,y_zone_temp_2005, label='Zone_5_temp', color='C4', linewidth=1)
        ax1.plot(x,y_zone_temp_2006, label='Zone_6_temp', color='C5', linewidth=1)
        
        
        
        ax1.plot(x,H_mean, label='Zone_humd_mean', color='cyan', linewidth=7)
        ax1.plot(x,y_zone_humd_2001, label='Zone_1_humd', color='C0', linewidth=1)
        ax1.plot(x,y_zone_humd_2002, label='Zone_2_humd', color='C1', linewidth=1)
        ax1.plot(x,y_zone_humd_2003, label='Zone_3_humd', color='C2', linewidth=1)
        ax1.plot(x,y_zone_humd_2004, label='Zone_4_humd', color='C3', linewidth=1)
        ax1.plot(x,y_zone_humd_2005, label='Zone_5_humd', color='C4', linewidth=1)
        ax1.plot(x,y_zone_humd_2006, label='Zone_6_humd', color='C5', linewidth=1)
        
        
        

        ax1.fill_between(x, 0, 100, where=work_time,
                facecolor='green', alpha=0.1, transform=trans_1)
        # plt.plot(x,np.array(outdoor_humidity), label='outdoorhumidity')
        # plt.plot(x,np.array(diffuse_solar), label='diffuse solar')
        
        ax1.scatter(x,np.array(hvac_htg_2001)*50, label='VAV_1', marker='s', color='C0')#, linestyle='--')
        ax1.scatter(x,np.array(hvac_htg_2002)*45, label='VAV_2', marker='s', color='C1')#, linestyle='--')
        ax1.scatter(x,np.array(hvac_htg_2003)*40, label='VAV_3', marker='s', color='C2')#, linestyle='--')
        ax1.scatter(x,np.array(hvac_htg_2004)*35, label='VAV_4', marker='s', color='C3')#, linestyle='--')
        ax1.scatter(x,np.array(hvac_htg_2005)*30, label='VAV_5', marker='s', color='C4')#, linestyle='--')
        ax1.scatter(x,np.array(hvac_htg_2006)*25, label='VAV_6', marker='s', color='C5')#, linestyle='--')

        ax1.tick_params(axis='both', which='major', labelsize=16)  # Adjust the font size as desired
        ax1.set_title('Zone Temperature and Thermostat Setpoint', fontsize=20)
        ax1.legend(loc=2, fontsize=16, framealpha=1, fancybox=True, bbox_to_anchor=(1.1,1), borderaxespad=0)#, facecolor='lightgray')
        
        

        # p2, = twin1.plot(x,np.array(E_HVAC), label='Electricity_HVAC', color='limegreen', linewidth=2)
        # twin1.set(ylim=(0, 5e7), ylabel="Electricity_HVAC")
        # # twin1.set_frame_on(True)
        # # twin1.patch.set_visible(False)
        # # plt.setp(twin1.spines.values(), visible=False)
        # # twin1.spines['right'].set_visible(True)
        # twin1.yaxis.label.set_color('darkgreen')
        # twin1.tick_params(axis='y', colors='darkgreen')
        # twin1.spines['right'].set_edgecolor('darkgreen')

        ax2.cla()
        cax.cla()
        th_map = ax2.imshow(THERMAL_MAP_2D, cmap = 'jet')
        # cax.set_ylim(0, 50)

        fig.colorbar(th_map, cax=cax, orientation='vertical')
        ax2.set_title('Zone Temperature Map (°F)', fontsize=20)
        ax2.set_xticks([])
        ax2.set_yticks([])

        ax2.text(Zone_Center_XY[0][0],Zone_Center_XY[0][1],str(int(temp_c_to_f(zone_temp_2001)))+'°', color='w', fontsize=20, ha='center', va='center', path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])
        ax2.text(Zone_Center_XY[1][0],Zone_Center_XY[1][1],str(int(temp_c_to_f(zone_temp_2002)))+'°', color='w', fontsize=20, ha='center', va='center', path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])
        ax2.text(Zone_Center_XY[2][0],Zone_Center_XY[2][1],str(int(temp_c_to_f(zone_temp_2003)))+'°', color='w', fontsize=20, ha='center', va='center', path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])
        ax2.text(Zone_Center_XY[3][0],Zone_Center_XY[3][1],str(int(temp_c_to_f(zone_temp_2004)))+'°', color='w', fontsize=20, ha='center', va='center', path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])
        ax2.text(Zone_Center_XY[4][0],Zone_Center_XY[4][1],str(int(temp_c_to_f(zone_temp_2005)))+'°', color='w', fontsize=20, ha='center', va='center', path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])
        ax2.text(Zone_Center_XY[5][0],Zone_Center_XY[5][1],str(int(temp_c_to_f(zone_temp_2006)))+'°', color='w', fontsize=20, ha='center', va='center', path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])
        # ax2.text(Zone_Center_XY[6][0],Zone_Center_XY[6][1],str(int(zone_temp_2007)), color='w', fontsize=20 ,ha='center', va='center', path_effects=[patheffects.withStroke(linewidth=2, foreground='b')])
        ax2.text(250, 25, 'Outdoor Temperature: '+str(int(temp_c_to_f(oa_temp)))+'°F',                 color='w', fontsize=20 ,ha='center', va='center', path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])
        ax2.text(250, 350, str(dt),                                                      color='w', fontsize=20 ,ha='center', va='center', path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])


        # Zone_Center_XY

        # cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        # plt.colorbar(th_map)
        # plt.imshow(map_2D, cmap = 'jet')
        # plt.colorbar()
        # ax2.pause(0.001)
        

    
        # ax3.set_ylim(30, 100)
        # ax3.plot(x,y_outdoor, label='Outdoor_temp', color='orange', linewidth=2)
        # # ax1.plot(x,y_zone, label='Zone_temp', color='deepskyblue', linewidth=2)
        # # ax3.plot(x,np.array(E_HVAC)/(1e4*NUM_HVAC), label='Electricity_HVAC', color='violet')
        
        # ax3.plot(x,T_mean, label='Zone_temp_mean', color='violet', linewidth=2)


        # ax3.axhline(y=77, label='Upper comfort line', color='g')
        # ax3.axhline(y=68, label='Lower comfort line', color='g')

        # ax3.scatter(x,np.array(hvac_htg_2001)-10, label='Zone_Heat_1', marker='s')#, color='r', linestyle='--')
        # ax3.scatter(x,np.array(hvac_htg_2002)-15, label='Zone_Heat_2', marker='s')#, color='r', linestyle='--')
        # ax3.scatter(x,np.array(hvac_htg_2003)-20, label='Zone_Heat_3', marker='s')#, color='r', linestyle='--')
        # ax3.scatter(x,np.array(hvac_htg_2004)-25, label='Zone_Heat_4', marker='s')#, color='r', linestyle='--')
        # ax3.scatter(x,np.array(hvac_htg_2005)-30, label='Zone_Heat_5', marker='s')#, color='r', linestyle='--')
        # ax3.scatter(x,np.array(hvac_htg_2006)-35, label='Zone_Heat_6', marker='s')#, color='r', linestyle='--')

        # # ax3.plot(x,np.array(work_time)*10+30, label='work time', color='black')
        # ax3.fill_between(x, 0, 100, where=work_time,
        #         facecolor='green', alpha=0.1, transform=trans_3)
        
        # # plt.plot(np.array(isweekday)+40, label='is weekday')
        # # ax3.plot(x,np.array(work_time)*10+30, label='work time', color='black')
        # # plt.plot(x,np.array(outdoor_humidity), label='outdoorhumidity')
        # # plt.plot(x,np.array(diffuse_solar), label='diffuse solar')
        
        # # ax3.plot(x,np.array(episode_reward)+100, label='reward', color='black')

        # ax3.set_title('Thermostat Setpoint in 6 Zones', fontsize=20)
        # ax3.legend(loc=2, fontsize=16, framealpha=1, fancybox=True)#, facecolor='lightgray')



        # plt.title('Zone_0 Temps and Thermostat Setpoint - DL, epoch:{}'.format(epoch), fontsize=20)
        # plt.savefig('./gif/'+str(time_interval)+'.jpg')
        # plt.show()
        # plt.show(False)
        
        
        
        plt.pause(0.001)
        



    # if time_interval % 5000 == 0 and EPLUS.RL_flag == True:
    #     torch.save(agent.target_q_net.state_dict(), f'./weights/Enet_{time_interval}.pth')

            

        
###############################################################################


'''
main function

'''
import matplotlib.transforms as mtransforms
import matplotlib.patheffects as patheffects
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits import axisartist


fig = plt.figure(figsize=(20,30), dpi=100)
fig.subplots_adjust(left=0.05, right=0.75)
ax1 = plt.subplot(2,1,1)
# ax3 = plt.subplot(2,1,2)
ax2 = plt.subplot(2,1,2)
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.1)
cax.tick_params(labelsize=20)
trans_1 = mtransforms.blended_transform_factory(ax1.transData, ax1.transAxes)
# trans_3 = mtransforms.blended_transform_factory(ax3.transData, ax3.transAxes)
# twin1 = ax1.twinx()
# twin2 = ax1.twinx()
# twin2.spines['right'].set_position(("axes", 2))

ax3 = ax1.twinx()
ax3.tick_params(labelsize=16)



'''
Building Model
'''

filename_to_run = 'test.idf'

ITRC_2 = Building("test.idf")
ITRC_2.idf.idfobjects['BuildingSurface:Detailed']
ITRC_2.AirWall_Switch('on')
ITRC_2.Roof_Switch('off')
ITRC_2.idf.saveas('test.idf')

THERMAL_MAP_2D = ITRC_2.init_map_2D()
building_floor = ITRC_2.building_floor




'''
E with rule based

'''
time_start = time.time()

EPLUS = Data_Bank()
EPLUS.FPS = 10000
EPLUS.RL_flag = False

filename_to_run = 'test.idf'


# 
api = EnergyPlusAPI()

E_state = api.state_manager.new_state()
api.runtime.set_console_output_status(E_state, False)
api.runtime.callback_begin_zone_timestep_after_init_heat_balance(E_state, callback_function_DQN)
   
api.runtime.run_energyplus(E_state,
    [
        '-w', './weather_data/USA_SC_Greenville-Spartanburg.Intl.AP.723120_TMY3.epw',
        '-d', 'out/',
        filename_to_run
    ]
)

# If you need to call run_energyplus again, then reset the state first
api.state_manager.reset_state(E_state)
api.state_manager.delete_state(E_state)


E_HVAC_all_RBC = copy.deepcopy(EPLUS.E_HVAC_all)
E_Facility_all_RBC = copy.deepcopy(EPLUS.E_Facility)


print(np.sum(EPLUS.E_HVAC_all))
print(np.mean(EPLUS.T_maen))
print(np.mean(EPLUS.T_diff))
print(np.mean(EPLUS.T_var))

time_end = time.time()
time_round = time_end-time_start
print(time_round)

# Air Infiltration On
# 277236608284.40936
# 69.4161690223531
# 0.9515681711048721
# 0.3451205798764253


# Air Infiltration Off
# 223151610356.13086
# 69.34490697081893
# 1.0882843834103764
# 0.3988661116151047

# Air Wall
# 212233225957.90146
# 69.39075664903316
# 0.5252089535131921
# 0.09681760914863688

E_defult = 212233225957




import cvxpy as cp
import numpy as np


class HVACMPC:
    def __init__(self):
        self.u = cp.Variable(6)
        # self.T = cp.Variable(6)
        self.airflow_rate = 0.5  # m³/s
        self.specific_heat_capacity_air = 1005  # J/kg°C

    def optimize(self, state):
        
        # Define the energy cost based on your system's energy model
        T = state['T_1']
        E = state['E_cost']
        W = state['Work']
        
        # print(E)
        
        E_HVAC_expressions = []
        
        T_diff = []
        
        
        for i in range(len(T)):
    
            # Define piecewise linear functions for cooling and heating energy
            temperature_difference_cooling = cp.maximum(T[i] - self.u[i], 0)
            cooling_energy = (E/6) * temperature_difference_cooling
            
            temperature_difference_heating = cp.maximum(self.u[i] - T[i], 0)
            heating_energy = 3 * (E/6) * temperature_difference_heating
            
            E_HVAC_expressions.append(cooling_energy + heating_energy)
            
            T_HVAC_expressions = (cp.abs(T[i] - 72))*(cp.maximum(0, self.u[i] - 77) + cp.maximum(0, 68 - self.u[i])) #cp.maximum(72 - self.u[i], 0)
            # T_HVAC_expressions = (cp.abs(T[i] - 72))
            T_diff.append(T_HVAC_expressions)
        
        E_total = cp.sum(E_HVAC_expressions) * 1e-9
        T_total = cp.sum(T_diff)
        # print(E, E_total.value)
        # print(T)
        
        Loss_total = E_total*W + T_total*(1-W)
        # print(E_total.value, T_total.value)

        # Define a simple objective function (minimize the sum of squared control inputs)
        self.objective = cp.Minimize(Loss_total)
        
        # Define constraints (e.g., temperature limits, VAV constraints, etc.) - Simplified for demonstration
        self.constraints = [

            self.u >= 60,
            self.u <= 90,
            # E_total >= E
        ]

        # Set the energy cost objective in the optimization problem
        self.problem = cp.Problem(self.objective, self.constraints)
        self.problem.solve()
        # print("Problem Status:", self.problem.status)


        # Apply the first control input to the system
        u_optimal = self.u.value
        new_action = np.array(u_optimal)
        # print(new_action)

        return Loss_total.value, new_action



# Initialize MPC controller
mpc_controller = HVACMPC()

state = {
            'T_outdoor': 60,
            'E_cost': 500,
            'Work': 1,
            'T_1':  [70,
                     77,
                     80,
                     65,
                     75,
                     72]
    }


new_action = mpc_controller.optimize(state)



'''
Hyperparameters for DQN

'''

epochs = 10
# lr = 1e-3
# num_episodes = 100

# gamma = 0.9
# epsilon = 0
# target_update = 20
# buffer_size = 10000
# minimal_size = 200
# batch_size = 128

# state_dim = 14
# action_dim = 2**6

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# agent = DQN(state_dim, action_dim, lr, gamma, epsilon, target_update, device)
# replay_buffer = ReplayBuffer(buffer_size)

# agent.q_net.load_state_dict(torch.load('./weights/Enet_last_19.pth'))




'''
DQN
'''

Benchmark = np.zeros((epochs,30), dtype=object)


for epoch in range(epochs):
    time_start = time.time()


    EPLUS = Data_Bank()
    EPLUS.FPS = 1000
    EPLUS.RL_flag = True
    # EPLUS.view_distance = 500

    print('iteration: ', epoch)
    # file_name= 'ITRC_2nd_1zone.osm'
    # output_name = 'test.idf'
    # utils.OSM(file_name, output_name, time_step, start_month, start_day, end_month, end_day)


    # 
    api = EnergyPlusAPI()

    E_state = api.state_manager.new_state()
    api.runtime.set_console_output_status(E_state, False)
    api.runtime.callback_begin_zone_timestep_after_init_heat_balance(E_state, 
                                                                     callback_function_DQN)
       
    api.runtime.run_energyplus(E_state,
        [
            '-w', './weather_data/USA_SC_Greenville-Spartanburg.Intl.AP.723120_TMY3.epw',
            '-d', 'out/',
            filename_to_run
        ]
    )
    
    # If you need to call run_energyplus again, then reset the state first
    api.state_manager.reset_state(E_state)
    api.state_manager.delete_state(E_state)

    # torch.save(agent.target_q_net.state_dict(), f'./weights/Enet_last_{epoch}.pth')
    # print(f'Saving to ./weights/Enet_last_{epoch}.pth')
    
    # # Save agent weights
    # for i, agent in enumerate(agents):
    #     torch.save(agent.actor.state_dict(), f"./weights/MADDPG_HVAC/agent_{i}_actor_weights.pth")
    #     torch.save(agent.critic.state_dict(), f"./weights/MADDPG_HVAC/agent_{i}_critic_weights.pth")
    


    E_HVAC_all_DQN = copy.deepcopy(EPLUS.E_HVAC_all)
    
    time_end = time.time()
    time_round = time_end-time_start
    print(time_round)
        
        
    
    # plt.figure(figsize=(30,10), dpi=100)
    # plt.plot(E_HVAC_all_RBC, label='Default')
    # plt.plot(E_HVAC_all_DQN, label='DQN')
    # plt.legend()
    
    
    # x_sum_1 = np.sum(E_HVAC_all_RBC)
    x_sum_1 = E_defult
    x_sum_2 = np.sum(E_HVAC_all_DQN)
    
    print(f'{x_sum_2}/{x_sum_1}')
    
    E_save = (x_sum_1-x_sum_2)/x_sum_1
    
    work_time_length = EPLUS.work_time.count(1)
    
    
    # work_time_length/len(E_HVAC_all_RBC)
    
    T_violation = len(EPLUS.T_Violation) / len(EPLUS.x)
    T_violation_new = len(EPLUS.T_Violation)/ work_time_length
    T_violation_offset = np.mean(EPLUS.T_Violation)
    
    H_Violation = len(EPLUS.H_Violation) / len(EPLUS.x)
    
    
    print(E_save)
    print(T_violation)
    print(T_violation_new)
    print(T_violation_offset)
    
    print(H_Violation)
    
    
    print(np.sum(EPLUS.signal))
    
    
    
    Benchmark[epoch, 0] = epoch
    Benchmark[0, 1] = E_HVAC_all_RBC
    Benchmark[epoch, 2] = E_HVAC_all_DQN
    Benchmark[epoch, 3] = E_save
    Benchmark[epoch, 4] = T_violation
    Benchmark[epoch, 5] = T_violation_offset
    Benchmark[epoch, 6] = EPLUS.T_Violation
    Benchmark[epoch, 7] = EPLUS.episode_reward
    Benchmark[epoch, 8] = EPLUS.action_list
    Benchmark[epoch, 9] = time_round
    # Benchmark[epoch, 10] = EPLUS.score
    Benchmark[0, 11] = EPLUS.time_line
    # Benchmark[0, 12] = EPLUS.months
    # Benchmark[0, 13] = EPLUS.y_outdoor
    # Benchmark[epoch, 14] = EPLUS.y_zone_temp_2003
    # Benchmark[epoch, 15] = np.array(EPLUS.E_HVAC)
    # Benchmark[epoch, 16] = EPLUS.T_maen
    # Benchmark[epoch, 17] = EPLUS.T_diff
    # Benchmark[epoch, 18] = EPLUS.T_var
    # Benchmark[epoch, 19] = np.sum(EPLUS.E_Facility)
    # Benchmark[epoch, 20] = np.array(EPLUS.y_zone_temp)
    # Benchmark[0, 21] = EPLUS.y_humd
    # Benchmark[epoch, 22] = np.array(EPLUS.y_wind)
    # Benchmark[epoch, 23] = np.array(EPLUS.y_solar)
    # Benchmark[epoch, 24] = np.array(EPLUS.y_zone_humd)
    # Benchmark[epoch, 25] = np.array(EPLUS.y_zone_window)
    # Benchmark[epoch, 26] = np.array(EPLUS.y_zone_ventmass)
    Benchmark[0, 27] = EPLUS.work_time
    # Benchmark[epoch, 28] = EPLUS.E_Heating
    # Benchmark[epoch, 29] = EPLUS.E_Cooling

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





# plt.figure(figsize=(30,10), dpi=100)
# # plt.plot(EPLUS.E_Heating, label='E_Heating')
# # plt.plot(EPLUS.E_Cooling, label='E_Cooling')
# # plt.plot(Benchmark[:, 3], label='E_save')
# # plt.plot(Benchmark[:, 4], label='T_violation')
# plt.plot(Benchmark[5, 14], label='y_zone_temp_2003')
# plt.legend()



# epoch_reward = []
# E_HVAC_all_DQN_list = []
# for i in range(len(Benchmark[:,7])):
#     epoch_reward.append(np.sum((Benchmark[i,7])))
#     E_HVAC_all_DQN_list.append(np.sum((Benchmark[i,2])))
    
    
# plt.figure(figsize=(30,10), dpi=100)
# # plt.plot(Benchmark[:,3], label='Save')
# # plt.plot(epoch_reward, label='Reward')
# # plt.plot(E_HVAC_all_DQN_list, label='HVAC_DQN')
# # plt.plot(EPLUS.score, label='EPLUS.score')

# plt.legend()


np.sum(Benchmark[:,9])/3600


# ##############################################################################

np.max(EPLUS.E_HVAC)
# 21726609.77728757
# 2e7

np.max(EPLUS.y_outdoor)

np.min(EPLUS.y_outdoor)


# np.mean(EPLUS.y_zone_temp_2003)
# np.sum(EPLUS.E_HVAC)

# Jan: Wall/AirWall
# 63.5311972239676 / 63.02235356113362 Cooling
# 69.7469089859147 / 71.22477989425911 Heating
# 23592735872.977375 / 23718605276.161263
# (23718605276.161263 - 23592735872.977375)/23718605276.161263
          
# July:
# 70.1122848302831 / 71.38110722009255 Cooling
# 76.7508946915637 / 80.09217829174628 Heating
# 18557742542.390762 / 18185587129.98734
# (18557742542.390762 - 18185587129.98734)/18557742542.390762





zone_temp_all = [EPLUS.y_zone_temp_2001,
                 EPLUS.y_zone_temp_2002,
                 EPLUS.y_zone_temp_2003,
                 EPLUS.y_zone_temp_2004,
                 EPLUS.y_zone_temp_2005,
                 EPLUS.y_zone_temp_2006
                 # EPLUS.y_zone_temp_2007
                 ]

np.mean(zone_temp_all)

np.mean(EPLUS.y_zone_temp_2003)
# Roof / NoRoof

# JAN
# 63.49234852849166 / 63.03043538080452
# 71.22477989425911 / 70.78489618879087
# 63.02235356113362 / 62.42053702072791


# JULY
# 72.17772861569378 / 71.42787137368529
# 80.09217829174628 / 79.3338461807454



# E_HVAC_all_Default = copy.deepcopy(E_HVAC_all)


# plt.figure(figsize=(30,10), dpi=100)
# plt.plot(E_HVAC_all_Default, label='Default')
# plt.plot(E_HVAC_all_DQN, label='DQN')
# plt.legend()


# x_mean_1 = np.mean(E_HVAC_all_Default[-5000:])
# x_mean_2 = np.mean(E_HVAC_all_DQN[-5000:])

# print(f'{x_mean_2}/{x_mean_1}')

# print((x_mean_1-x_mean_2)/x_mean_1)





# api = EnergyPlusAPI()
# state = api.state_manager.new_state()

# api.runtime.callback_begin_zone_timestep_after_init_heat_balance(state, callback_function)
# api.runtime.run_energyplus(state,
#     [
#         '-w', './weather_data/USA_SC_Greenville-Spartanburg.Intl.AP.723120_TMY3.epw',
#         '-d', 'out',
#         filename_to_run
#     ]
# )

# # If you need to call run_energyplus again, then reset the state first
# api.state_manager.reset_state(state)



# # plt.plot(y_air)
# # plt.plot(y_zone)

# # plt.figure(figsize=(5,5),dpi=200)
# # # plt.plot(y_zone)
# # plt.plot(y_test)




# plt.plot(E_Facility)
# plt.plot(E_HVAC)
# plt.plot(E_Heating)
# plt.plot(E_Cooling)


    

plt.plot(Benchmark[:,3])
plt.plot(Benchmark[:,4])

# # x = np.random.randint(10,30,(len(T),2))
# # x = torch.Tensor(x).unsqueeze(1).to(device)
# T0 = torch.Tensor(y_zone).unsqueeze(1).to(device)
# E0 = torch.Tensor(E_HVAC).unsqueeze(1).to(device)*1e-4
# W0 = torch.Tensor(y_outdoor).unsqueeze(1).to(device)
# H0 = torch.Tensor(y_htg).unsqueeze(1).to(device)
# C0 = torch.Tensor(y_clg).unsqueeze(1).to(device)
# # S0 = (torch.Tensor((isweekday)) * torch.Tensor((sun_is_up))).unsqueeze(1).to(device)
# S0 = (torch.Tensor(work_time)).unsqueeze(1).to(device)
# D0 = torch.Tensor(outdoor_humidity).unsqueeze(1).to(device)
# R0 = torch.Tensor(diffuse_solar).unsqueeze(1).to(device)
# # I = torch.Tensor(wind_speed).unsqueeze(1).to(device)

# X = torch.cat((T0,W0,S0),1).to(device)


# hc = torch.cat((H0,C0),1).to(device)

# # X = torch.cat((T0,W0,S0),1).to(device)

# # T_mean = copy.deepcopy(C0)
# # T_mean = torch.Tensor(T_mean).unsqueeze(1).to(device)
# # T_mean = torch.full((X.shape[0],1), 75., device=device)
# # # T_mean = torch.Tensor([71.]).unsqueeze(1).to(device)
# # E_base = torch.full((X.shape[0],1),  0., device=device)





# D0_T = copy.deepcopy(y_zone)
# D0_E = copy.deepcopy(E_HVAC)
# D0_W = copy.deepcopy(y_outdoor)
# D0_H = copy.deepcopy(y_htg)
# D0_C = copy.deepcopy(y_clg)

# print(np.array(D0_E).sum())









# ##############################################################################


plt.figure(figsize=(30,10), dpi=100)
plt.plot(E_HVAC_all_RBC, label='Default')
plt.plot(E_HVAC_all_DQN, label='DQN')
plt.legend()


x_sum_1 = np.sum(E_HVAC_all_RBC)
x_sum_2 = np.sum(E_HVAC_all_DQN)

print(f'{x_sum_2}/{x_sum_1}')

print((x_sum_1-x_sum_2)/x_sum_1)

print(len(EPLUS.T_Violation)/ len(E_HVAC_all_DQN))




plt.figure(figsize=(30,10), dpi=100)
plt.plot(EPLUS.E_Heating, label='E_Heating')
plt.plot(EPLUS.E_Cooling, label='E_Cooling')
plt.legend()


E_sum_1 = np.sum(EPLUS.E_Heating)
E_sum_2 = np.sum(EPLUS.E_Cooling)

print((E_sum_1)/(E_sum_1+E_sum_2))
print((E_sum_2)/(E_sum_1+E_sum_2))




# x_sum_1 = 304346141135
# x_sum_2 = 295272470795


# print((x_sum_1-x_sum_2)/x_sum_1)



# 6 ZONE TEST

# Annual
# Solid Wall
# 304346141135.4219
# 69.49808747308315
# 1.221687707052417
# 0.6984579654458986

# Air Wall
# 295272470795.2401
# 69.45470519610978
# 0.47378044027921984
# 0.08852141182890427

# Jan
# Solid Wall
# 50853195503.72462
# 68.04659971256895
# 0.502361640413852
# 0.23274991563187647

# Air Wall
# 49016114578.74759
# 68.02803879914435
# 0.329923940253477
# 0.07365338972564173


# July
# Solid Wall
# 22262069773.675007
# 71.25202681230533
# 2.1130047606866205
# 1.4238726730392306

# Air Wall
# 21994331596.054016
# 71.10467217176982
# 0.5864088934979266
# 0.10676893927187905

















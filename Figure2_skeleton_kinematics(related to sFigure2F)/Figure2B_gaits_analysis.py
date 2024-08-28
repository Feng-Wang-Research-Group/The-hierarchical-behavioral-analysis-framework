#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 17:26:41 2022

@author: yejohnny
"""


import os 
import pandas as pd 
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from itertools import chain
import mpl_scatter_density
import math
import scipy.optimize as optimize
import time


normalized_ske_file_dir   = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\01_BehaviorAtlas_collated_data'
revised_data_path_dir = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure2_skeleton_kinematics(related to sFigure2F)\stride'
if not os.path.exists(output_dir):                                    
    os.mkdir(output_dir) 
    
def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if file_name.endswith(content):
            USN = int(file_name.split('-')[1])
            file_path_dict.setdefault(USN,file_dir+'/'+file_name)
    return(file_path_dict)

skip_file_list = [1,3,28,29,110,122] 


Skeleton_path = get_path(normalized_ske_file_dir,'normalized_skeleton_XYZ.csv')
Movement_Label_path = get_path(revised_data_path_dir,'revised_Movement_Labels.csv')
Feature_space_path = get_path(revised_data_path_dir,'Feature_Space.csv')

animal_info_csv = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Table_S1_animal_information.csv'
animal_info = pd.read_csv(animal_info_csv)
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)]


Morning_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Morning') & (animal_info['LightingCondition']=='Light-on')]
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-off')]

stress_info = animal_info[animal_info['ExperimentCondition']=='Stress']

movement_order = ['running','trotting','left_turning','right_turning','walking','stepping','sniffing',
                  'rising','hunching','rearing','climbing','jumping','grooming','scratching','pausing']


movement_color_dict = {'running':'#FF3030',
                       'trotting':'#E15E8A',                       
                       'left_turning':'#F6BBC6', 
                       'right_turning':'#F8C8BA',
                       'walking':'#EB6148',
                       'stepping':'#C6823F',  
                       'sniffing':'#2E8BBE',
                       'rising':'#84CDD9',
                       'hunching':'#D4DF75',
                       'rearing':'#88AF26',
                       'climbing':'#2E7939',                           
                       'jumping':'#24B395',                                              
                       'grooming':'#973C8D',
                       'scratching':'#EADA33',
                       'pausing':'#B0BEC5',}

locomotion_list = ['running','trotting','walking','stepping'] #'left_turning','right_turning',

def get_color(name):
    if name.startswith('Morning'):
        color = '#F5B25E'
    elif name.startswith('Afternoon'):
        color = '#936736'
    elif name.startswith('Night_lightOn'):
        color = '#398FCB'
    elif name.startswith('Night_lightOff'):
        color = '#003960'
    elif name.startswith('Stress') | name.startswith('stress'):
        color = '#f55e6f'
    else:
        print(name)
    
    return(color)


dataset = Morning_lightOn_info
dataset_name = 'Morning_lightOn'
dataset_color = get_color(dataset_name)


dataset2 = stress_info
dataset_name2 = 'stress'
dataset_color2 = get_color(dataset_name2)


def judge_ascending(alist,blist):       
    new_list1 = []                    
    new_list2 = []               
    peak_true = []                #peak_index
    valley_true = []              #value_index
    for i in range(1,len(alist)):
        v1 = alist[i-1]
        v2 = alist[i]
        v3 = blist[i-1]
        v4 = blist[i]
        if v1 < v2:
            new_list1.append(v1)
            if v3 > 0:
                peak_true.append(v1)
                new_list2.append(v3)
            else:
                valley_true.append(v1)
                new_list2.append(-v3)
        else:
            break
    return(peak_true,valley_true,new_list1,new_list2)


def find_sequence_peaks(arr,distance,prominence):
    
    peaks, properties= find_peaks(arr, threshold=None, distance=distance, prominence=prominence, width=None, wlen=None, rel_height=None, plateau_size=None)
    #time_index = indices[0]
    #value = indices[1]['peak_heights']
    return(peaks, arr[peaks])
    


def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'valid')  # numpy的卷积函数


def calculate_GapAndValue(distance_name,data,mv):
    
    arr = moving_average(data,2)
    
    
    if mv == 'walking':
        height1=height2 = 48
        distance = 3
        prominence = 5
        length = 8
    if mv == 'running':
        #height1=height2 = 40
        distance = 1
        prominence = 1
        length = 4
    if mv == 'trotting':
        #height1=height2 = 40
        distance = 1
        prominence = 1
        length = 4
    if mv == 'stepping':
        #height1=height2 = 40
        distance = 3
        prominence = 5
        length = 5

    num = 0
    step_info_dict = {'revised_movement_label':[],'step_type':[],'bouts':[],'valley_time_gap':[],'vally_value':[],'peak_time_gap':[],'peak_value':[]}
    peak1_time_index,peak1_value = find_sequence_peaks(arr,distance=distance,prominence=prominence)
    valley1_time_index,valley1_value = find_sequence_peaks(-arr,distance=distance,prominence=prominence)
    
    if len(peak1_time_index) >0 and len(valley1_time_index) >0:
        if peak1_time_index[0] < valley1_time_index[0]:
            time_sequence = list(chain.from_iterable(zip(peak1_time_index, valley1_time_index)))
            value = list(chain.from_iterable(zip(peak1_value, valley1_value)))
        else:
            time_sequence = list(chain.from_iterable(zip(valley1_time_index, peak1_time_index)))
            value = list(chain.from_iterable(zip(valley1_value, peak1_value)))
        
        peak_true_index_list,valley_true_index_list,new_time_sequence,new_value = judge_ascending(time_sequence,value)

        if len(peak_true_index_list+valley_true_index_list)>length:
            peak_gap_average = np.mean(np.diff(peak_true_index_list))
            peak_value_average = np.mean(data[peak_true_index_list])
            valley_gap_average = np.mean(np.diff(valley_true_index_list))
            valley_value_average = np.mean(data[valley_true_index_list])
            step_info_dict['peak_time_gap'].append(peak_gap_average)
            step_info_dict['peak_value'].append(peak_value_average)
            step_info_dict['valley_time_gap'].append(valley_gap_average)
            step_info_dict['vally_value'].append(valley_value_average)
            step_info_dict['revised_movement_label'].append(mv)
            step_info_dict['bouts'].append(num)
            step_info_dict['step_type'].append(distance_name)
            
    df = pd.DataFrame(step_info_dict)
    return(df)

def averageXYsquence(peak_value,peak_gap,valley_value,valley_gap,num):
    
    x_list = [0,]
    y_list = [valley_value,]
    
    x = 0
    y = valley_value
    for num in range(1,num):
        if num%2 != 0:
            x += peak_gap
            y = peak_value
            x_list.append(x)
            y_list.append(y)
        else:
            x += valley_gap
            y = valley_value
            x_list.append(x)
            y_list.append(y)
    return(x_list,y_list)

def sinY(arr,peak_value,valley_value,peak_gap,valley_gap):
    A = (peak_value-valley_value)/2            
    B = (2*np.pi)/(peak_gap+valley_gap)        
    C = 4.75                                   
    D = (peak_value + valley_value)/2          
    arrY = []
    for x in arr:
        y = A*math.sin(B*x + C) + D
        arrY.append(y)
    return(arrY)




def plot_curve(all_df,dataset_name):

    fig,ax = plt.subplots(nrows=4,ncols=1,figsize=(12,10),dpi=300,sharex=True,sharey=True)
    ax_num = 0
    for i in locomotion_list:
        
        datase1_Data = all_df[all_df['group']==dataset_name]
        color = movement_color_dict[i]
        peak_time_gap_value = datase1_Data.loc[datase1_Data['revised_movement_label']==i,'peak_time_gap'].round(3)
        peak_value_value = datase1_Data.loc[datase1_Data['revised_movement_label']==i,'peak_value'].round(3)
        valley_time_gap_value = datase1_Data.loc[datase1_Data['revised_movement_label']==i,'valley_time_gap'].round(3)
        valley_alue_value = datase1_Data.loc[datase1_Data['revised_movement_label']==i,'vally_value'].round(3)
        
        peak_gap = np.mean(peak_time_gap_value)
        peak_value = np.mean(peak_value_value)
        valley_gap = np.mean(valley_time_gap_value)
        valley_value = np.mean(valley_alue_value)
        
        step_strike_x, step_strike_y = averageXYsquence(peak_value,peak_gap,valley_value,valley_gap,num)
        
        linestype = '--'
     
        t = np.arange(0.0,60,0.1)

        s = sinY(t,peak_value,valley_value,peak_gap,valley_gap)
        ax[ax_num].plot(t, s,c=dataset_color,lw=7,linestyle= linestype,alpha=1)
        
        
        datase2_Data = all_df[all_df['group']==dataset_name2]
        color = movement_color_dict[i]
        peak_time_gap_value = datase2_Data.loc[datase2_Data['revised_movement_label']==i,'peak_time_gap'].round(3)
        peak_value_value = datase2_Data.loc[datase2_Data['revised_movement_label']==i,'peak_value'].round(3)
        valley_time_gap_value = datase2_Data.loc[datase2_Data['revised_movement_label']==i,'valley_time_gap'].round(3)
        valley_alue_value = datase2_Data.loc[datase2_Data['revised_movement_label']==i,'vally_value'].round(3)
        
        peak_gap = np.mean(peak_time_gap_value)
        peak_value = np.mean(peak_value_value)
        valley_gap = np.mean(valley_time_gap_value)
        valley_value = np.mean(valley_alue_value)
        
        step_strike_x, step_strike_y = averageXYsquence(peak_value,peak_gap,valley_value,valley_gap,num)
        
        linestype = '-'
     
        t = np.arange(0.0,60,0.1)

        s = sinY(t,peak_value,valley_value,peak_gap,valley_gap)
        ax[ax_num].plot(t, s,c=dataset_color2,lw=7,linestyle= linestype,alpha=1 )
        ax[ax_num].grid(axis='y',lw=3,zorder=-1)
        
        
        ax[ax_num].set_yticks(np.arange(30,51,10))
        ax[ax_num].set_ylim(30,55)
        ax[ax_num].spines['bottom'].set_linewidth(4)
        ax[ax_num].spines['left'].set_linewidth(4)
        ax[ax_num].spines['top'].set_visible(False)
        ax[ax_num].spines['right'].set_visible(False)
        ax[ax_num].tick_params(width=4,length=6)
        ax[ax_num].yaxis.set_major_formatter(plt.NullFormatter())
        ax[ax_num].xaxis.set_major_formatter(plt.NullFormatter())
        ax[ax_num].set_ylabel('stride length')
        ax[ax_num].set_title(i)
        
        ax_num += 1
        #ax.plot(step_strike_x,step_strike_y,c=color,lw=5)
    plt.xlabel('stride frequency')
    plt.savefig('{}/{}_{}_strike_info_all.png'.format(output_dir,dataset_name,dataset_name2),dpi=300,transparent=True)




t1 = time.time()
df_list = []
#for key in list(FeA_path.keys())[0:5]:
for key in list(dataset['video_index']):
    FeA_file = pd.read_csv(Feature_space_path[key])
    FeA_file = FeA_file[FeA_file['revised_movement_label'].isin(locomotion_list)]
    Skeleton_file = pd.read_csv(Skeleton_path[key])
    for mv in locomotion_list:#FeA_file['movement_label'].unique(): #,'left_turning','right_turning'
        singleLocomotionFeA = FeA_file[FeA_file['revised_movement_label']==mv]
        num = 0
        for i in singleLocomotionFeA.index:
            time_sequence = [0]
            value = [0]
            segBoundary_start = singleLocomotionFeA.loc[i,'segBoundary_start']
            segBoundary_end = singleLocomotionFeA.loc[i,'segBoundary_end']
            frame_length = singleLocomotionFeA.loc[i,'frame_length']
            distance1 = np.array(np.sqrt((Skeleton_file.loc[segBoundary_start:segBoundary_end,'left_front_claw_x']-Skeleton_file.loc[segBoundary_start:segBoundary_end,'left_hind_claw_x'])**2+(Skeleton_file.loc[segBoundary_start:segBoundary_end,'left_front_claw_y']-Skeleton_file.loc[segBoundary_start:segBoundary_end,'left_hind_claw_y'])**2))
            #distance2 = np.array(np.sqrt((Skeleton_file.loc[segBoundary_start:segBoundary_end,'right_front_claw_x']-Skeleton_file.loc[segBoundary_start:segBoundary_end,'right_hind_claw_x'])**2+(Skeleton_file.loc[segBoundary_start:segBoundary_end,'right_front_claw_y']-Skeleton_file.loc[segBoundary_start:segBoundary_end,'right_hind_claw_y'])**2))
            #distance3 = np.array(np.sqrt((Skeleton_file.loc[segBoundary_start:segBoundary_end,'left_front_claw_x']-Skeleton_file.loc[segBoundary_start:segBoundary_end,'right_hind_claw_x'])**2+(Skeleton_file.loc[segBoundary_start:segBoundary_end,'left_front_claw_y']-Skeleton_file.loc[segBoundary_start:segBoundary_end,'right_hind_claw_y'])**2))
            
            #fig = plt.figure(figsize=(10,10),constrained_layout=True,dpi=300)
            #ax = fig.add_subplot(1, 1, 1)
            #ax.plot(distance1)

            df_distance1 = calculate_GapAndValue('distance1',distance1,mv)
            df_distance1['group'] = dataset_name
            if df_distance1.shape[0] > 0:
                num += 1
                df_distance1['bouts'] = num
                df_list.append(df_distance1)

for key in list(dataset2['video_index']):
    FeA_file = pd.read_csv(Feature_space_path[key])
    FeA_file = FeA_file[FeA_file['revised_movement_label'].isin(locomotion_list)]
    Skeleton_file = pd.read_csv(Skeleton_path[key])
    for mv in locomotion_list:#FeA_file['movement_label'].unique(): #,'left_turning','right_turning'
        singleLocomotionFeA = FeA_file[FeA_file['revised_movement_label']==mv]
        num = 0
        for i in singleLocomotionFeA.index:
            time_sequence = [0]
            value = [0]
            segBoundary_start = singleLocomotionFeA.loc[i,'segBoundary_start']
            segBoundary_end = singleLocomotionFeA.loc[i,'segBoundary_end']
            frame_length = singleLocomotionFeA.loc[i,'frame_length']
            distance1 = np.array(np.sqrt((Skeleton_file.loc[segBoundary_start:segBoundary_end,'left_front_claw_x']-Skeleton_file.loc[segBoundary_start:segBoundary_end,'left_hind_claw_x'])**2+(Skeleton_file.loc[segBoundary_start:segBoundary_end,'left_front_claw_y']-Skeleton_file.loc[segBoundary_start:segBoundary_end,'left_hind_claw_y'])**2))
            #distance2 = np.array(np.sqrt((Skeleton_file.loc[segBoundary_start:segBoundary_end,'right_front_claw_x']-Skeleton_file.loc[segBoundary_start:segBoundary_end,'right_hind_claw_x'])**2+(Skeleton_file.loc[segBoundary_start:segBoundary_end,'right_front_claw_y']-Skeleton_file.loc[segBoundary_start:segBoundary_end,'right_hind_claw_y'])**2))
            #distance3 = np.array(np.sqrt((Skeleton_file.loc[segBoundary_start:segBoundary_end,'left_front_claw_x']-Skeleton_file.loc[segBoundary_start:segBoundary_end,'right_hind_claw_x'])**2+(Skeleton_file.loc[segBoundary_start:segBoundary_end,'left_front_claw_y']-Skeleton_file.loc[segBoundary_start:segBoundary_end,'right_hind_claw_y'])**2))
            
            #fig = plt.figure(figsize=(10,10),constrained_layout=True,dpi=300)
            #ax = fig.add_subplot(1, 1, 1)
            #ax.plot(distance1)

            df_distance1 = calculate_GapAndValue('distance1',distance1,mv)
            df_distance1['group'] = dataset_name2
            if df_distance1.shape[0] > 0:
                num += 1
                df_distance1['bouts'] = num
                df_list.append(df_distance1)


all_df = pd.concat(df_list,axis=0)
all_df.reset_index(drop=True,inplace=True)
all_df.to_csv('{}/{}_{}_strike_info_all.csv'.format(output_dir,dataset_name,dataset_name2))

plot_curve(all_df,dataset_name)
t2 = time.time()
print('Time comsue:{:.2f} min \n'.format((t2-t1)/60))


                

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 09:53:52 2024

@author: Jialin Ye
@institution: SIAT
@Contact_email: jl.ye@siat.ac.cn

"""

import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm
import scipy.signal as signal
import statsmodels.api as sm
from scipy.stats import gaussian_kde
import joypy
from random import sample

revisedMovement_path_dir = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label"
orignial_data_path_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\01_BehaviorAtlas_collated_data'
output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure2_skeleton_kinematics(related to sFigure2F)\movement_parametter\movement_para_distribution'


def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-')) & (file_name.endswith(content)):
            video_index = int(file_name.split('-')[1])
            file_path_dict.setdefault(video_index,file_dir+'\\'+file_name)
    return(file_path_dict)

Movement_Label_path = get_path(revisedMovement_path_dir,'Movement_Labels.csv')       
normalized_ske_path = get_path(orignial_data_path_dir,'Cali_Data3d.csv')  
origin_ske_path = get_path(orignial_data_path_dir,'normalized_skeleton_XYZ.csv')

skip_file_list = [1,3,28,29,110,122] 
animal_info_csv = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Table_S1_animal_information.csv'
animal_info = pd.read_csv(animal_info_csv)
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)]


Morning_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Morning') & (animal_info['LightingCondition']=='Light-on')]
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-off')]

Stress_info = animal_info[(animal_info['ExperimentCondition']=='Stress')]

movement_order = ['running','trotting','walking','right_turning','left_turning','stepping',
                  'rising','hunching','rearing','climbing','jumping','sniffing','grooming','scratching','pausing',]
  
locomotion = ['running','trotting','walking','stepping']  #,'right_turning','left_turning',


def get_color(name):
    if name.startswith('Morning'):
        color = '#F5B25E'
    elif name.startswith('Afternoon'):
        color = '#936736'
    elif name.startswith('Night_lightOn'):
        color = '#398FCB'
    elif name.startswith('Night_lightOff'):
        color = '#003960'
    elif name.startswith('Stress'):
        color = '#f55e6f'
    else:
        print(name)
    
    return(color)

### smooth data
def flitter(df):
    df_temp = df.copy()
    arr1 = np.array(df_temp['back_z'])
    X1 = signal.medfilt(arr1,kernel_size=31)
    
    arr2 = np.array(df_temp['nose_z'])
    X2 = signal.medfilt(arr2,kernel_size=31)
    
    df_temp['smooth_back_z'] = X1
    df_temp['smooth_nose_z'] = X2
    return(df_temp)
        
def reshape_df(df,bodypart):
    mv_height = {}
    for mv in locomotion:   
        temp_df = df[df['revised_movement_label']==mv]
        print(temp_df)
        values = sample(list(temp_df[bodypart].values),10)
        mv_height.setdefault(mv,values)    
    df_out = pd.DataFrame(mv_height)
    return(df_out)
    

def calcualte_parameter_distribution(dataset1,dataset2,dataset1_name,dataset2_name,dataset1_color,dataset2_color,count_variable):

    Mov_list = []
    
    for index in dataset1.index:
        video_index = dataset1.loc[index,'video_index']
        ExperimentCondition = dataset1.loc[index,'ExperimentCondition']
        LightingCondition = dataset1.loc[index,'LightingCondition']
        Mov = pd.read_csv(Movement_Label_path[video_index],usecols=['revised_movement_label','smooth_speed'])
        Ske = pd.read_csv(origin_ske_path[video_index],usecols=['back_z','nose_z'])
        Nor_Ske = pd.read_csv(normalized_ske_path[video_index],usecols=['back_z','nose_z'])
        
        combine_df = pd.concat([Mov,Nor_Ske],axis=1)        
        combine_df = flitter(combine_df)
        
        #mv_height_df = reshape_df(combine_df,'nose_z') 
        calculated_data = combine_df[combine_df['revised_movement_label'].isin(locomotion)]
        calculated_data = calculated_data[['revised_movement_label',count_variable]]
        
        calculated_data.rename(columns={count_variable:dataset1_name},inplace=True)
        Mov_list.append(calculated_data)
    
    
    for index in dataset2.index:
        video_index = dataset2.loc[index,'video_index']
        ExperimentCondition = dataset2.loc[index,'ExperimentCondition']
        LightingCondition = dataset2.loc[index,'LightingCondition']
        Mov = pd.read_csv(Movement_Label_path[video_index],usecols=['revised_movement_label','smooth_speed'])
        Ske = pd.read_csv(origin_ske_path[video_index],usecols=['back_z','nose_z'])
        Nor_Ske = pd.read_csv(normalized_ske_path[video_index],usecols=['back_z','nose_z'])
        
        combine_df = pd.concat([Mov,Nor_Ske],axis=1)        
        combine_df = flitter(combine_df)
        
        #mv_height_df = reshape_df(combine_df,'nose_z') 
        calculated_data = combine_df[combine_df['revised_movement_label'].isin(locomotion)]
        calculated_data = calculated_data[['revised_movement_label',count_variable]]
        calculated_data.rename(columns={count_variable:dataset2_name},inplace=True)
        Mov_list.append(calculated_data)
    
    
    calculated_data_df = pd.concat(Mov_list,axis=0)
    
    if count_variable == 'smooth_speed':
        xrange = [0,501]
    elif count_variable == 'smooth_nose_z':
        xrange = [0,51]
    elif count_variable == 'smooth_back_z':
        xrange = [0,51]
    
    fig, axes = joypy.joyplot(calculated_data_df,by='revised_movement_label',column=["Morning_lightOn","Stress"],
                              figsize=(7,10),x_range=xrange,alpha=0.7,color=[dataset1_color,dataset2_color],
                              lw=5,linecolor="black",ylabels=False,xlabels=False)
    
    return(calculated_data_df)




dataset1 = Morning_lightOn_info
dataset1_name = 'Morning_lightOn'
dataset1_color = get_color(dataset1_name)

dataset2 = Stress_info
dataset2_name ='Stress'
dataset2_color = get_color(dataset2_name)


for variable in ['smooth_speed','smooth_nose_z','smooth_back_z']:
    count_variable = variable                                        ### smooth_speed,smooth_nose_z,smooth_back_z
    para_df = calcualte_parameter_distribution(dataset1,dataset2,dataset1_name,dataset2_name,dataset1_color,dataset2_color,count_variable)
    
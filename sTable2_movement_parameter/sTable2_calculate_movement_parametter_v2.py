# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 09:22:15 2024

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
from scipy.stats import bootstrap



#### calculate parameters

para_dir_path = r'F:\spontaneous_behavior\GitHub\upload_raw_data_all\01_BehaviorAtlas_data'
mov_dir_path = r"F:\spontaneous_behavior\GitHub\spontaneous_behavior_analysis\02_revised_movement_label"
normalized_ske = r'F:\spontaneous_behavior\GitHub\spontaneous_behavior_analysis\01_BehaviorAtlas_collated_data'
output_dir = r'F:\spontaneous_behavior\GitHub\spontaneous_behavior_analysis\sTable2'
if not os.path.exists(output_dir):                                                                       
    os.mkdir(output_dir)

skip_file_list = [1,3,28,29,110,122] 

def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-'))&(file_name.endswith(content)):
            USN = int(file_name.split('-')[1])
            #date = i.split('-')[3][0:8]
            #file_name = 'rec-{0}-G1-{1}'.format(USN,date)
            file_path_dict.setdefault(USN,file_dir+'\\'+file_name)
    return(file_path_dict)


Movement_Label_path = get_path(mov_dir_path,'revised_Movement_Labels.csv')
Parameters_path = get_path(para_dir_path,'para-eachFrames.csv')
normalized_ske_path = get_path(normalized_ske,'normalized_skeleton_XYZ.csv')

animal_info_csv = r'F:\spontaneous_behavior\GitHub\spontaneous_behavior_analysis\Table_S1_animal_information.csv' 
animal_info = pd.read_csv(animal_info_csv)
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)]


Morning_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Morning') & (animal_info['LightingCondition']=='Light-on')]
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-off')]

Stress_info = animal_info[(animal_info['ExperimentCondition']=='Stress')]

movement_order = ['running','trotting','left_turning','right_turning','walking','stepping','sniffing',
                  'rising','hunching','rearing','climbing','jumping','grooming','scratching','pausing',]


dataset = Night_lightOff_info
dataset_name = 'Night_Light-off'       #Morning_Light-on

output_dir = output_dir + '/' + dataset_name
if not os.path.exists(output_dir):                                                                       
    os.mkdir(output_dir)


para_data_list = []
for video_index in list(dataset['video_index']):
#for index in Morning_lightOn_info.index:
    
    para_data = pd.read_csv(Parameters_path[video_index])
    Mov_data = pd.read_csv(Movement_Label_path[video_index],usecols=['original_label','revised_movement_label'])
    Ske_data = pd.read_csv(normalized_ske_path[video_index])
    if Ske_data.shape[1] == 49:
        Ske_data = Ske_data.iloc[:,1:]   
    conbime_data = pd.concat([Mov_data,para_data,Ske_data],axis=1)    
    para_data_list.append(conbime_data)

all_df = pd.concat(para_data_list,axis=0)
all_df.reset_index(drop=True,inplace=True)


calculate_column = ['speed', 'distance', 'Nose_back_tail_2Dangle', 'Nose_back_tail_3Dangle', 'nose_height','neck_height', 'back_height', 'left_front_claw_height', 'right_front_claw_height', 'body_length2D', 'body_length3D',]
momentum_col = ['nose_momentum', 'head_momentum', 'claw_momentum']
calcualted_mv = 'revised_movement_label'      #revised_movement_label   original_label

		







def calculate_momentum(df):
    df_copy = df.copy()
    df_copy.reset_index(drop=True,inplace=True)
    
    nose_x_average = df_copy['nose_x'].mean()
    nose_y_average = df_copy['nose_y'].mean()
    nose_z_average = df_copy['nose_z'].mean()
    
    left_ear_x_average = df_copy['left_ear_x'].mean()
    left_ear_y_average = df_copy['left_ear_y'].mean()
    left_ear_z_average = df_copy['left_ear_z'].mean()
    
    right_ear_x_average = df_copy['right_ear_x'].mean()
    right_ear_y_average = df_copy['right_ear_y'].mean()
    right_ear_z_average = df_copy['right_ear_z'].mean()
    
    left_front_claw_x_average = df_copy['left_front_claw_x'].mean()
    left_front_claw_y_average = df_copy['left_front_claw_y'].mean()
    left_front_claw_z_average = df_copy['left_front_claw_z'].mean()
    
    right_front_claw_x_average = df_copy['right_front_claw_x'].mean()
    right_front_claw_y_average = df_copy['right_front_claw_y'].mean()
    right_front_claw_z_average = df_copy['right_front_claw_z'].mean()
    
    left_hind_claw_x_average = df_copy['left_hind_claw_x'].mean()
    left_hind_claw_y_average = df_copy['left_hind_claw_y'].mean()
    left_hind_claw_z_average = df_copy['left_hind_claw_z'].mean()
    
    right_hind_claw_x_average = df_copy['right_hind_claw_x'].mean()
    right_hind_claw_y_average = df_copy['right_hind_claw_y'].mean()
    right_hind_claw_z_average = df_copy['right_hind_claw_z'].mean()
    
    
    df_copy['nose_momentum'] =  np.sqrt(np.square(df_copy['nose_x']-nose_x_average)+np.square(df_copy['nose_y']-nose_y_average)+np.square(df_copy['nose_z']-nose_z_average))
    df_copy['left_ear_momentum'] =  np.sqrt(np.square(df_copy['left_ear_x']-left_ear_x_average)+np.square(df_copy['left_ear_y']-left_ear_y_average)+np.square(df_copy['left_ear_z']-left_ear_z_average))
    df_copy['right_ear_momentum'] =  np.sqrt(np.square(df_copy['right_ear_x']-right_ear_x_average)+np.square(df_copy['right_ear_y']-right_ear_y_average)+np.square(df_copy['right_ear_z']-right_ear_z_average))
    df_copy['left_front_claw_momentum'] =  np.sqrt(np.square(df_copy['left_front_claw_x']-left_front_claw_x_average)+np.square(df_copy['left_front_claw_y']-left_front_claw_y_average)+np.square(df_copy['left_front_claw_z']-left_front_claw_z_average))
    df_copy['right_front_claw_momentum'] =  np.sqrt(np.square(df_copy['right_front_claw_x']-right_front_claw_x_average)+np.square(df_copy['right_front_claw_y']-right_front_claw_y_average)+np.square(df_copy['right_front_claw_z']-right_front_claw_z_average))
    df_copy['left_hind_claw_momentum'] =  np.sqrt(np.square(df_copy['left_hind_claw_x']-left_hind_claw_x_average)+np.square(df_copy['left_hind_claw_y']-left_hind_claw_y_average)+np.square(df_copy['left_hind_claw_z']-left_hind_claw_z_average))
    df_copy['right_hind_claw_momentum'] =  np.sqrt(np.square(df_copy['right_hind_claw_x']-right_hind_claw_x_average)+np.square(df_copy['right_hind_claw_y']-right_hind_claw_y_average)+np.square(df_copy['right_hind_claw_z']-right_hind_claw_z_average))
    
    df_copy['head_momentum'] = df_copy['nose_momentum'] + df_copy['left_ear_momentum'] + df_copy['right_ear_momentum']
    df_copy['front_claw_momentum'] = df_copy['left_front_claw_momentum'] + df_copy['right_front_claw_momentum']
    df_copy['hind_claw_momentum'] = df_copy['left_hind_claw_momentum'] + df_copy['right_hind_claw_momentum'] 
    
    return(df_copy[['nose_momentum','head_momentum']])#,'front_claw_momentum','hind_claw_momentum']])

df_parameters = pd.DataFrame()


duration_csv = pd.read_csv(r'F:\spontaneous_behavior\GitHub\spontaneous_behavior_analysis\Figure1&Figure3_behavioral_characteristics(related to sFigure3-5)\movement_statistic\movement_statistic.csv',usecols=['group','movement_label','movement_duration','movement_intervels'])
duration_csv = duration_csv[(duration_csv['movement_duration']>0) & (duration_csv['movement_intervels']>0)]


num = 0
for mv in movement_order:                                                         #all_df[calcualted_mv].unique():
    temp_df = all_df[all_df[calcualted_mv]==mv]
    if len(temp_df) > 0:
        temp_duration = duration_csv[(duration_csv['group']==dataset_name)&(duration_csv['movement_label']==mv)]
        for c in calculate_column:
            df_parameters.loc[num,calcualted_mv] = mv
            arr = np.array(temp_df[c])
            
            df_parameters.loc[num,'{}_mean'.format(c)] = round(np.mean(arr),2)
            df_parameters.loc[num,'{}_median'.format(c)] = round(np.median(arr),2)
            df_parameters.loc[num,'{}_std'.format(c)] = round(np.std(arr),2)
            df_parameters.loc[num,'{}_range'.format(c)] = str([round(np.percentile(arr,5),2),round(np.percentile(arr,95),2)])
            
        momentum = calculate_momentum(temp_df.iloc[:,-49:])
        
        for c in momentum.columns:
            arr = np.array(momentum[c])
            
            df_parameters.loc[num,'{}_mean'.format(c)] = round(np.mean(arr),2)
            df_parameters.loc[num,'{}_median'.format(c)] = round(np.median(arr),2)
            df_parameters.loc[num,'{}_std'.format(c)] = round(np.std(arr),2)
            df_parameters.loc[num,'{}_range'.format(c)] = str([round(np.percentile(arr,5),2),round(np.percentile(arr,95),2)])
        
        for c in temp_duration.columns[1:-1]:
            arr = np.array(temp_duration[c])
            
            df_parameters.loc[num,'{}_mean'.format(c)] = round(np.mean(arr),2)
            df_parameters.loc[num,'{}_median'.format(c)] = round(np.median(arr),2)
            df_parameters.loc[num,'{}_std'.format(c)] = round(np.std(arr),2)
            df_parameters.loc[num,'{}_range'.format(c)] = str([round(np.percentile(arr,5),2),round(np.percentile(arr,95),2)])
        
        num += 1
    else:
        temp_duration = duration_csv[(duration_csv['group']==dataset_name)&(duration_csv['movement_label']==mv)]
        for c in calculate_column:
            df_parameters.loc[num,calcualted_mv] = mv
            arr = np.array(temp_df[c])
            
            df_parameters.loc[num,'{}_mean'.format(c)] = np.nan
            df_parameters.loc[num,'{}_median'.format(c)] =  np.nan
            df_parameters.loc[num,'{}_std'.format(c)] =  np.nan
            df_parameters.loc[num,'{}_range'.format(c)] =  np.nan
            
        momentum = ['nose_momentum','head_momentum']
        
        for c in momentum:
            
            df_parameters.loc[num,'{}_mean'.format(c)] = np.nan
            df_parameters.loc[num,'{}_median'.format(c)] = np.nan
            df_parameters.loc[num,'{}_std'.format(c)] = np.nan
            df_parameters.loc[num,'{}_range'.format(c)] = np.nan
        
        for c in temp_duration.columns[1:]:
            arr = np.array(temp_duration[c])
            
            df_parameters.loc[num,'{}_mean'.format(c)] = np.nan
            df_parameters.loc[num,'{}_median'.format(c)] = np.nan
            df_parameters.loc[num,'{}_std'.format(c)] = np.nan
            df_parameters.loc[num,'{}_range'.format(c)] = np.nan
        
        num += 1
        
df_parameters.to_csv('{}\{}_{}_pose&kinematis_parameters4.csv'.format(output_dir,dataset_name,calcualted_mv),index=None)





















































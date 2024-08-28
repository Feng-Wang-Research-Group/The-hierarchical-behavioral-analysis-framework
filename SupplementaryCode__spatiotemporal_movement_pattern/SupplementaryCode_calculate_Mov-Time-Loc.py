#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 16:25:05 2022

@author: Jialin Ye
@institution: SIAT
@Contact_email: jl.ye@siat.ac.cn

"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats


InputData_path_dir = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label"                      
output_dir = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\SupplementaryCode_spatiotemporal_movement_pattern" 

if not os.path.exists(output_dir):                                                                     
    os.mkdir(output_dir)

skip_file_list = [1,3,28,29,110,122]                                                       

count_variable = 'revised_movement_label'               ### 'movement_label'                                                


time_window1 = 0
time_window2 = 60                                             


movement_order = ['running','trotting','left_turning','right_turning','walking','stepping','sniffing',
                  'rising','hunching','rearing','climbing','jumping','grooming','scratching','pausing',]

cluster_order = ['locomotion','exploration','maintenance','nap']                                 
     

#### 分组信息

animal_info_csv = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Table_S1_animal_information.csv'              
animal_info = pd.read_csv(animal_info_csv)
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)]

Morning_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Morning') & (animal_info['LightingCondition']=='Light-on')]
Morning_lightOn_info_male = Morning_lightOn_info[Morning_lightOn_info['gender']=='male']
Morning_lightOn_info_female = Morning_lightOn_info[Morning_lightOn_info['gender']=='female']


Afternoon_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOff_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-off')]
Night_lightOff_info_male = Night_lightOff_info[Night_lightOff_info['gender']=='male']
Night_lightOff_info_female = Night_lightOff_info[Night_lightOff_info['gender']=='female']

stress_animal_info = animal_info[animal_info['ExperimentCondition']=='Stress']

stress_animal_info_male = stress_animal_info[stress_animal_info['gender']=='male']
stress_animal_info_female = stress_animal_info[stress_animal_info['gender']=='female']

dataset = Afternoon_lightOn_info
dataset_name = 'Afternoon_lightOn'


variables = ['gender','ExperimentCondition','LightingCondition']                                                                                      

def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-')) & (file_name.endswith(content)):
            video_index = int(file_name.split('-')[1])
            #date = i.split('-')[3][0:8]
            #file_name = 'rec-{0}-G1-{1}'.format(video_index,date)
            file_path_dict.setdefault(video_index,file_dir+'\\'+file_name)
    return(file_path_dict)

Movement_Label_path = get_path(InputData_path_dir,'revised_Movement_Labels.csv')


def add_location(Mov_loc_data_copy,boundary):
    Mov_loc_data_copy['data_driven_location'] = 'perimeter'   
    Mov_loc_data_copy.loc[(Mov_loc_data_copy['back_x']>(250-boundary))&(Mov_loc_data_copy['back_x']<(250+boundary))&(Mov_loc_data_copy['back_y']>(250-boundary))&(Mov_loc_data_copy['back_y']<(250+boundary)),'data_driven_location'] = 'center'
    Mov_loc_data_copy.loc[(Mov_loc_data_copy['back_x']<=(250-boundary))&(Mov_loc_data_copy['back_y']<=(250-boundary)),'data_driven_location'] = 'corner'
    Mov_loc_data_copy.loc[(Mov_loc_data_copy['back_x']<=(250-boundary))&(Mov_loc_data_copy['back_y']>=(250+boundary)),'data_driven_location'] = 'corner'
    Mov_loc_data_copy.loc[(Mov_loc_data_copy['back_x']>=(250+boundary))&(Mov_loc_data_copy['back_y']>=(250+boundary)),'data_driven_location'] = 'corner'
    Mov_loc_data_copy.loc[(Mov_loc_data_copy['back_x']>=(250+boundary))&(Mov_loc_data_copy['back_y']<=(250-boundary)),'data_driven_location'] = 'corner'
    
    return(Mov_loc_data_copy)

def add_traditional_location(Mov_loc_data,boundary=125):
    Mov_loc_data_copy = Mov_loc_data.copy()
    Mov_loc_data_copy['traditional_location'] = 'perimeter'
    Mov_loc_data_copy.loc[(Mov_loc_data_copy['back_x']>125) & (Mov_loc_data_copy['back_x']<375) & (Mov_loc_data_copy['back_y']>125) & (Mov_loc_data_copy['back_y']<375),'traditional_location'] = 'center'
    return(Mov_loc_data_copy)


def original_label_count(df):
    data = df.copy()
    label_number = data[count_variable].value_counts()

    df_output = pd.DataFrame()
    num = 0
    for mv in range(1,41):
        df_output.loc[num,count_variable] = mv
        if not mv in label_number.index:
            df_output.loc[num,'label_frequency'] = 0
        else:
            df_output.loc[num,'label_frequency'] = label_number[mv] / label_number.values.sum()
        df_output.loc[num,'accumulative_time'] = df_output.loc[num,'label_frequency']*(time_window2-time_window1)
        num += 1
    return(df_output)    

def movement_label_count(df):
    data = df.copy()
    label_number = data[count_variable].value_counts()

    df_output = pd.DataFrame()
    num = 0
    for mv in movement_order:
        df_output.loc[num,count_variable] = mv
        if not mv in label_number.index:
            df_output.loc[num,'label_frequency'] = 0
        else:
            df_output.loc[num,'label_frequency'] = round(label_number[mv] / label_number.values.sum(),2)
        df_output.loc[num,'accumulative_time'] = df_output.loc[num,'label_frequency']*(time_window2-time_window1)
        num += 1
    return(df_output)

def movement_label_count2(df):
    data = df.copy()
    label_number = data[count_variable].value_counts()

    df_output = pd.DataFrame()
    num = 0
    for mv in movement_order:
        df_output.loc[num,count_variable] = mv
        if not mv in label_number.index:
            df_output.loc[num,'label_frequency'] = int(0)
        else:
            df_output.loc[num,'label_frequency'] = label_number[mv]
        num += 1
    return(df_output)


def movement_cluster_label_count(df):
    data = df.copy()
    label_number = data[count_variable].value_counts()

    df_output = pd.DataFrame()
    num = 0
    for mv in cluster_order:
        df_output.loc[num,count_variable] = mv
        if not mv in label_number.index:
            df_output.loc[num,'label_frequency'] = 0
        else:
            df_output.loc[num,'label_frequency'] = label_number[mv] / label_number.values.sum()
        df_output.loc[num,'accumulative_time'] = df_output.loc[num,'label_frequency']*(time_window2-time_window1)
        num += 1
    return(df_output)


mv_sort_order = {}
for i in range(len(movement_order)):
    mv_sort_order.setdefault(movement_order[i],i)

mv_sort_order2 = {}
for i in range(len(cluster_order)):
    mv_sort_order2.setdefault(cluster_order[i],i)


def removeNAN_align(df):
    M = df.to_numpy()
    #get True or False depending on the null status of each entry
    condition = ~np.isnan(M)

    #for each array, get entries that are not null
    step1 = [np.compress(ent,arr) for ent,arr in zip(condition,M)]
    step1

    #concatenate each dataframe 
    step2 = pd.concat([pd.DataFrame(ent).T for ent in step1],ignore_index=True)
    step2.index = df.index
    return(step2)


repeat_list = []
for ID in animal_info['video_index'].unique():
    if len(animal_info[animal_info['video_index']==ID]) >=2:
        repeat_list.append(ID)
print('repeated video index : {}'.format(repeat_list))

movement_frequency_each_mice = []

test_df_list = []
singleMice_Mov_list = []


for i in dataset.index:
    video_index = dataset.loc[i,'video_index']
    
    if video_index in skip_file_list:
        pass
    else:
        mouse_id = animal_info.loc[i,'mouse_id']                                                               
        if count_variable == 'origin_label':
            Movement_Label_file = pd.read_csv(Movement_Label_path[video_index])
            Movement_Label_file = Movement_Label_file.iloc[time_window1*30*60:time_window2*30*60,:]
            df = original_label_count(Movement_Label_file)
      
            for v in variables:
                df[v] = animal_info.loc[i,v]
    
            df['mouse_id'] = mouse_id
            df['order'] = df['origin_label']
            movement_frequency_each_mice.append(df)
            
        elif count_variable == 'revised_movement_label':
            Movement_Label_file = pd.read_csv(Movement_Label_path[video_index])
            #Movement_Label_file = add_movement_label(Movement_Label_file)
            Movement_Label_file = Movement_Label_file.iloc[time_window1*30*60:time_window2*30*60,:]
            Movement_Label_file = add_traditional_location(Movement_Label_file)
            Movement_Label_file = add_location(Movement_Label_file, 175)
            df = movement_label_count2(Movement_Label_file)
            
            singleMice_Mov = pd.DataFrame( [df['label_frequency'].values],columns= df['revised_movement_label'].values,index=[video_index])
            singleMice_Mov_list.append(singleMice_Mov)
            
            Movement_Label_file['frame'] = Movement_Label_file.index
            for loc in Movement_Label_file['data_driven_location'].unique():
                loc_df = Movement_Label_file[ Movement_Label_file['data_driven_location']==loc]
                start = 0
                end = 0
                for min_i in range(5,61,5):
                    end = min_i * 30*60
                    time_df = loc_df[(loc_df['frame']>=start) &(loc_df['frame']<=end)]

                    time_df_count = movement_label_count2(time_df)
                    time_df_count['time'] = min_i
                    time_df_count['location'] = loc
                    time_df_count['mice'] = video_index
                    test_df_list.append(time_df_count)
                    start = end
            df['order'] = df['revised_movement_label'].map(mv_sort_order)
            
            movement_frequency_each_mice.append(df)
            
            
        elif count_variable == 'movement_cluster_label':
            Movement_Label_file = pd.read_csv(Movement_Label_path[video_index])
            #Movement_Label_file = add_movement_label(Movement_Label_file)
            #Movement_Label_file = add_cluster(Movement_Label_file)
            Movement_Label_file = Movement_Label_file.iloc[time_window1*30*60:time_window2*30*60,:]
            df = movement_cluster_label_count(Movement_Label_file)
            for v in variables:
                df[v] = animal_info.loc[i,v]
            df['mouse_id'] = mouse_id
            df['order'] = df['movement_cluster_label'].map(mv_sort_order2)
            movement_frequency_each_mice.append(df)
    

variables.insert(0, count_variable)
variables.insert(-1, 'order')


df_singleMice_Mov = pd.concat(singleMice_Mov_list)
df_singleMice_Mov.to_csv(r'{}\{}_singleMice_Mov.csv'.format(output_dir,dataset_name),index=None)

df_out_test = pd.concat(test_df_list)

df_out_test_tocsv = pd.DataFrame()
num = 0
for loc in ['center','perimeter','corner']:
    loc_df = df_out_test[df_out_test['location']==loc]
    for mv in movement_order:
        loc_mv_df = loc_df[loc_df['revised_movement_label']==mv]

        arr1 = loc_mv_df['time'].values
        arr2 = loc_mv_df['label_frequency'].values
        
        r,p = stats.pearsonr(arr1, arr2)
        if p < 0.05:
            df_out_test_tocsv.loc[num,'location'] = loc
            df_out_test_tocsv.loc[num,'movement'] = mv        
            df_out_test_tocsv.loc[num,'r'] = r
            df_out_test_tocsv.loc[num,'p'] = p
            num += 1
            
df_out_test_tocsv.to_csv(r'{}\{}_Temporal_locMoc.csv'.format(output_dir,dataset_name),index=None)




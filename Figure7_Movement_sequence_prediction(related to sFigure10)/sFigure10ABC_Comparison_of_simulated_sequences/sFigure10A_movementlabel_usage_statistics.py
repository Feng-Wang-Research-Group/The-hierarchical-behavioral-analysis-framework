#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 16:25:05 2022

@author: yejohnny
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



InputData_path_dir1 = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label"                   
Predicted_Movementsequences_day = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure7_Movement_sequence_prediction\predicted_movement_sequences\new_sequence-MOTP_MODP'                    
Predicted_Movementsequences_stress = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure7_Movement_sequence_prediction\predicted_movement_sequences\new_sequence-ASTP_ASDP'

output_dir = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure7_Movement_sequence_prediction\sFigure22_Comparison_of_simulated_sequences" 

if not os.path.exists(output_dir):                                                                     
    os.mkdir(output_dir)

skip_file_list = [1,3,28,29,110,122]                                                        

count_variable = 'revised_movement_label'   ### 'movement_label'                                               


time_window1 = 0
time_window2 = 60                                             
movement_order = ['running','trotting','left_turning','right_turning','walking','stepping','sniffing',
                  'rising','hunching','rearing','climbing','jumping','grooming','scratching','pausing',]

cluster_order = ['locomotion','exploration','maintenance','nap']
     
animal_info_csv = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Table_S1_animal_information.csv' 
animal_info = pd.read_csv(animal_info_csv)
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)]                                                                                                                   

variables = ['gender','ExperimentCondition','LightingCondition']                                                                                      

##################################################################################################################################################


def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-')) & (file_name.endswith(content)):
            video_index = int(file_name.split('-')[1])
            file_path_dict.setdefault(video_index,file_dir+'\\'+file_name)
    return(file_path_dict)

Movement_Label_path = get_path(InputData_path_dir1,'revised_Movement_Labels.csv')

def get_path2(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-')) & (file_name.endswith(content)):
            video_index = file_name.split('-')[1]
            file_path_dict.setdefault(video_index,file_dir+'\\'+file_name)
    return(file_path_dict)

Predicted_Label_path_day = get_path2(Predicted_Movementsequences_day,'Movement_Labels.csv')   
Predicted_Label_path_stress = get_path2(Predicted_Movementsequences_stress,'Movement_Labels.csv') 

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
            df_output.loc[num,'label_frequency'] = (label_number[mv] / label_number.values.sum())*100
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
            df_output.loc[num,'label_frequency'] = (label_number[mv] / label_number.values.sum())*100
        df_output.loc[num,'accumulative_time'] = df_output.loc[num,'label_frequency']*(time_window2-time_window1)
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
            df_output.loc[num,'label_frequency'] =( label_number[mv] / label_number.values.sum())*100
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

Morning_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Morning') & (animal_info['LightingCondition']=='Light-on')]
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-off')]

Stress_info = animal_info[(animal_info['ExperimentCondition']=='Stress')]

for i in animal_info.index:
    video_index = animal_info.loc[i,'video_index']
    
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
            
            df = movement_label_count(Movement_Label_file)

            for v in variables:
                df[v] = animal_info.loc[i,v]
            df['mouse_id'] = mouse_id
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
    

for key in Predicted_Label_path_day.keys():
    mouse_id = key                                                              
    Movement_Label_file = pd.read_csv(Predicted_Label_path_day[key])
    Movement_Label_file = Movement_Label_file.iloc[time_window1*30*60:time_window2*30*60,:]
    df = movement_label_count(Movement_Label_file)
    df['gender'] = 'None'
    df['ExperimentCondition'] = 'predicted_Morning'
    df['LightingCondition'] = 'Light-on'
    df['mouse_id'] = mouse_id
    df['order'] = df['revised_movement_label'].map(mv_sort_order)
    movement_frequency_each_mice.append(df)

for key in Predicted_Label_path_stress.keys():
    mouse_id = key                                                             
    Movement_Label_file = pd.read_csv(Predicted_Label_path_stress[key])
    #Movement_Label_file = add_movement_label(Movement_Label_file)
    Movement_Label_file = Movement_Label_file.iloc[time_window1*30*60:time_window2*30*60,:]
    df = movement_label_count(Movement_Label_file)
    df['gender'] = 'None'
    df['ExperimentCondition'] = 'predicted_stress'
    df['LightingCondition'] = 'Light-on'
    df['mouse_id'] = mouse_id
    df['order'] = df['revised_movement_label'].map(mv_sort_order)
    movement_frequency_each_mice.append(df)


variables.insert(0, count_variable)
variables.insert(-1, 'order')


df_out = pd.concat(movement_frequency_each_mice)

df_out1 = df_out.pivot_table(values='label_frequency', index=variables, columns='mouse_id',)
df_out1.to_csv(r'{}\Full_info_{}_count_{}-{}min_validation.csv'.format(output_dir,count_variable,time_window1, time_window2))
df_out2 = removeNAN_align(df_out1)
df_out2 = df_out2.sort_index(level='order',)
df_out2.to_csv(r'{}\Simplify_{}_count_{}-{}_validation.csv'.format(output_dir,count_variable,time_window1, time_window2))



df_out_morning = df_out[df_out['ExperimentCondition'].isin(['Morning','predicted_Morning'])]
df_out_stress =  df_out[df_out['ExperimentCondition'].isin(['Stress','predicted_stress'])]


fig,ax = plt.subplots(nrows=2,ncols=1, figsize=(10,10),dpi=300,constrained_layout=True)

sns.barplot(data=df_out_morning,x='revised_movement_label',y='label_frequency',order=movement_order,hue='ExperimentCondition',hue_order=['Morning','predicted_Morning'],palette=['#F5B25E','#FFFF00'],ax=ax[0],errorbar=('ci',95))
ax[0].set_xticklabels(ax[0].get_xticklabels(),rotation=60)
ax[0].set_ylabel('probability')
ax[0].legend(loc='upper left')
sns.barplot(data=df_out_stress,x='revised_movement_label',y='label_frequency',order=movement_order,hue='ExperimentCondition',hue_order=['Stress','predicted_stress'],palette=['#f55e6f','#b71726'],ax=ax[1],errorbar=('ci',95))
plt.xticks(rotation=60)
ax[1].set_xticklabels(ax[1].get_xticklabels(),rotation=60)
ax[1].set_ylabel('probability')
ax[1].legend(loc='upper left')

plt.savefig('{}/comparision_of_movement_fractions.png'.format(output_dir),dpi=300,transparent=True)




























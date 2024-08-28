# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:24:39 2023

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
from scipy.stats import norm
import random
from tkinter import _flatten
from scipy import signal
from sklearn.neighbors import KernelDensity
from multiprocessing import Pool
import time
from scipy.stats import gaussian_kde



InputData_path_dir = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure7_Movement_sequence_prediction(related to sFigure10)\predicted_movement_sequences'
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


Movement_Label_path = get_path(InputData_path_dir,'revised_Movement_Labels.csv')
Feature_space_path = get_path(InputData_path_dir,'revised_Feature_Space.csv')

animal_info_csv = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Table_S1_animal_information.csv' 
animal_info = pd.read_csv(animal_info_csv)
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)]


Morning_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Morning') & (animal_info['LightingCondition']=='Light-on')]
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-off')]

Stress_info = animal_info[(animal_info['ExperimentCondition']=='Stress')]

movement_order = ['running','trotting','left_turning','right_turning','walking','stepping','sniffing',
                  'rising','hunching','rearing','climbing','jumping','grooming','scratching','pausing',]

step = 5

Movement_transtion_training_dataset = Morning_lightOn_info
Movement_transtion_training_dataset_name = 'MO'                 ## MO:Morning_lightOn  AO:Afternoon_lightOn, NO:night_lightOn, NF: night_lightOff
Movement_duration_training_dataset = Stress_info
Movement_duration_training_dataset_name = 'AS'

output_dir = output_dir + '/new_sequence-{}TP_{}DP'.format(Movement_transtion_training_dataset_name,Movement_duration_training_dataset_name)
if not os.path.exists(output_dir):                                                                       
    os.mkdir(output_dir)


def countFragementLengthEach(df,time_segement):
    start = 0
    end = 0
    
    df_output = pd.DataFrame()
    num = 0
    for i in range(time_segement,61,time_segement):
        end = i*60*30
        temp_FeA = df[(df['segBoundary_start']>start) & (df['segBoundary_end']<=end)]
        for mv in movement_order:
            temp_FeA_mv = temp_FeA[temp_FeA['revised_movement_label']==mv]
            if len(temp_FeA_mv) > 0:
                for length in temp_FeA_mv['frame_length']:
                    df_output.loc[num,'time'] = i
                    df_output.loc[num,'revised_movement_label'] = mv
                    df_output.loc[num,'duration'] = length
                    num += 1
        start = end
    return(df_output)  


MV_duration_list = []
for i in Movement_duration_training_dataset.index:

    video_index = Movement_duration_training_dataset.loc[i,'video_index']
    ExperimentCondition = Movement_duration_training_dataset.loc[i,'ExperimentCondition']
    gender = Movement_duration_training_dataset.loc[i,'gender']

    FeA_data = pd.read_csv(Feature_space_path[video_index]) 
    
    MV_duration = countFragementLengthEach(FeA_data,1)
    MV_duration_list.append(MV_duration)
     
MV_duration_df = pd.concat(MV_duration_list)                                                         ### obtain  per minutes MV duration probabilities
MV_duration_df['time_stage'] = (((MV_duration_df['time']-1) // step)+1) * step





rng = np.random.default_rng()


def find_indices(list_to_check,item_to_find):
    return([idx for idx, value in enumerate(list_to_check) if value == item_to_find])

def count_percentage(label_assemble):
    df = pd.DataFrame()
    num = 0
    for key in movement_order:
        count = label_assemble.count(key)
        df.loc[num,'later_label'] = key
        df.loc[num,'count'] = count
        num += 1
    df['percentage'] = df['count']/df['count'].sum()
    return(df)

def rate_random(data_list,rate_list):
    start = 0
    random_num = random.random()
    for idx,score in enumerate(rate_list):
        start += score
        if random_num <= start:
            break
    return(data_list[idx])

def count_duration_rate(ci_l, ci_u,arr1):
    se1 = pd.cut(arr1, np.arange(ci_l, ci_u,((ci_u-ci_l)/10)))
    count = se1.value_counts()
    
    df_out = pd.DataFrame()
    num = 0
    for k in count.index:
        value1 = float(str(k).strip('(').split(',')[0])
        value2 = float(str(k).strip(']').split(',')[1])
        average_value = (value1+value2)/2
        
        value = count[k] 
        
        df_out.loc[num,'frame_length'] = average_value
        df_out.loc[num,'count'] = value
        df_out.loc[num,'probility'] = value / len(arr1)
        
        num += 1

    return(df_out)
    
        
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def perturb(x, amount):
    value = x + random.uniform(-amount, amount)
    return(value)
    #value = 0
    #while value <= 0:
    #    value = x + random.uniform(-amount, amount)
    #    if value > 0:
    #        return(value)

def bootstrap_(arr):
    resample_list = []
    for j in range(1000):
        resample_value = np.random.choice(arr, 1)
        resample_value = perturb(resample_value, np.std(arr))                    ## add  perturbation
        resample_list.append(resample_value[0])
    return(resample_list)


def get_mv_duration(movement_duration_data,time_stage,mv):
    time_mv_df = movement_duration_data[(movement_duration_data['time_stage']==time_stage)&(movement_duration_data['revised_movement_label']==mv)]
    arr = np.array(time_mv_df['duration'])
    if len(arr) <= 1 or arr.sum() == 0:
        duration_length = 1
    elif len(set(list(arr))) == 1:
        duration_length = int(np.mean(arr))
    else:
        resample_list = bootstrap_(arr)
        data = np.array(resample_list)
        x_plot = np.linspace(min(data), max(data), 100)
        kde = gaussian_kde(data, bw_method='scott')
        dens = kde(x_plot)/np.sum(kde(x_plot))

        duration_length = int(perturb(rate_random(x_plot,dens),np.std(data)))
        

        if duration_length <1:
            duration_length = 1

    return(duration_length)


def get_next_mv(time_trans_df,time_stage,previous_label):
    trans_df = time_trans_df[ (time_trans_df['time']==time_stage) &  (time_trans_df['previous_label']==previous_label)]
    data_list = list(trans_df['later_label'])
    rate_list =  list(trans_df['percentage'])
    next_mv = rate_random(data_list,rate_list)
    return(next_mv)


def MoV2FeA(df):
    MoV_file = df.copy()
    FeA_file = {'revised_movement_label':[],'segBoundary':[],'segBoundary_start':[],'segBoundary_end':[],'length':[]}
    last_boundary = 0
    for pos in range(1,MoV_file.shape[0]):
        x1 = pos -1
        x2 = pos
        movement_label1 = MoV_file.loc[x1,'revised_movement_label']
        movement_label2 = MoV_file.loc[x2,'revised_movement_label']
        if x2 != MoV_file.shape[0]-1:
            if movement_label1 != movement_label2:
                
                FeA_file['revised_movement_label'].append(movement_label1)
                FeA_file['segBoundary'].append(x1+1)
                FeA_file['segBoundary_start'].append(last_boundary)
                FeA_file['segBoundary_end'].append(x1+1)
                FeA_file['length'].append(x1+1-last_boundary)
                last_boundary = x1+1
        else:
            FeA_file['revised_movement_label'].append(movement_label2)
            FeA_file['segBoundary'].append(x2+1)
            FeA_file['segBoundary_start'].append(last_boundary)
            FeA_file['segBoundary_end'].append(x2+1)
            FeA_file['length'].append(x2+1-last_boundary)
    FeA_df = pd.DataFrame(FeA_file)
    return(FeA_df)

def calculate_MV_trans(sentences,time_tag):
    df_list = []
    for key in movement_order:
        label_behind_key_sum = []
        
        for sentence in sentences:
            if key in sentence:
                index_ID = np.array(find_indices(sentence,key))
                if len(sentence)-1 in index_ID:
                    index_ID = np.delete(index_ID, find_indices(index_ID,len(sentence)-1)[0])     ### next movement
        
                    #del index_ID[find_indices(index_ID,len(sentence)-1)[0]]                      ### previous movement 
                label_behind_key = np.array(sentence)[index_ID+1]
            else:
                label_behind_key = ['']
            
            #index_ID = np.array(find_indices(sentence,key))
            #if 0 in index_ID:
            #    index_ID = np.delete(index_ID, find_indices(index_ID,0)[0])
            #label_before_key = np.array(sentence)[index_ID-1]
            
            label_behind_key_sum.extend(label_behind_key)
        df_count = count_percentage(label_behind_key_sum)
        df_count['previous_label'] = key
        df_count = df_count[['previous_label','later_label','count','percentage']]
        df_count['time'] = time_tag
        df_list.append(df_count)
    all_df = pd.concat(df_list)
    all_df.reset_index(drop=True,inplace=True)
    return(all_df)

start = 0
end = 0
num = 0

first_MV = []
temporal_trans_list = []

for i in range(step,61,step):
    sentences = []
    end = i * 30 * 60
    for key in Movement_transtion_training_dataset['video_index']:
        
        df = pd.read_csv(Feature_space_path[key])
        temp_df = df[ (df['segBoundary_start']>=start) & (df['segBoundary_end']<end)]
        if i == step:
            first_MV.append(df.loc[0,'revised_movement_label'])
        sentences.append(list(temp_df['revised_movement_label'].values))
    MVtrans_df = calculate_MV_trans(sentences,i)
    temporal_trans_list.append(MVtrans_df)
    start=end

temporal_trans_df = pd.concat(temporal_trans_list)
temporal_trans_df.reset_index(drop=True,inplace=True)                                           ### obtain per minutes MV transition probabilities



def processing_data(i):
#for i in range(10):
    first_stage_mv = []
    first_stage_rate = []
    for mv in  set(_flatten(first_MV)):
        first_stage_mv.append(mv)
        first_stage_rate.append(list(_flatten(first_MV)).count(mv)/len(list(_flatten(first_MV))))
    
    mv_list = []
    
    first_mv = rate_random(first_stage_mv,first_stage_rate)
    mv_list.extend([first_mv]*get_mv_duration(MV_duration_df,step,first_mv))
    
    while len(mv_list) < 60*30*60:
        time_stage = ((len(mv_list) // (step*60*30)) + 1) * step
        previous_label = mv_list[-1]
        next_mv = get_next_mv(temporal_trans_df,time_stage,previous_label)
        mv_list.extend([next_mv]*get_mv_duration(MV_duration_df,time_stage,next_mv))
    df_output = pd.DataFrame(data=mv_list[:30*60*60],index=range(30*60*60),columns=['revised_movement_label'])
    df_output.to_csv(r'{}\rec-{}-new{}{}_Movement_Labels.csv'.format(output_dir,i,Movement_transtion_training_dataset_name,Movement_duration_training_dataset_name),index=None)
    FeA_df = MoV2FeA(df_output)
    FeA_df.to_csv(r'{}\rec-{}-new{}{}_Feature_Space.csv'.format(output_dir,i,Movement_transtion_training_dataset_name,Movement_duration_training_dataset_name),index=None)


def main():
    t1 = time.time()
    p = Pool(16)
    p.map(processing_data,range(50))                           ### generate 50 1-h long stimulated movement sequences
    p.close()
    p.join()
    t2 = time.time()
    print('Time comsue:{:.2f} min \n'.format((t2-t1)/60))

if __name__=='__main__':
    main()


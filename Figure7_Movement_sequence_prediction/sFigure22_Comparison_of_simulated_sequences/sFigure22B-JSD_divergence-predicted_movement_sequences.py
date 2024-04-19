# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 22:10:48 2023

@author: Jialin Ye
@institution: SIAT
@Contact_email: jl.ye@siat.ac.cn

"""

import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats


InputData_path_dir = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label"
InputData_path_dir_predict_MO = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure7_Movement_sequence_prediction\predicted_movement_sequences\new_sequence-MOTP_MODP'
InputData_path_dir_predict_AS = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure7_Movement_sequence_prediction\predicted_movement_sequences2\new_sequence-ASTP_ASDP'

output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure7_Movement_sequence_prediction\sFigure22_Comparison_of_simulated_sequences'
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

def get_path2(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-'))&(file_name.endswith(content)):
            USN = file_name.split('-')[1]
            #date = i.split('-')[3][0:8]
            #file_name = 'rec-{0}-G1-{1}'.format(USN,date)
            file_path_dict.setdefault(USN,file_dir+'\\'+file_name)
    return(file_path_dict)

Movement_Label_path = get_path(InputData_path_dir,'revised_Movement_Labels.csv')

Movement_Label_path_predict_MO = get_path2(InputData_path_dir_predict_MO,'Movement_Labels.csv')
Movement_Label_path_predict_AS = get_path2(InputData_path_dir_predict_AS,'Movement_Labels.csv')

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



def JS_divergence(p,q):
    M=(p+q)/2
    return (0.5*scipy.stats.entropy(p, M, base=2)+0.5*scipy.stats.entropy(q, M, base=2))


def movement_count(df,start,end):
    if end > start:
        temp_df = df.iloc[start*60*30:end*60*30,:]
        label_count = temp_df.value_counts('revised_movement_label')
    
        df = pd.DataFrame(data=0,index=movement_order,columns=['count'])
        for i in label_count.index:
            df.loc[i,'count'] =  label_count[i]
    
    return(df)

start = 0
end = 60


dataset1 = Morning_lightOn_info
dataset2 = Stress_info

dataset_name1 = 'Morning_lightOn'
dataset_name2 = 'Stress'


dataset3 = Movement_Label_path_predict_MO
dataset4 = Movement_Label_path_predict_AS

dataset_name3 = 'predict_Morning'
dataset_name4 = 'predict_stress'

array1 = []
array2 = []
all_array = []


for video_index in dataset1['video_index']:
    if video_index in skip_file_list:
        pass
    else:
        MoV_file = pd.read_csv(Movement_Label_path[video_index])
        MoV_count = movement_count(MoV_file,start,end)
        array1.append(MoV_count['count'].values)
        all_array.append(MoV_count['count'].values)

for video_index in dataset3.keys():
    MoV_file = pd.read_csv(Movement_Label_path_predict_MO[video_index])
    MoV_count = movement_count(MoV_file,start,end)
    #array1.append(MoV_count['count'].values)
    all_array.append(MoV_count['count'].values)

for video_index in dataset2['video_index']:
    if video_index in skip_file_list:
        pass
    else:
        MoV_file = pd.read_csv(Movement_Label_path[video_index])
        MoV_count = movement_count(MoV_file,start,end)
        array2.append(MoV_count['count'].values)
        all_array.append(MoV_count['count'].values)



for video_index in dataset4.keys():
    MoV_file = pd.read_csv(Movement_Label_path_predict_AS[video_index])
    MoV_count = movement_count(MoV_file,start,end)
    #array2.append(MoV_count['count'].values)
    all_array.append(MoV_count['count'].values)





df_array1 = pd.DataFrame(data=array1,columns=movement_order)
array1_average = df_array1.mean(axis=0).apply(lambda x: round(x)).values
array1_average = array1_average / array1_average.sum()

df_array2 = pd.DataFrame(data=array2,columns=movement_order)
array2_average = df_array2.mean(axis=0).apply(lambda x: round(x)).values
array2_average = array2_average / array2_average.sum()



df = pd.DataFrame(index=range(13),columns=range(13))
for i in range(len(all_array)):
    Q = all_array[i] / all_array[i].sum()
    for j in range(len(all_array)):
        P = all_array[j] / all_array[j].sum()

        df.loc[i,j] = JS_divergence(Q, P)
#df = df.replace([np.inf,-np.inf],2)
df = df.astype('float32')
fig,ax = plt.subplots(nrows=1, ncols=1,figsize=(10,10),dpi=600)
     
im = ax.imshow(df.values,cmap='copper',norm='linear',vmin=0,vmax=0.15)   #copper
plt.xticks([])
plt.yticks([])

plt.savefig('{}/{}&{}_JSD.png'.format(output_dir,dataset_name1,dataset_name2),dpi=300,transparent=True)



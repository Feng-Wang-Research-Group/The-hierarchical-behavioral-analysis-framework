# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 20:21:21 2023

@author: Jialin Ye
@institution: SIAT
@Contact_email: jl.ye@siat.ac.cn

"""

import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import matplotlib
import networkx as nx
from tkinter import _flatten
import scipy.stats
#from itertools import chain
    
    
InputData_path_dir =r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure6_Movement_transition\Movement_transition_Entropy'


skip_file_list = [1,3,28,29,110,122] 

def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-'))&(file_name.endswith(content)):
            USN = int(file_name.split('-')[1])
            file_path_dict.setdefault(USN,file_dir+'\\'+file_name)
    return(file_path_dict)


Movement_Label_path = get_path(InputData_path_dir,'revised_Movement_Labels.csv')
Feature_space_path = get_path(InputData_path_dir,'Feature_Space.csv')
 


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


movement_order = ['running','trotting','walking','right_turning','left_turning','stepping',
                  'rising','hunching','rearing','climbing','jumping','sniffing','grooming','scratching','pausing',]


end = 60 * 30 *60     ## 60min, 30 frame for 1s
start = 0
roundTime = 0

num = 0

def count_percentage2(label_assemble,key):
    while '' in label_assemble:
        label_assemble.remove('')
    df = pd.DataFrame()
    num = 0
    for key in movement_order:
        count = label_assemble.count(key)
        df.loc[num,'later_label'] = key
        df.loc[num,'count'] = count
        
        if count / len(label_assemble) <0.05:
            df.loc[num,'count'] = 0
        
        num += 1
    df['percentage'] = df['count']/df['count'].sum()
    
    return(df)

def find_indices(list_to_check,item_to_find):
    return([idx for idx, value in enumerate(list_to_check) if value == item_to_find])

def calculate_MVtrans(sentences):
    df_list = []
    for key in movement_order:
        label_behind_key_sum = []
        
        for sentence in sentences:
            if key in sentence:
                index_ID = np.array(find_indices(sentence,key))
                if len(sentence)-1 in index_ID:
                    index_ID = np.delete(index_ID, find_indices(index_ID,len(sentence)-1)[0])
        
                    #del index_ID[find_indices(index_ID,len(sentence)-1)[0]]
                label_behind_key = np.array(sentence)[index_ID+1]
            else:
                label_behind_key = ['']
            
            #index_ID = np.array(find_indices(sentence,key))
            #if 0 in index_ID:
            #    index_ID = np.delete(index_ID, find_indices(index_ID,0)[0])
            #label_before_key = np.array(sentence)[index_ID-1]
            
            label_behind_key_sum.extend(label_behind_key)
        while '' in label_behind_key_sum:
            label_behind_key_sum.remove('')
        if len(label_behind_key_sum) > 10:
            df_count = count_percentage2(label_behind_key_sum,key)
            df_count['previous_lable'] = key
            df_count = df_count[['previous_lable','later_label','count','percentage']]
            df_list.append(df_count)
        
    all_df = pd.concat(df_list,axis=0)
    all_df.reset_index(drop=True,inplace=True)
    return(all_df)


df_entropy = pd.DataFrame()
for index in animal_info.index:
    video_index = animal_info.loc[index,'video_index']
    ExperimentCondition = animal_info.loc[index,'ExperimentCondition']
    LightingCondition = animal_info.loc[index,'LightingCondition']
    sentences = []
    df = pd.read_csv(Feature_space_path[video_index])
    temp_df = df[ (df['segBoundary_start']>=start) & (df['segBoundary_end']<end)]
    sentences.append(list(temp_df['revised_movement_label'].values))
    trans_df = calculate_MVtrans(sentences)
    trans_df = trans_df[trans_df['percentage']>0.05]
    entropy_dataset2 = scipy.stats.entropy(np.array(trans_df['count'].values),base=2)
    df_entropy.loc[num,'mouse_id'] = video_index
    df_entropy.loc[num,'ExperimentCondition'] = ExperimentCondition+"_"+LightingCondition
    df_entropy.loc[num,'entropy'] = entropy_dataset2
    num +=1
df_entropy.to_csv('{}/entropy_rate_MVtransition.csv'.format(output_dir))

np.array([1/2, 1/2]) 

fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(8,4),dpi=300)
sns.stripplot(data=df_entropy, x='ExperimentCondition',y='entropy',jitter=True,dodge=False,edgecolor='black',linewidth=1,ax=ax,alpha=0.9,
              order=['Morning_Light-on','Afternoon_Light-on','Night_Light-on','Night_Light-off','Stress_Light-on'],palette=['#F5B25E','#936736','#3498DB','#003960','#E75C6B'])
sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'k', 'ls': '-', 'lw': 3},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            saturation=1,
            dodge = True,
            width = 0.8,
            x="ExperimentCondition",
            y="entropy",
            order=['Morning_Light-on','Afternoon_Light-on','Night_Light-on','Night_Light-off','Stress_Light-on'],
            data=df_entropy,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=ax)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_ylabel('Entroy(bits)')
ax.set_xlabel('')
ax.set_title('Entroy')
#ax.set_yticks(np.arange(2,4.1,0.5))
ax.set_xticklabels(ax.get_xticklabels(), rotation = 30, ha="center")
plt.savefig('{}\Entroy_rate(bits).png'.format(output_dir),dpi=300)

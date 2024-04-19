# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:34:15 2023

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


InputData_path_dir = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure4_time-varying'

def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-')) & (file_name.endswith(content)):
            video_index = int(file_name.split('-')[1])
            file_path_dict.setdefault(video_index,file_dir+'\\'+file_name)
    return(file_path_dict)

Movement_Label_path = get_path(InputData_path_dir,'revised_Movement_Labels.csv')
Feature_space_path = get_path(InputData_path_dir,'Feature_Space.csv')


skip_file_list = [1,3,28,29,110,122] 
animal_info_csv =  r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Table_S1_animal_information.csv'                
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


movement_order = ['running','trotting','left_turning','right_turning','walking','stepping','sniffing',
                  'rising','hunching','rearing','climbing','jumping','grooming','scratching','pausing',]


def get_color(name):
    if name.startswith('Morning'):
        color = '#F5B25E'
    elif name.startswith('Afternoon'):
        color = '#936736'
    elif name.startswith('Night_lightOn'):
        color = '#398FCB'
    elif name.startswith('Night_lightOff'):
        color = '#003960'
    elif (name.startswith('Stress')) | (name.startswith('stress')):
        color = '#f55e6f'
    else:
        print(name)
    
    return(color)
        

dataset1 = Morning_lightOn_info
dataset1_name = 'Morning_lightOn'
dataset1_color = get_color(dataset1_name)

dataset2 = stress_animal_info
dataset2_name = 'stress_animal'
dataset2_color = get_color(dataset2_name)

group_set = [dataset1,dataset2]

def count_MV_ByMin(Mov_data,step,mv):
    start = 0
    end = 0
    num = 0
    df_single_animal = pd.DataFrame()
    for i in range(step,61,step):        
        end = i
        temp_df = Mov_data.iloc[start*30*60:end*30*60,:]
        MV_count = temp_df['revised_movement_label'].value_counts()
        #df_single_animal.loc[num,'time'] = end
        if mv not in MV_count.index:
            df_single_animal.loc[num,'MV_count'] = 0
        else:
            df_single_animal.loc[num,'MV_count'] = MV_count[mv]
        
        num +=1
        start = end
    df_single_animal.rename(columns={'MV_count':mv},inplace=True)
    return(df_single_animal.T)

df_list = []

for mv in movement_order:
    for index in list(dataset1.index):
        file_name = dataset1.loc[index,'video_index']
        Mov_data = pd.read_csv(Movement_Label_path[file_name])
        singlie_mice_mv = count_MV_ByMin(Mov_data=Mov_data,step=1,mv=mv)
        df_list.append(singlie_mice_mv)
    for index in list(dataset2.index):
        file_name = dataset2.loc[index,'video_index']
        Mov_data = pd.read_csv(Movement_Label_path[file_name])
        singlie_mice_mv = count_MV_ByMin(Mov_data=Mov_data,step=1,mv=mv)
        df_list.append(singlie_mice_mv)

df = pd.concat(df_list)

fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(10,11),dpi=1200)
sns.heatmap(df,cbar=False,yticklabels=[],cmap='RdYlBu_r',vmin=0.1,vmax=0.2)

mouse_num = 0
for i in movement_order:
    for j in range(len(group_set)):
        ax.axhline(mouse_num,lw=1,color='#18FFFF',linestyle='--',zorder=1)
        if j%2 == 1:
            ax.axhspan(ymin=mouse_num, ymax=mouse_num+len(group_set[j]), xmin=0.99, xmax=1,color=dataset1_color,zorder=3,ec='black',in_layout=False)
        else:
            ax.axhspan(ymin=mouse_num, ymax=mouse_num+len(group_set[j]), xmin=0.99, xmax=1,color=dataset2_color,zorder=3,ec='black')
        mouse_num += len(group_set[j])
    ax.text(62,mouse_num-len(group_set[j]),i)
    ax.axhline(mouse_num,lw=2,color='#FFFF00',zorder=3)
plt.xlim(0,61)  

plt.xticks(range(0,61,5),range(0,61,5))

plt.savefig('{}\{}_{}_movement_expression.png'.format(output_dir,dataset1_name,dataset2_name),dpi=1200,transparent=True)
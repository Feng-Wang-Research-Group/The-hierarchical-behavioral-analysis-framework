# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 10:14:38 2023

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
output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure1&Figure3_behavioral_characteristics(related to sFigure3-5)\proportion_pieplot'


def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-'))&(file_name.endswith(content)):
            USN = int(file_name.split('-')[1])
            file_path_dict.setdefault(USN,file_dir+'\\'+file_name)
    return(file_path_dict)

Movement_Label_path = get_path(InputData_path_dir,'revised_Movement_Labels.csv')
Feature_space_path = get_path(InputData_path_dir,'Feature_Space.csv')

skip_file_list = [1,3,28,29,110,122] 
animal_info_csv =  r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Table_S1_animal_information.csv'
animal_info = pd.read_csv(animal_info_csv)
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)]


Morning_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Morning') & (animal_info['LightingCondition']=='Light-on')]
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-off')]

Stress_info = animal_info[(animal_info['ExperimentCondition']=='Stress')]

movement_order = ['running','trotting','left_turning','right_turning','walking','stepping','sniffing',
                  'rising','hunching','rearing','climbing','jumping','grooming','scratching','pausing',]

cluster_order = ['locomotion','exploration','maintenance','nap']


cluster_color_dict={'locomotion':'#DC2543',                     
                     'exploration':'#009688',
                     'maintenance':'#A13E97',
                     'nap':'#B0BEC5'}

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

plot_movement_order = {}
for i in range(len(movement_order)):
    plot_movement_order.setdefault(movement_order[i],i)

plot_cluster_order = {}
for i in range(len(cluster_order)):
    plot_movement_order.setdefault(cluster_order[i],i)


dataset1 = Morning_lightOn_info
dataset1_name = 'Morning_lightOn'

dataset2 = Stress_info
dataset2_name = 'Stress_info'

#%%  plot cluster fraction


dataset1_list = []
for index in dataset1.index:
    video_index = dataset1.loc[index,'video_index']
    gender = dataset1.loc[index,'gender']
    annoMV_data = pd.read_csv(Movement_Label_path[video_index],usecols=['movement_cluster_label'])
    annoMV_data['gender'] = gender
    dataset1_list.append(annoMV_data)
    
df_dataset1_all = pd.concat(dataset1_list,axis=0)


dataset2_list = []
for index in dataset2.index:
    video_index = dataset2.loc[index,'video_index']
    gender = dataset2.loc[index,'gender']
    annoMV_data = pd.read_csv(Movement_Label_path[video_index],usecols=['movement_cluster_label'])
    annoMV_data['gender'] = gender
    dataset2_list.append(annoMV_data)
    
df_dataset2_all = pd.concat(dataset2_list,axis=0)


dataset_list = []

### movement
for sex in ['male','female']:
#for loc in ['wall']:
    temp_df_dataset1 = df_dataset1_all[df_dataset1_all['gender']==sex]
    dataset1_count_df = temp_df_dataset1.value_counts('movement_cluster_label').to_frame(name='count')
    dataset1_count_df['percentage'] = dataset1_count_df['count'] / dataset1_count_df['count'].sum()
    dataset1_count_df['color'] = dataset1_count_df.index.map(cluster_color_dict)
    dataset1_count_df['plot_order'] = dataset1_count_df.index.map(plot_cluster_order)
    dataset1_count_df.sort_values(by=['plot_order'],ascending=True,inplace=True)
    dataset1_count_df['gender'] = sex
    dataset1_count_df['ExperimentTime'] = dataset1_name.split('_')[0]
    dataset1_count_df['LightingCondition'] = dataset1_name.split('_')[1]
    dataset_list.append(dataset1_count_df)
    
    temp_df_dataset2 = df_dataset2_all[df_dataset2_all['gender']==sex]
    dataset2_count_df = temp_df_dataset2.value_counts('movement_cluster_label').to_frame(name='count')
    dataset2_count_df['percentage'] = dataset2_count_df['count'] / dataset2_count_df['count'].sum()
    dataset2_count_df['color'] = dataset2_count_df.index.map(cluster_color_dict)
    dataset2_count_df['plot_order'] = dataset2_count_df.index.map(plot_cluster_order)
    dataset2_count_df.sort_values(by=['plot_order'],ascending=True,inplace=True)
    dataset2_count_df['gender'] = sex
    dataset2_count_df['ExperimentTime'] = dataset2_name.split('_')[0]
    dataset2_count_df['LightingCondition'] = dataset2_name.split('_')[1]
    day_radius = night_radius = 1
    dataset_list.append(dataset2_count_df)

    #plt.figure(figsize=(10, 10),dpi=300)
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,10),dpi=600)
    ax.pie(dataset1_count_df['percentage']/2,counterclock=True,normalize=False,startangle = 90,
            #explode = [0.08,0.08,0.08,0.08],
            colors = dataset1_count_df['color'],
            radius = day_radius,
            #autopct= '%1.1f%%',pctdistance=1.2,
            wedgeprops = {'linewidth':3, 'edgecolor': "black"},)
            #wedgeprops = {'linewidth':3, 'edgecolor': "#F5B25E"},)
            #hatch=['.', '.', '.', '.'])
    plt.pie(dataset2_count_df['percentage']/2,counterclock=False,normalize=False,startangle = 90,
            #explode = [0.08,0.08,0.08,0.08],
            colors = dataset2_count_df['color'],
            radius = night_radius,
            #autopct= '%1.1f%%',pctdistance=1.2,
            wedgeprops = {'linewidth':3, 'edgecolor': "black"})
            #wedgeprops = {'linewidth':3, 'edgecolor': "#003960"},)
            #hatch=['.', '.', '.', '.'])
    plt.pie([1], colors = ['#ffffff'], radius = 0.5)
    plt.savefig('{}/{}-{}_{}_cluster_proportion.png'.format(output_dir,dataset1_name,dataset2_name,sex),transparent=True,dpi=600)


df_summary = pd.concat(dataset_list,axis=0)
df_summary.to_csv('{}/{}-{}_cluster_proportion.csv'.format(output_dir,dataset1_name,dataset2_name))


#%%  plot movement fraction

dataset1_list = []
for index in dataset1.index:
    video_index = dataset1.loc[index,'video_index']
    gender = dataset1.loc[index,'gender']
    annoMV_data = pd.read_csv(Movement_Label_path[video_index],usecols=['revised_movement_label'])
    annoMV_data['gender'] = gender
    dataset1_list.append(annoMV_data)
    
df_dataset1_all = pd.concat(dataset1_list,axis=0)


dataset2_list = []
for index in dataset2.index:
    video_index = dataset2.loc[index,'video_index']
    gender = dataset2.loc[index,'gender']
    annoMV_data = pd.read_csv(Movement_Label_path[video_index],usecols=['revised_movement_label'])
    annoMV_data['gender'] = gender
    dataset2_list.append(annoMV_data)
    
df_dataset2_all = pd.concat(dataset2_list,axis=0)


dataset_list = []

### movement
for sex in ['male','female']:
#for loc in ['wall']:
    temp_df_dataset1 = df_dataset1_all[df_dataset1_all['gender']==sex]
    dataset1_count_df = temp_df_dataset1.value_counts('revised_movement_label').to_frame(name='count')
    dataset1_count_df['percentage'] = dataset1_count_df['count'] / dataset1_count_df['count'].sum()
    dataset1_count_df['color'] = dataset1_count_df.index.map(movement_color_dict)
    dataset1_count_df['plot_order'] = dataset1_count_df.index.map(plot_movement_order)
    dataset1_count_df.sort_values(by=['plot_order'],ascending=True,inplace=True)
    dataset1_count_df['gender'] = sex
    dataset1_count_df['ExperimentTime'] = dataset1_name.split('_')[0]
    dataset1_count_df['LightingCondition'] = dataset1_name.split('_')[1]
    dataset_list.append(dataset1_count_df)
    
    temp_df_dataset2 = df_dataset2_all[df_dataset2_all['gender']==sex]
    dataset2_count_df = temp_df_dataset2.value_counts('revised_movement_label').to_frame(name='count')
    dataset2_count_df['percentage'] = dataset2_count_df['count'] / dataset2_count_df['count'].sum()
    dataset2_count_df['color'] = dataset2_count_df.index.map(movement_color_dict)
    dataset2_count_df['plot_order'] = dataset2_count_df.index.map(plot_movement_order)
    dataset2_count_df.sort_values(by=['plot_order'],ascending=True,inplace=True)
    dataset2_count_df['gender'] = sex
    dataset2_count_df['ExperimentTime'] = dataset2_name.split('_')[0]
    dataset2_count_df['LightingCondition'] = dataset2_name.split('_')[1]
    day_radius = night_radius = 1
    dataset_list.append(dataset2_count_df)

    #plt.figure(figsize=(10, 10),dpi=300)
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,10),dpi=600)
    ax.pie(dataset1_count_df['percentage']/2,counterclock=True,normalize=False,startangle = 90,
            #explode = [0.08,0.08,0.08,0.08],
            colors = dataset1_count_df['color'],
            radius = day_radius,
            #autopct= '%1.1f%%',pctdistance=1.2,
            wedgeprops = {'linewidth':3, 'edgecolor': "black"},)
            #wedgeprops = {'linewidth':3, 'edgecolor': "#F5B25E"},)
            #hatch=['.', '.', '.', '.'])
    plt.pie(dataset2_count_df['percentage']/2,counterclock=False,normalize=False,startangle = 90,
            #explode = [0.08,0.08,0.08,0.08],
            colors = dataset2_count_df['color'],
            radius = night_radius,
            #autopct= '%1.1f%%',pctdistance=1.2,
            wedgeprops = {'linewidth':3, 'edgecolor': "black"})
            #wedgeprops = {'linewidth':3, 'edgecolor': "#003960"},)
            #hatch=['.', '.', '.', '.'])
    plt.pie([1], colors = ['#ffffff'], radius = 0.5)
    plt.savefig('{}/{}-{}_{}_movement_fraction.png'.format(output_dir,dataset1_name,dataset2_name,sex),transparent=True,dpi=600)


df_summary = pd.concat(dataset_list,axis=0)
df_summary.to_csv('{}/{}-{}_movement_fraction.csv'.format(output_dir,dataset1_name,dataset2_name))




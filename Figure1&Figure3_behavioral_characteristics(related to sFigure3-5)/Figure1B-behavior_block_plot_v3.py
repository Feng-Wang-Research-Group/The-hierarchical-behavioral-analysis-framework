# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 21:40:57 2023

@author: Jialin Ye
@institution: SIAT
@Contact_email: jl.ye@siat.ac.cn

"""

import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from random import sample
import matplotlib.patches as patches
import matplotlib.path as mpath


InputData_path_dir = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure1&Figure3_behavioral_characteristics(related to sFigure3-5)'

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
Feature_space_path = get_path(InputData_path_dir,'Feature_Space.csv')

skip_file_list = [1,3,28,29,110,122] 

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


# =============================================================================
# movement_order2 = ['running','trotting','walking','right_turning','left_turning','stepping',
#                   'jumping','climbing','rearing','hunching','rising','sniffing',
#                   'grooming','pausing']
# =============================================================================

cluster_order = ['locomotion','exploration','maintenance','nap']



cluster_color_dict={'locomotion':'#DC2543',                     
                     'exploration':'#009688',
                     'maintenance':'#EADA33',
                     'nap':'#B0BEC5'}

movement_color_dict = {'running':'#FF3030',
                       'trotting':'#E15E8A',                       
                       'left_turning':'#F6BBC6', 
                       'right_turning':'#F8C8BA',
                       'walking':'#EB6148',
                       'stepping':'#C6823F',  
                       'sniffing':'#2E8BBE',
                       'rising':'#84CDD9',    #'#FFEA00'  ####FFEE58
                       'hunching':'#D4DF75',
                       'rearing':'#88AF26',
                       'climbing':'#2E7939',                           
                       'jumping':'#24B395',
                                              
                       'grooming':'#973C8D',
                       'scratching':'#EADA33',
                       'pausing':'#B0BEC5',}

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
        


plot_dataset = Morning_lightOn_info
dataset1_name = 'Morning_lightOn'
plot_dataset_male = plot_dataset[plot_dataset['gender']=='male']
plot_dataset_female = plot_dataset[plot_dataset['gender']=='female']
dataset1_color = get_color(dataset1_name)



plot_dataset2 = Stress_info
dataset2_name = 'Stress_info'
plot_dataset2_male = plot_dataset2[plot_dataset2['gender']=='male']
plot_dataset2_female = plot_dataset2[plot_dataset2['gender']=='female']
dataset2_color = get_color(dataset2_name)

event_list = []
mouse_id_list = []
night_num = 0
female_num = 0



for video_index in plot_dataset2_male.sample(5)['video_index']:
#for video_index in nightLightOff_info_female['video_index']:
    start = 0
    end = 0
    baseline = 0
    segBoundary_dict = {}
    FeA_file = pd.read_csv(Feature_space_path[video_index])
    for index in range(FeA_file.shape[0]):
        movement_label = FeA_file.loc[index,'revised_movement_label']
        segBoundary_dict.setdefault(movement_label,[])
        end = FeA_file.loc[index,'segBoundary_end'] + baseline*30*60*10
        lasting_time = end - start + 1
        segBoundary_dict[movement_label].append((start,lasting_time))
        start = end
    baseline +=1
    event_list.append(segBoundary_dict)
    mouse_id_list.append(video_index)
    #night_num +=1
    female_num += 1

for video_index in plot_dataset2_female.sample(5)['video_index']:
#for video_index in nightLightOff_info_male['video_index']:
    start = 0
    end = 0
    baseline = 0
    segBoundary_dict = {}
    FeA_file = pd.read_csv(Feature_space_path[video_index])
    for index in range(FeA_file.shape[0]):
        movement_label = FeA_file.loc[index,'revised_movement_label']
        segBoundary_dict.setdefault(movement_label,[])
        end = FeA_file.loc[index,'segBoundary_end'] + baseline*30*60*10
        lasting_time = end - start + 1
        segBoundary_dict[movement_label].append((start,lasting_time))
        start = end
    baseline +=1
    event_list.append(segBoundary_dict)
    mouse_id_list.append(video_index)
    #night_num +=1



for video_index in plot_dataset_male.sample(5)['video_index']:
#for video_index in nightLightOff_info_female['video_index']:
    start = 0
    end = 0
    baseline = 0
    segBoundary_dict = {}
    FeA_file = pd.read_csv(Feature_space_path[video_index])
    for index in range(FeA_file.shape[0]):
        movement_label = FeA_file.loc[index,'revised_movement_label']
        segBoundary_dict.setdefault(movement_label,[])
        end = FeA_file.loc[index,'segBoundary_end'] + baseline*30*60*10
        lasting_time = end - start + 1
        segBoundary_dict[movement_label].append((start,lasting_time))
        start = end
    baseline +=1
    event_list.append(segBoundary_dict)
    mouse_id_list.append(video_index)
    night_num +=1
    female_num += 1

for video_index in plot_dataset_female.sample(5)['video_index']:
#for video_index in nightLightOff_info_male['video_index']:
    start = 0
    end = 0
    baseline = 0
    segBoundary_dict = {}
    FeA_file = pd.read_csv(Feature_space_path[video_index])
    for index in range(FeA_file.shape[0]):
        movement_label = FeA_file.loc[index,'revised_movement_label']
        segBoundary_dict.setdefault(movement_label,[])
        end = FeA_file.loc[index,'segBoundary_end'] + baseline*30*60*10
        lasting_time = end - start + 1
        segBoundary_dict[movement_label].append((start,lasting_time))
        start = end
    baseline +=1
    event_list.append(segBoundary_dict)
    mouse_id_list.append(video_index)
    night_num +=1




# =============================================================================
# plot_list = ['stepping','sniffing','hunching','grooming','rising','climbing']                  ## morniong lightOn vs night lightOff difference movement
# plot_list2 = ['grooming']                                                                      ## morning lightOn vs afternoon light difference movement
# plot_list3 = ['walking','stepping','hunching','rising','grooming','pausing']        ## night lightOn vs night lightOff  difference movement
# plot_list4 = ['grooming','sniffing','rising','scratching','stepping','right_turning','climbing','left_turning','walking','hunching']
# 
# =============================================================================
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(40,14),constrained_layout=True,dpi=300)  ## 44:11 stress
ax.add_patch(
patches.Rectangle(
    (-3000, 0),   # (x,y)
    109000,          # width
    0.55*night_num,          # height
    alpha=0.1,ec=dataset2_color, fc=dataset2_color,lw=0.1,zorder=0)
    )

ax.add_patch(
patches.Rectangle(
    (-3000, 0.55*night_num),   # (x,y)
    110000,          # width
    0.55*night_num,          # height
    alpha=0.1,ec=dataset1_color, fc=dataset1_color,lw=0.1,zorder=0)                                   ### Morning F5B25E   night-lightOFF 003960  afternoon 936736 
    )                                                                                       ### night-lightOn  398FCB  # stress #f55e6f
num = 0
for i in range(len(event_list)):
    mouse_event = event_list[i]
    mouse_id = mouse_id_list[i]
    for key in mouse_event.keys():
        ##if key =='grooming':
        #if key in plot_list4:                                                                ## plot the select movement                                                                   
        movement_label = key
        #category = return_category5(movement_label)
        #category_color = cluster_color_dict[category]
        color = movement_color_dict[movement_label]
        ax.broken_barh(mouse_event[key],(num,0.5),facecolors=color,zorder=1)


    num +=0.55
plt.hlines(0.55*5,-100,109000,'black',linestyles='dashed',linewidths=5)
plt.hlines(0.55*15,-100,109000,'black',linestyles='dashed',linewidths=5)
plt.hlines(0.55*night_num,-100,109000,'black',linestyles='dashed',linewidths=8)

plt.vlines(-4000,-0.15,11,'black',linestyles='-',linewidths=6)
plt.hlines(-0.15,-4000,109000,'black',linestyles='-',linewidths=6)
for i in range(7):
    plt.vlines(30*60*10*i,-0.15,-0.3,'black',linestyles='-',linewidths=6)

plt.axis("off")

plt.savefig(r'{}\{}_{}.png'.format(output_dir,dataset1_name,dataset2_name),dpi=300, transparent=True)
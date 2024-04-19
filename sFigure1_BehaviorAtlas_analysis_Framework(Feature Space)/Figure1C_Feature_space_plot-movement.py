# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 09:31:58 2021

@author: 12517
"""

# selet point


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import matplotlib.colors as mcolors
import random

def get_path(file_dir,content):
    file_path_dict = {}
    for i in os.listdir(file_dir):
        if i.endswith(content):
            file_path_dict.setdefault(int(i.split('-')[1]),file_dir+'/'+i)
    return(file_path_dict)

def return_movement(num):
    for key in movement_dict.keys():
        if num in movement_dict[key]:
            return(key)
    return(key)


BehaviorAtlasData_dir =  r'F:\spontaneous_behavior\04返修阶段\01_BehaviorAtlas_collated_data'
revised_movement_dir = r'F:\spontaneous_behavior\04返修阶段\02_revised_movement_label'
FeA_path = get_path(BehaviorAtlasData_dir,'Feature_Space.csv')
#Mov_path = get_path(revised_movement_dir,'Movement_Labels.csv')


movement_dict = {'running':[29,28,13],                     # > 250 mm/s
                 'trotting':[14,22],                       # > 200 mm/s
                 'walking':[23,33,26,19],             # > 80 mm/s
                 'right_turning':[18,17,2,1],              # > 
                 'left_turning':[27,12],
                 'stepping':[9],                           # >50mm/s
                 'climbing':[31,32,25],	
                 'rearing':[16],
                 'hunching':[24],
                 'rising':[5,8,34],
                 'grooming':[37,40,15],
                 'sniffing':[10,11,30,35,36,6,38,4,3],
                 'pausing':[39,20,21],
                 'jumping':[7],
                 }

movement_dict_light_off = {'running':[30,29,35],                     # > 250 mm/s
                             'trotting':[16,5],                       # > 200 mm/s
                             'walking':[6,38,7,8,22,33],             # > 80 mm/s
                             'right_turning':[15,34,23],              # > 
                             'left_turning':[31,21,11],
                             'stepping':[28,27,],                           # >50mm/s
                             'climbing':[3,14,39],	                     ## 3有turning 
                             'rearing':[4,13],
                             'hunching':[1,26],
                             'rising':[2],
                             'grooming':[36,40,25,17],                   #### 杂
                             'sniffing':[10,18,12,20,9,32,19,24],
                             'pausing':[],
                             'jumping':[37],
                             }


cluster_color_dict={'locomotion':'#DC2543',                     
                     'exploration':'#009688',
                     'maintenance':'#A13E97',
                     'nap':'#D3D4D4'}

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



skip_file_list = [1,3,28,29,110,122] 

animal_info_csv = r'F:\spontaneous_behavior\04返修阶段\Table_S1_animal_information_clean.csv'
animal_info = pd.read_csv(animal_info_csv)
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)]


Morning_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Morning') & (animal_info['LightingCondition']=='Light-on')]
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-off')]

Stress_info = animal_info[(animal_info['ExperimentCondition']=='Stress')]

movement_order = ['running','trotting','left_turning','right_turning','walking','stepping','sniffing',
                  'rising','hunching','rearing','climbing','jumping','grooming','scratching','pausing',]


process_dataset = Morning_lightOn_info                        # change dataset here


plot_file_list = []
for  index in process_dataset.index:
    video_index = process_dataset.loc[index,'video_index']
    ExperimenTime = process_dataset.loc[index,'ExperimentCondition']
    LightingCondition = process_dataset.loc[index,'LightingCondition']
    gender = process_dataset.loc[index,'gender']
    mouse_id = process_dataset.loc[index,'mouse_id']
    
    if LightingCondition=='Light-on':
        plot_file_list.append(video_index)
        
start = 0
end = 0
n = 0
info_dict = {'file':[],'OriginalDigital_label':[],'boundary':[],'frames_number':[],'x':[],'y':[],'z':[]}

#for video_index in random.sample(plot_file_list,10): 
for video_index in plot_file_list:
    FeA_Data = pd.read_csv(FeA_path[video_index])
    for i in FeA_Data.index:
        end = FeA_Data.loc[i,'segBoundary']
        label = FeA_Data.loc[i,'OriginalDigital_label']
        mov_label = FeA_Data.loc[i,'OriginalDigital_label']
        info_dict['file'].append(video_index)
        info_dict['OriginalDigital_label'].append(label)
        info_dict['boundary'].append((start,end))
        info_dict['frames_number'].append(end-start+1)
        info_dict['x'].append(FeA_Data['umap1'][i])
        info_dict['y'].append(FeA_Data['umap2'][i])
        info_dict['z'].append(FeA_Data['zs_velocity'][i])
        start = end
        n += 1
df_info = pd.DataFrame(info_dict)

def random_point(df,label,number):
    label_df = df[df['OriginalDigital_label']==label]
    try:
        random_select = label_df.sample(number)
    except ValueError:
        random_select = label_df
    return(random_select)


df_info['d'] = ''
for label in df_info['OriginalDigital_label'].unique():
    temp_df = df_info[df_info['OriginalDigital_label']==label]
    if len(temp_df) >500:
        temp_df = temp_df.sample(500)
    else:
        temp_df = temp_df
    point_array = np.array(temp_df[['x','y','z']].astype('float64'))
    center_point = np.array((point_array[:,0].mean(),point_array[:,1].mean(),point_array[:,2].mean()))
    for i in temp_df.index:
        coor = np.array(temp_df.loc[i,['x','y','z']].astype('float64'))
        d = pdist(np.vstack((center_point,coor)),'euclidean')
        df_info.loc[i,'d'] = d[0]

#df_info = df_info[df_info['z']<0.5]




#fig, ax1 = plt.subplots(nrows=1, ncols=1,figsize=(10, 10),dpi=300)
fig = plt.figure(figsize=(10, 10),dpi=1200)
ax1 = fig.add_subplot(111, projection='3d')

#ax1.set_axis_off()
for i in df_info['OriginalDigital_label'].unique():
    label = i
    temp_df = df_info[df_info['OriginalDigital_label']==i]
    movement_label = return_movement(label)
    color = movement_color_dict[movement_label]
    x,y,z = temp_df['x'],temp_df['y'],temp_df['z']
    ax1.scatter(x,y,z,color=color,s=5,alpha=1)
    #ax1.text(x.tolist()[0],y.tolist()[0],z.tolist()[0],s=label,fontsize=15)
ax1.set_zlim(-2,6)
ax1.set_zticks(range(-2,7,2))
ax1.set_xlim(-2,2)
ax1.set_ylim(-2,2)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_zticklabels([])
#ax1.set_title('Moring')
#ax1.set_xlabel("umap1")
#ax1.set_ylabel("umap2")
#ax1.set_zlabel("zs_velocity")
#ax1.grid(False)
ax1.view_init(40,45)
ax1.xaxis._axinfo["grid"].update({"linewidth":2, "color" : "black"})
ax1.yaxis._axinfo["grid"].update({"linewidth":2, "color" : "black"})
ax1.zaxis._axinfo["grid"].update({"linewidth":2, "color" : "black"})
ax1.xaxis.set_pane_color('white')
ax1.yaxis.set_pane_color('white')
ax1.zaxis.set_pane_color('white')
#ax1.yaxis.pane.set_pane_color('white')
#ax1.zaxis.pane.set_pane_color('white')

#ax1.view_init(90,0)

plt.savefig(r'F:\spontaneous_behavior\04返修阶段\Figure_and_code\Figure3_Movement_fraction\3D_movement_feature_space.png',dpi=1200,transparent =True)


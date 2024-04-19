# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:42:22 2022

@author: Jialin Ye
@institution: SIAT
@Contact_email: jl.ye@siat.ac.cn

"""

import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import scipy.stats as st
import random



output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure5_spatial_perference\Example_trajectory\10min'
InputData_path_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\01_BehaviorAtlas_collated_data'
Mov_file_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label'

def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-')) & (file_name.endswith(content)):
            video_index = int(file_name.split('-')[1])
            file_path_dict.setdefault(video_index,file_dir+'\\'+file_name)
    return(file_path_dict)

Movement_Label_path = get_path(Mov_file_dir,'revised_Movement_Labels.csv')  
Corr_path = get_path(InputData_path_dir,'normalized_coordinates_back_XY.csv')  
Speed_path = get_path(InputData_path_dir,'speed&distance.csv')  

skip_file_list = [1,3,28,29,110,122] 

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


cluster_color_dict = {'locomotion':'#F94040','exploration':'#0C8766','maintenance':'#914C99','nap':'#D4D4D4'}  #B0BEC5

movement_color_dict = {'running':'#FF3030',
                       'trotting':'#F06292',
                       'walking':'#FF5722',
                       'right_turning':'#F29B78',
                       'left_turning':'#FFBFB4',                       
                       'stepping':'#A1887F',  
                       'rising':'#FFEA00',
                       'hunching':'#ECAD4F',
                       'rearing':'#C0CA33',
                       'climbing':'#2E7939',                           
                       'jumping':'#80DEEA',
                       'sniffing':'#2C93CB',                       
                       'grooming':'#A13E97',
                       'scratching':'#00ACC1',
                       'pausing':'#B0BEC5',}


boundary10_Moring = 175
boundary10_Stress = 175

boundary1060_Moring = 175
boundary1060_Stress = 175

time_window1 = 0
time_window2 = 10 


video_index_morning = random.sample(list(Morning_lightOn_info['video_index'].values),1)[0]   ### select a mouse randomly
video_index_stress = random.sample(list(stress_animal_info['video_index'].values),1)[0]

dataset1_name = 'Morning_lightOn'
dataset2_name = 'stress'

if time_window1 == 0:
    dataset1_boundary = boundary10_Moring
    dataset2_boundary = boundary10_Stress
elif time_window1 == 10:
    dataset1_boundary = boundary1060_Moring
    dataset2_boundary = boundary1060_Stress
else:
    print('wrong and check')


Mov_data_Morning = pd.read_csv(Movement_Label_path[video_index_morning])
Mov_data_Stress = pd.read_csv(Movement_Label_path[video_index_stress])


### plot track scatter distributioin (tranditional method)
def plot_scatter_distributioin(coor_data,dataset_name,time_window1,time_window2,boundary,color):
    coor_data['location'] = 'periphery'
    coor_data.loc[(coor_data['back_x']>= (250-boundary))&(coor_data['back_x']<=(250+boundary))&(coor_data['back_y']>= (250-boundary))&(coor_data['back_y']<=(250+boundary)),'location'] = 'center'
    coor_data = coor_data.iloc[time_window1*30*60:time_window2*30*60,:]
    coor_data.reset_index(drop=True,inplace=True)
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,10),dpi=300)
    sns.scatterplot(data=coor_data,x='back_x',y='back_y', edgecolor="none", s=20, hue='location',hue_order=['center','periphery'],
                     legend=False,palette=['#2C7EC2','#9E9E9E'],alpha = 1, ax=ax,zorder=1) ##009688  #BDBDBD
    
    ax.plot([0,0],[0,500],color='black',lw=8,zorder=2)
    ax.plot([0,500],[500,500],color='black',lw=8,zorder=2)
    ax.plot([500,500],[500,0],color='black',lw=8,zorder=2)
    ax.plot([500,0],[0,0],color='black',lw=8,zorder=2)
    # 
    ax.plot([(250-boundary),(250-boundary)],[0,500],color='black',lw=8,zorder=2) # tranditional defined center  (x1,x2)(y1,y2)  #2C7EC2
    ax.plot([(250+boundary),(250+boundary)],[0,500],color='black',lw=8,zorder=2)
    ax.plot([0,500],[(250-boundary),(250-boundary)],color='black',lw=8,zorder=2)
    ax.plot([0,500],[(250+boundary),(250+boundary)],color='black',lw=8,zorder=2)
    
    a = [(250-boundary),(250-boundary),(250+boundary),(250+boundary)]
    b = [(250-boundary),(250+boundary),(250+boundary),(250-boundary)]
    
    ax.fill(a,b,color=color,alpha=0.5,zorder=0)
    
    plt.axis('off')
    plt.savefig('{}/{}_coordinates_disribution.png'.format(output_dir,dataset_name),dpi=300,transparent=True)

plot_scatter_distributioin(Mov_data_Morning,dataset1_name,time_window1,time_window2,dataset1_boundary,color='#F5B25E')
plot_scatter_distributioin(Mov_data_Stress,dataset2_name,time_window1,time_window2,dataset2_boundary,color='#f55e6f')

### plot mice postion 2D distributioin 
def plot_distribution_density(coor_data,dataset_name,time_window1,time_window2,boundary,color):
    coor_data_copy = coor_data.copy()
    coor_data_copy = coor_data.iloc[time_window1*30*60:time_window2*30*60,:]
    coor_data.reset_index(drop=True,inplace=True)
    g = sns.JointGrid(height=10, ratio=5, space=.05,xlim=(0, 500), ylim=(0, 500))
    sns.scatterplot(data=coor_data_copy,x='back_x',y='back_y', edgecolor="none", s=8, color=color,alpha =1,ax=g.ax_joint)
    g.set_axis_labels(xlabel='',ylabel='')
    
    g.ax_joint.plot([0,0],[0,500],color='black',lw=5)
    g.ax_joint.plot([0,500],[500,500],color='black',lw=5)
    g.ax_joint.plot([500,500],[500,0],color='black',lw=5)
    g.ax_joint.plot([500,0],[0,0],color='black',lw=5)
    
    
    g.ax_joint.plot([(250-boundary),(250-boundary)],[0,500],color='black',lw=5) # tranditional defined center  (x1,x2)(y1,y2)  #2C7EC2
    g.ax_joint.plot([(250+boundary),(250+boundary)],[0,500],color='black',lw=5)
    g.ax_joint.plot([0,500],[(250-boundary),(250-boundary)],color='black',lw=5)
    g.ax_joint.plot([0,500],[(250+boundary),(250+boundary)],color='black',lw=5)
    
    g.ax_joint.set_xticklabels([])
    g.ax_joint.set_yticklabels([])
    
    sns.kdeplot(data=coor_data_copy,x='back_x', color = color,fill=color, linewidth=4, ax=g.ax_marg_x)
    sns.kdeplot(data=coor_data_copy,y='back_y', color = color,fill=color, linewidth=4, ax=g.ax_marg_y)
    plt.savefig('{}/{}_coordinates_distribution_with_density.png'.format(output_dir,dataset_name),dpi=600,transparent=True)

plot_distribution_density(Mov_data_Morning,dataset1_name,time_window1,time_window2,dataset1_boundary,color='#F5B25E')
plot_distribution_density(Mov_data_Stress,dataset2_name,time_window1,time_window2,dataset2_boundary,color='#f55e6f')


def plot_speed_trajectory(coor_Mov_data,dataset_name,time_window1,time_window2,boundary,color):
### plot speed_trajectory
    coor_Mov_data_copy = coor_Mov_data.copy()
    cmap1 = cm.RdYlBu_r
    norm1 = mcolors.Normalize(vmin = 0, vmax= 400)
    coor_Mov_data_copy['speed_color'] = list(cmap1(norm1(coor_Mov_data_copy['smooth_speed'])))
    coor_Mov_data_copy = coor_Mov_data_copy.iloc[time_window1*30*60:time_window2*30*60,:]
    coor_Mov_data_copy.reset_index(drop=True,inplace=True)
    
    fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10),constrained_layout=True,dpi=300)
    ax.plot([0,0],[0,500],color='black',lw=12,zorder=1)
    ax.plot([0,500],[500,500],color='black',lw=12,zorder=1)
    ax.plot([500,500],[500,0],color='black',lw=12,zorder=1)
    ax.plot([500,0],[0,0],color='black',lw=12,zorder=1)
    
    ax.plot([(250-boundary),(250-boundary)],[0,500],color='black',lw=12,zorder=1)
    ax.plot([(250+boundary),(250+boundary)],[0,500],color='black',lw=12,zorder=1)
    ax.plot([0,500],[(250-boundary),(250-boundary)],color='black',lw=12,zorder=1)
    ax.plot([0,500],[(250+boundary),(250+boundary)],color='black',lw=12,zorder=1)
    
    plt.axis('off')
    
    for i in range(1,coor_Mov_data_copy.shape[0]):
        x_coor1 = coor_Mov_data_copy.loc[i-1,'back_x']
        y_coor1 = coor_Mov_data_copy.loc[i-1,'back_y']
        x_coor2 = coor_Mov_data_copy.loc[i,'back_x']
        y_coor2 = coor_Mov_data_copy.loc[i,'back_y']
        color = coor_Mov_data_copy.loc[i,'speed_color']
        ax.plot([x_coor1,x_coor2],[y_coor1,y_coor2],c=color,alpha =1,lw=8,linestyle='solid',zorder=0)
    #plt.colorbar(cm.ScalarMappable(norm=norm1, cmap=cmap1), ax=ax)
    plt.savefig('{}/{}_speed_trajectory.png'.format(output_dir,dataset_name),dpi=300,transparent=True)
    plt.show()

plot_speed_trajectory(Mov_data_Morning,dataset1_name,time_window1,time_window2,dataset1_boundary,color='#F5B25E')
plot_speed_trajectory(Mov_data_Stress,dataset2_name,time_window1,time_window2,dataset2_boundary,color='#F5B25E')



### plot_movement trajectory
def plot_movement_trajectory(coor_Mov_data,dataset_name,time_window1,time_window2,boundary,color):
    coor_Mov_data_copy = coor_Mov_data.copy()
    coor_Mov_data_copy = coor_Mov_data_copy.iloc[time_window1*30*60:time_window2*30*60,:]
    coor_Mov_data_copy.reset_index(drop=True,inplace=True)
    fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10),constrained_layout=True,dpi=300)
    for c in movement_color_dict.keys():
        color = movement_color_dict[c]
        ax.scatter(coor_Mov_data_copy.loc[coor_Mov_data_copy['revised_movement_label']==c,'back_x'].values,coor_Mov_data_copy.loc[coor_Mov_data_copy['revised_movement_label']==c,'back_y'].values,color= color,s=30,zorder=0,alpha=0.8)
    ax.plot([0,0],[0,500],color='black',lw=12)
    ax.plot([0,500],[500,500],color='black',lw=12)
    ax.plot([500,500],[500,0],color='black',lw=12)
    ax.plot([500,0],[0,0],color='black',lw=12)
    
    ax.plot([(250-boundary),(250-boundary)],[0,500],color='black',lw=12) # tranditional defined center  (x1,x2)(y1,y2)  #2C7EC2
    ax.plot([(250+boundary),(250+boundary)],[0,500],color='black',lw=12)
    ax.plot([0,500],[(250-boundary),(250-boundary)],color='black',lw=12)
    ax.plot([0,500],[(250+boundary),(250+boundary)],color='black',lw=12)
    
    plt.axis('off')
    plt.savefig('{}/{}_movement_trajectory.png'.format(output_dir,dataset_name),dpi=300,transparent=True)

plot_movement_trajectory(Mov_data_Morning,dataset1_name,time_window1,time_window2,dataset1_boundary,color='#F5B25E')
plot_movement_trajectory(Mov_data_Stress,dataset2_name,time_window1,time_window2,dataset2_boundary,color='#F5B25E')

### plot_movement cluster trajectory
def plot_cluster_trajectory(coor_Mov_data,dataset_name,time_window1,time_window2,boundary,color):
    coor_Mov_data_copy = coor_Mov_data.copy()
    coor_Mov_data_copy = coor_Mov_data_copy.iloc[time_window1*30*60:time_window2*30*60,:]
    coor_Mov_data_copy.reset_index(drop=True,inplace=True)
    fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10),constrained_layout=True,dpi=300)
    
    for c in cluster_color_dict.keys():
        color = cluster_color_dict[c]
        ax.scatter(coor_Mov_data_copy.loc[coor_Mov_data_copy['movement_cluster_label']==c,'back_x'].values,coor_Mov_data_copy.loc[coor_Mov_data_copy['movement_cluster_label']==c,'back_y'].values,color= color,s=30,zorder=0,alpha=0.8)
    
       
    ax.plot([0,0],[0,500],color='black',lw=12)
    ax.plot([0,500],[500,500],color='black',lw=12)
    ax.plot([500,500],[500,0],color='black',lw=12)
    ax.plot([500,0],[0,0],color='black',lw=8)
    
    ax.plot([(250-boundary),(250-boundary)],[0,500],color='black',lw=12) # tranditional defined center  (x1,x2)(y1,y2)  #2C7EC2
    ax.plot([(250+boundary),(250+boundary)],[0,500],color='black',lw=12)
    ax.plot([0,500],[(250-boundary),(250-boundary)],color='black',lw=12)
    ax.plot([0,500],[(250+boundary),(250+boundary)],color='black',lw=12)
        
    plt.axis('off')
    plt.savefig('{}/{}_movement_cluster_trajectory.png'.format(output_dir,dataset_name),dpi=300,transparent=True)

plot_cluster_trajectory(Mov_data_Morning,dataset1_name,time_window1,time_window2,dataset1_boundary,color='#F5B25E')
plot_cluster_trajectory(Mov_data_Stress,dataset2_name,time_window1,time_window2,dataset2_boundary,color='#F5B25E')



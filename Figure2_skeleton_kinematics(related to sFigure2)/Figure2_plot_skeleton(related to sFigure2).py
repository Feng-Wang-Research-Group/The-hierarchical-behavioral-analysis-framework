#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 09:22:20 2022

@author: yejohnny
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from scipy.stats import gaussian_kde
import mpl_scatter_density
import os 
from matplotlib.patches import Polygon


InputData_path_dir = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label"
InputData_path_dir2 =  r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\01_BehaviorAtlas_collated_data"
output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure2_skeleton_kinematics(related to sFigure2)\overall_15movement_skeleton'

if not os.path.exists(output_dir):                                    
    os.mkdir(output_dir) 


def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-')) & (file_name.endswith(content)):
            video_index = int(file_name.split('-')[1])
            #date = i.split('-')[3][0:8]
            #file_name = 'rec-{0}-G1-{1}'.format(video_index,date)
            file_path_dict.setdefault(video_index,file_dir+'\\'+file_name)
    return(file_path_dict)

Movement_Label_path = get_path(InputData_path_dir,'revised_Movement_Labels.csv')       ####### 生成 文件名 与 路径 对应的字典
Skeleton_path = get_path(InputData_path_dir2,'normalized_skeleton_XYZ.csv')


skip_file_list = [1,3,28,29,110,122] 

animal_info_csv = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Table_S1_animal_information.csv'             
animal_info = pd.read_csv(animal_info_csv)
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)]

Light_on_info = animal_info[animal_info['LightingCondition']=='Light-on']

Morning_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Morning') & (animal_info['LightingCondition']=='Light-on')]
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-off')]

stress_info = animal_info[animal_info['ExperimentCondition']=='Stress']


movement_order = ['running','trotting','left_turning','right_turning','walking','stepping','sniffing',
                  'rising','hunching','rearing','climbing','jumping','grooming','scratching','pausing',]

ske_data_list = []
#for video_index in list(animal_info['video_index']):
for video_index in animal_info.index:
    ske_data = pd.read_csv(Skeleton_path[video_index])
    Mov_data = pd.read_csv(Movement_Label_path[video_index],usecols=['original_label','revised_movement_label','smooth_speed'])
    conbime_data = pd.concat([Mov_data,ske_data],axis=1)
    #select_data = conbime_data[conbime_data['revised_movement_label'].isin(pausing_list)]
    #select_data = conbime_data[conbime_data['OriginalDigital_label'].isin(OriginalDigital_label_list)]
    ske_data_list.append(conbime_data)


all_df = pd.concat(ske_data_list,axis=0)
all_df.reset_index(drop=True,inplace=True)



df_select_list = []
for MV in movement_order:
    df_singleLocomotion = all_df[all_df['revised_movement_label']==MV]
    if len(df_singleLocomotion) < 10000:
        df_frame_select = df_singleLocomotion
    else:
        df_frame_select = df_singleLocomotion.sample(n=10000, random_state=2024) # weights='locomotion_speed_smooth',
    print(MV,len(df_frame_select))
    df_select_list.append(df_frame_select)


df_select = pd.concat(df_select_list)
df_select.reset_index(drop=True,inplace=True)


def cal_ang(point_1, point_2, point_3):
    """
    Calculate the angle based on three point coordinates
    :param point_1: point1 coordinates
    :param point_2: point2 coordinates
    :param point_3: point3 coordinates
    :return: Returns the angle value (point2 is the angle)
    """
    a=math.sqrt((point_2[0]-point_3[0])*(point_2[0]-point_3[0])+(point_2[1]-point_3[1])*(point_2[1] - point_3[1]))
    b=math.sqrt((point_1[0]-point_3[0])*(point_1[0]-point_3[0])+(point_1[1]-point_3[1])*(point_1[1] - point_3[1]))
    c=math.sqrt((point_1[0]-point_2[0])*(point_1[0]-point_2[0])+(point_1[1]-point_2[1])*(point_1[1]-point_2[1]))
    A=math.degrees(math.acos((a*a-b*b-c*c)/(-2*b*c)))
    B=math.degrees(math.acos((b*b-a*a-c*c)/(-2*a*c)))
    C=math.degrees(math.acos((c*c-a*a-b*b)/(-2*a*b)))
    return (B)

color_list = {'nose':'#1e2c59',
              'left_ear':'#192887',
              'right_ear':'#1b3a95',
              'neck':'#204fa1',
              'left_front_limb':'#1974ba',
              'right_front_limb':'#1ea2ba',
              'left_hind_limb':'#42b799',
              'right_hind_limb':'#5cb979',
              'left_front_claw':'#7bbf57',
              'right_front_claw':'#9ec036',
              'left_hind_claw':'#beaf1f',
              'right_hind_claw':'#c08719',
              'back':'#bf5d1c',
              'root_tail':'#be3320',
              'mid_tail':'#9b1f24',
              'tip_tail':'#6a1517',}

def plot_TopView_skeleton(df_select):
    for i in df_select['revised_movement_label'].unique():
        df_singleLocomotion = df_select[df_select['revised_movement_label'] == i]
    #for i in df_select['OriginalDigital_label'].unique():       
    #    df_singleLocomotion = df_select[df_select['OriginalDigital_label'] == i]
        df_singleLocomotion = df_singleLocomotion.iloc[:,3:]
        df_singleLocomotion.reset_index(drop=True,inplace=True)
        df_singleLocomotion.loc[len(df_singleLocomotion.index)-1] = df_singleLocomotion.mean(axis=0)
        
        
        face_color = 'white' #movement_color_dict[i]  #'white' #movement_color_dict[return_AnnoMovement(i)]
        
        nose = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_y']]
        left_ear = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_ear_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_ear_y']]
        right_ear = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_ear_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_ear_y']]
        neck = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_y']]
        left_front_limb = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_limb_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_limb_y']]
        right_front_limb = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_limb_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_limb_y']]
        left_hind_limb = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_limb_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_limb_y']]
        right_hind_limb = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_limb_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_limb_y']]
        left_front_claw = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_claw_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_claw_y']]
        right_front_claw = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_claw_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_claw_y']]
        left_hind_claw = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_claw_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_claw_y']]
        right_hind_claw = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_claw_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_claw_y']]
        back = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_y']]
        root_tail = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_y']]
        mid_tail = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'mid_tail_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'mid_tail_y']]
        tip_tail = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'tip_tail_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'tip_tail_y']]
        
        
   
        fig = plt.figure(figsize=(5,5),constrained_layout=True,dpi=300)
        ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
        #ax.scatter(nose[0],nose[1],s=60,c='green',alpha = 0.8)
        ax.scatter(nose[0],nose[1],s=150,c='#9C27B0',marker='o',ec = 'black')
        ax.scatter(left_ear[0],left_ear[1],s=150,c='#B0BEC5',alpha = 0.8,ec = 'black')
        ax.scatter(right_ear[0],right_ear[1],s=150,c='#B0BEC5',alpha = 0.8,ec = 'black')
        
        ax.plot([nose[0],left_ear[0]],[nose[1],left_ear[1]],c='black',alpha = 0.8,lw=5)
        ax.plot([nose[0],right_ear[0]],[nose[1],right_ear[1]],c='black',alpha = 0.8,lw=5)
        ax.plot([left_ear[0],right_ear[0]],[left_ear[1],right_ear[1]],c='black',alpha = 0.8,lw=5)
        
        ax.scatter(neck[0],neck[1],s=150,c='#B0BEC5',alpha = 0.8,ec = 'black')
        ax.scatter(back[0],back[1],s=150,c='#B0BEC5',alpha = 0.8,ec = 'black')
        ax.scatter(root_tail[0],root_tail[1],s=150,c='#B0BEC5',alpha = 0.8,ec = 'black')
        ax.scatter(mid_tail[0],mid_tail[1],s=150,c='#B0BEC5',alpha = 0.8,ec = 'black')
        #ax.scatter(tip_tail[0],tip_tail[1],s=40,c='#607D5B',alpha = 0.8)
        
        ax.scatter(left_front_limb[0],left_front_limb[1],s=150,c='#B0BEC5',alpha = 0.8)
        ax.scatter(right_front_limb[0],right_front_limb[1],s=150,c='#B0BEC5',alpha = 0.8)
        ax.scatter(left_hind_limb[0],left_hind_limb[1],s=150,c='#B0BEC5',alpha = 0.8)
        ax.scatter(right_hind_limb[0],right_hind_limb[1],s=150,c='#B0BEC5',alpha = 0.8)
        
        ax.scatter(left_front_claw[0],left_front_claw[1],s=150,c='#FF1744',marker='^',ec = 'black',alpha = 0.8)
        ax.scatter(right_front_claw[0],right_front_claw[1],s=150,c='#F57C00',marker='^',ec = 'black',alpha = 0.8)
        ax.scatter(left_hind_claw[0],left_hind_claw[1],s=150,c='#FFFF00',marker='^',ec = 'black',alpha = 0.8)
        ax.scatter(right_hind_claw[0],right_hind_claw[1],s=150,c='#00E676',marker='^',ec = 'black',alpha = 0.8)
        
        ax.plot([left_front_limb[0],left_hind_limb[0]],[left_front_limb[1],left_hind_limb[1]],c='black',alpha = 0.8,lw=5)
        ax.plot([left_hind_limb[0],right_hind_limb[0]],[left_hind_limb[1],right_hind_limb[1]],c='black',alpha = 0.8,lw=5)
        ax.plot([right_hind_limb[0],right_front_limb[0]],[right_hind_limb[1],right_front_limb[1]],c='black',alpha = 0.8,lw=5)
        ax.plot([right_front_limb[0],left_front_limb[0]],[right_front_limb[1],left_front_limb[1]],c='black',alpha = 0.8,lw=5)
        
        ax.plot([left_ear[0],neck[0]],[left_ear[1],neck[1]],c='black',alpha = 0.8,lw=4)
        ax.plot([right_ear[0],neck[0]],[right_ear[1],neck[1]],c='black',alpha = 0.8,lw=4)
        ax.plot([left_front_limb[0],neck[0]],[left_front_limb[1],neck[1]],c='black',alpha = 0.8,lw=4)
        ax.plot([right_front_limb[0],neck[0]],[right_front_limb[1],neck[1]],c='black',alpha = 0.8,lw=4)
        ax.plot([left_hind_limb[0],root_tail[0]],[left_hind_limb[1],root_tail[1]],c='black',alpha = 0.8,lw=5)
        ax.plot([right_hind_limb[0],root_tail[0]],[right_hind_limb[1],root_tail[1]],c='black',alpha = 0.8,lw=5)
        
        ax.plot([root_tail[0],mid_tail[0]],[root_tail[1],mid_tail[1]],c='black',alpha = 0.8,lw=5)
        #ax.plot([mid_tail[0],tip_tail[0]],[mid_tail[1],tip_tail[1]],c='black',alpha = 0.8,lw=5)
        
    
        ax.plot([left_front_limb[0],back[0]],[left_front_limb[1],back[1]],c='black',alpha = 0.8,lw=4)
        ax.plot([right_front_limb[0],back[0]],[right_front_limb[1],back[1]],c='black',alpha = 0.8,lw=4)
        ax.plot([left_hind_limb[0],back[0]],[left_hind_limb[1],back[1]],c='black',alpha = 0.8,lw=4)
        ax.plot([right_hind_limb[0],back[0]],[right_hind_limb[1],back[1]],c='black',alpha = 0.8,lw=4)
        
        ax.add_patch(Polygon([(back[0],back[1]),(left_front_limb[0],left_front_limb[1]),(right_front_limb[0],right_front_limb[1])], color=face_color,alpha=0.5))
        ax.add_patch(Polygon([(back[0],back[1]),(left_front_limb[0],left_front_limb[1]),(left_hind_limb[0],left_hind_limb[1])], color=face_color,alpha=0.5))
        ax.add_patch(Polygon([(back[0],back[1]),(right_front_limb[0],right_front_limb[1]),(right_hind_limb[0],right_hind_limb[1])], color=face_color,alpha=0.5))
        ax.add_patch(Polygon([(back[0],back[1]),(left_hind_limb[0],left_hind_limb[1]),(right_hind_limb[0],right_hind_limb[1])], color=face_color,alpha=0.5))
        ax.add_patch(Polygon([(root_tail[0],root_tail[1]),(left_hind_limb[0],left_hind_limb[1]),(right_hind_limb[0],right_hind_limb[1])], color=face_color,alpha=0.5))
        
        ax.add_patch(Polygon([(nose[0],nose[1]),(left_ear[0],left_ear[1]),(right_ear[0],right_ear[1])], color=face_color,alpha=0.5))
        plt.title(i,fontsize = 25)
        
        ax.scatter_density(df_singleLocomotion.loc[:,'left_front_claw_x'], df_singleLocomotion.loc[:,'left_front_claw_y'], color = '#FF1744',alpha=1,dpi=30)
        ax.scatter_density(df_singleLocomotion.loc[:,'right_front_claw_x'], df_singleLocomotion.loc[:,'right_front_claw_y'], color = '#F57C00',alpha=1,dpi=30)
        ax.scatter_density(df_singleLocomotion.loc[:,'left_hind_claw_x'], df_singleLocomotion.loc[:,'left_hind_claw_y'], color ='#FFFF00',alpha=1,dpi=30)
        ax.scatter_density(df_singleLocomotion.loc[:,'right_hind_claw_x'], df_singleLocomotion.loc[:,'right_hind_claw_y'], color ='#00E676',alpha=1,dpi=30)
        ax.scatter_density(df_singleLocomotion.loc[:,'nose_x'], df_singleLocomotion.loc[:,'nose_y'], color ='#9C27B0',alpha=1,dpi=30)
        ax.set_xlim(-100,100)
        ax.set_ylim(-100,100)
        plt.axis('off')
        #if i == 40:
        plt.savefig('{}/{}_skeleton_topview.png'.format(output_dir,i),dpi=300,transparent=True)
plot_TopView_skeleton(df_select)


def plot_2DsideView_skeleton(df_select):    
    #### sideview
    for i in df_select['revised_movement_label'].unique():
        df_singleLocomotion = df_select[df_select['revised_movement_label'] == i]
    #for i in df_select['revised_movement_label'].unique():
    #    df_singleLocomotion = df_select[df_select['revised_movement_label'] == i]
        df_singleLocomotion = df_singleLocomotion.iloc[:,4:]
        df_singleLocomotion.reset_index(drop=True,inplace=True)
        df_singleLocomotion.loc[len(df_singleLocomotion.index)] = df_singleLocomotion.mean(axis=0)
        
        
        face_color = 'white'   # movement_color_dict[i] # 'white' #movement_color_dict[return_AnnoMovement(i)]
        
        nose = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_z']]
        left_ear = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_ear_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_ear_z']]
        right_ear = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_ear_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_ear_z']]
        neck = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_z']]
        left_front_limb = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_limb_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_limb_z']]
        right_front_limb = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_limb_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_limb_z']]
        left_hind_limb = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_limb_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_limb_z']]
        right_hind_limb = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_limb_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_limb_z']]
        left_front_claw = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_claw_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_claw_z']]
        right_front_claw = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_claw_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_claw_z']]
        left_hind_claw = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_claw_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_claw_z']]
        right_hind_claw = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_claw_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_claw_z']]
        back = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_z']]
        root_tail = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_z']]
        mid_tail = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'mid_tail_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'mid_tail_z']]
        tip_tail = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'tip_tail_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'tip_tail_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'tip_tail_z']]
    
        fig = plt.figure(figsize=(5,5),constrained_layout=True,dpi=300)
        ax = fig.add_subplot(1, 1, 1, projection='scatter_density')

        ax.scatter(nose[0],nose[1],s=150,c='#9C27B0',marker='o',ec = 'black',zorder=20)
        ax.scatter(left_ear[0],left_ear[1],s=150,c='#B0BEC5',ec = 'black',alpha = 1,zorder=20)
        ax.scatter(right_ear[0],right_ear[1],s=150,c='#B0BEC5',ec = 'black',alpha = 1,zorder=20)
        
        ax.plot([nose[0],left_ear[0]],[nose[1],left_ear[1]],c='black',alpha = 0.8,lw=5)
        ax.plot([nose[0],right_ear[0]],[nose[1],right_ear[1]],c='black',alpha = 0.8,lw=5)
        ax.plot([left_ear[0],right_ear[0]],[left_ear[1],right_ear[1]],c='black',alpha = 0.8,lw=5)
        
     
        ax.scatter(neck[0],neck[1],s=150,c='#03A9F4',ec = 'black',alpha = 1,zorder=20)
        ax.scatter(back[0],back[1],s=150,c='#66BB6A',ec = 'black',alpha = 1,zorder=20)
        ax.scatter(root_tail[0],root_tail[1],s=150,c='#009688',ec = 'black',alpha = 1,zorder=20)
        ax.scatter(mid_tail[0],mid_tail[1],s=150,c='#9E9D24',ec = 'black',alpha = 1,zorder=20)
        #ax.scatter(tip_tail[0],tip_tail[1],s=40,c='#607D5B',alpha = 0.8)
        
        ax.scatter(left_front_limb[0],left_front_limb[1],s=150,c='#B0BEC5',ec = 'black',alpha = 1,zorder=20)
        ax.scatter(right_front_limb[0],right_front_limb[1],s=150,c='#B0BEC5',ec = 'black',alpha = 1,zorder=20)
        ax.scatter(left_hind_limb[0],left_hind_limb[1],s=150,c='#B0BEC5',ec = 'black',alpha = 1,zorder=20)
        ax.scatter(right_hind_limb[0],right_hind_limb[1],s=150,c='#B0BEC5',ec = 'black',alpha = 1,zorder=20)
              
        
        ax.scatter(left_front_claw[0],left_front_claw[1],s=150,c='#FF1744',marker='^',ec = 'black',alpha =1,zorder=20)
        ax.scatter(right_front_claw[0],right_front_claw[1],s=150,c='#F57C00',marker='^',ec = 'black',alpha = 1,zorder=20)
        ax.scatter(left_hind_claw[0],left_hind_claw[1],s=150,c='#00E676',marker='^',ec = 'black',alpha = 1,zorder=30)
        ax.scatter(right_hind_claw[0],right_hind_claw[1],s=150,c='#FFFF00',marker='^',ec = 'black',alpha = 1,zorder=20)

        
        ax.plot([left_front_limb[0],left_hind_limb[0]],[left_front_limb[1],left_hind_limb[1]],c='black',alpha = 0.8,lw=5)
        ax.plot([left_hind_limb[0],right_hind_limb[0]],[left_hind_limb[1],right_hind_limb[1]],c='black',alpha = 0.8,lw=5)
        ax.plot([right_hind_limb[0],right_front_limb[0]],[right_hind_limb[1],right_front_limb[1]],c='black',alpha = 0.8,lw=5)
        ax.plot([right_front_limb[0],left_front_limb[0]],[right_front_limb[1],left_front_limb[1]],c='black',alpha = 0.8,lw=5)
        
        ax.plot([left_ear[0],neck[0]],[left_ear[1],neck[1]],c='black',alpha = 0.8,lw=5)
        ax.plot([right_ear[0],neck[0]],[right_ear[1],neck[1]],c='black',alpha = 0.8,lw=5)
        ax.plot([left_front_limb[0],neck[0]],[left_front_limb[1],neck[1]],c='black',alpha = 0.8,lw=5)
        ax.plot([right_front_limb[0],neck[0]],[right_front_limb[1],neck[1]],c='black',alpha = 0.8,lw=5)
        ax.plot([left_hind_limb[0],root_tail[0]],[left_hind_limb[1],root_tail[1]],c='black',alpha = 0.8,lw=5)
        ax.plot([right_hind_limb[0],root_tail[0]],[right_hind_limb[1],root_tail[1]],c='black',alpha = 0.8,lw=5)
        
        
        #ax.plot([mid_tail[0],tip_tail[0]],[mid_tail[1],tip_tail[1]],c='black',alpha = 0.8,lw=5)
        
        
        ax.plot([left_front_limb[0],back[0]],[left_front_limb[1],back[1]],c='black',alpha = 0.8,lw=5)
        ax.plot([right_front_limb[0],back[0]],[right_front_limb[1],back[1]],c='black',alpha = 0.8,lw=5)
        ax.plot([left_hind_limb[0],back[0]],[left_hind_limb[1],back[1]],c='black',alpha = 0.8,lw=5)
        ax.plot([right_hind_limb[0],back[0]],[right_hind_limb[1],back[1]],c='black',alpha = 0.8,lw=5)
        
        ax.plot([left_front_limb[0],left_front_claw[0]],[left_front_limb[1],left_front_claw[1]],c='black',alpha = 0.8,lw=5)
        ax.plot([right_front_limb[0],right_front_claw[0]],[right_front_limb[1],right_front_claw[1]],c='black',alpha = 0.8,lw=5)
        ax.plot([left_hind_limb[0],left_hind_claw[0]],[left_hind_limb[1],left_hind_claw[1]],c='black',alpha = 0.8,lw=5)
        ax.plot([right_hind_limb[0],right_hind_claw[0]],[right_hind_limb[1],right_hind_claw[1]],c='black',alpha = 0.8,lw=5)
        
        
        ax.plot([neck[0],nose[0]],[neck[1],nose[1]],c='black',alpha = 1,lw=5)
        ax.plot([neck[0],back[0]],[neck[1],back[1]],c='black',alpha = 1,lw=5)
        ax.plot([root_tail[0],back[0]],[root_tail[1],back[1]],c='black',alpha = 1,lw=5)
        ax.plot([root_tail[0],mid_tail[0]],[root_tail[1],mid_tail[1]],c='black',alpha = 0.8,lw=5)
        
        ax.scatter_density(-df_singleLocomotion.loc[:,'left_front_claw_y'], df_singleLocomotion.loc[:,'left_front_claw_z'], color = '#FF1744',alpha=1)
        ax.scatter_density(-df_singleLocomotion.loc[:,'right_front_claw_y'], df_singleLocomotion.loc[:,'right_front_claw_z'], color = '#F57C00',alpha=1)
        ax.scatter_density(-df_singleLocomotion.loc[:,'left_hind_claw_y'], df_singleLocomotion.loc[:,'left_hind_claw_z'], color ='#00E676',alpha=1)
        ax.scatter_density(-df_singleLocomotion.loc[:,'right_hind_claw_y'], df_singleLocomotion.loc[:,'right_hind_claw_z'], color ='#FFFF00',alpha=1)
        ax.scatter_density(-df_singleLocomotion.loc[:,'nose_y'], df_singleLocomotion.loc[:,'nose_z'], color ='#9C27B0',alpha=1)
        
        plt.title(i,fontsize = 25)
        ax.add_patch(Polygon([(back[0],back[1]),(left_front_limb[0],left_front_limb[1]),(right_front_limb[0],right_front_limb[1])], color=face_color,alpha=0.5))
        ax.add_patch(Polygon([(back[0],back[1]),(left_front_limb[0],left_front_limb[1]),(left_hind_limb[0],left_hind_limb[1])], color=face_color,alpha=0.5))
        ax.add_patch(Polygon([(back[0],back[1]),(right_front_limb[0],right_front_limb[1]),(right_hind_limb[0],right_hind_limb[1])], color=face_color,alpha=0.5))
        ax.add_patch(Polygon([(back[0],back[1]),(left_hind_limb[0],left_hind_limb[1]),(right_hind_limb[0],right_hind_limb[1])], color=face_color,alpha=0.5))
        ax.add_patch(Polygon([(root_tail[0],root_tail[1]),(left_hind_limb[0],left_hind_limb[1]),(right_hind_limb[0],right_hind_limb[1])], color=face_color,alpha=0.5))
        ax.add_patch(Polygon([(nose[0],nose[1]),(left_ear[0],left_ear[1]),(right_ear[0],right_ear[1])], color=face_color,alpha=0.5))
        
        ax.set_xlim(-100,100)
        ax.set_ylim(-70,170)
        plt.axis('off')

        plt.savefig('{}/{}_skeleton_sideview.png'.format(output_dir,i),dpi=300,transparent=True)
        
plot_2DsideView_skeleton(df_select)


def plot_3DsideView_skeleton(df_select):
#### 3D###
    for i in df_select['revised_movement_label'].unique():
        df_singleLocomotion = df_select[df_select['revised_movement_label'] == i]
    #for i in df_select['OriginalDigital_label'].unique():
        #df_singleLocomotion = df_select[df_select['OriginalDigital_label'] == i]
        df_singleLocomotion = df_singleLocomotion.iloc[:,4:]
        df_singleLocomotion.reset_index(drop=True,inplace=True)
        df_singleLocomotion.loc[len(df_singleLocomotion.index)] = df_singleLocomotion.mean(axis=0)
        
        
        nose = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_z']]
        left_ear = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_ear_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_ear_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_ear_z']]
        right_ear = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_ear_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_ear_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_ear_z']]
        neck = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_z']]
        left_front_limb = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_limb_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_limb_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_limb_z']]
        right_front_limb = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_limb_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_limb_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_limb_z']]
        left_hind_limb = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_limb_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_limb_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_limb_z']]
        right_hind_limb = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_limb_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_limb_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_limb_z']]
        left_front_claw = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_claw_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_claw_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_claw_z']]
        right_front_claw = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_claw_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_claw_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_claw_z']]
        left_hind_claw = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_claw_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_claw_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_claw_z']]
        right_hind_claw = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_claw_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_claw_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_claw_z']]
        back = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_z']]
        root_tail = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_z']]
        mid_tail = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'mid_tail_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'mid_tail_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'mid_tail_z']]
        tip_tail = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'tip_tail_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'tip_tail_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'tip_tail_z']]
    
        fig = plt.figure(figsize=(8,8),constrained_layout=True,dpi=300)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
          
        ax.scatter(nose[0],nose[1],nose[2],s=150,c='#9C27B0',marker='o',ec = 'black',zorder=20)
        ax.scatter(left_ear[0],left_ear[1],left_ear[2],s=150,c='#B0BEC5',ec = 'black',alpha = 1,zorder=20)
        ax.scatter(right_ear[0],right_ear[1],right_ear[2],s=150,c='#B0BEC5',ec = 'black',alpha = 1,zorder=20)
        
        ax.scatter(neck[0],neck[1],neck[2],s=150,c='#03A9F4',ec = 'black',alpha = 1,zorder=20)
        ax.scatter(back[0],back[1],back[2],s=150,c='#66BB6A',ec = 'black',alpha = 1,zorder=20)
        ax.scatter(root_tail[0],root_tail[1],root_tail[2],s=150,c='#009688',ec = 'black',alpha = 1,zorder=20)
        ax.scatter(mid_tail[0],mid_tail[1],mid_tail[2],s=150,c='#9E9D24',ec = 'black',alpha = 1,zorder=20)
        #ax.scatter(tip_tail[0],tip_tail[1],s=40,c='orange')
        
        ax.scatter(left_front_limb[0],left_front_limb[1],left_front_limb[2],s=150,c='#B0BEC5',ec = 'black',alpha = 1,zorder=20)
        ax.scatter(right_front_limb[0],right_front_limb[1],right_front_limb[2],s=150,c='#B0BEC5',ec = 'black',alpha = 1,zorder=20)
        ax.scatter(left_hind_limb[0],left_hind_limb[1],left_hind_limb[2],s=150,c='#B0BEC5',ec = 'black',alpha = 1,zorder=20)
        ax.scatter(right_hind_limb[0],right_hind_limb[1],right_hind_limb[2],s=150,c='#B0BEC5',ec = 'black',alpha = 1,zorder=20)
        
        
        ax.scatter(left_front_claw[0],left_front_claw[1],left_front_claw[2],s=250,c='#FF1744',marker='^',ec = 'black',alpha =1,zorder=20)
        ax.scatter(right_front_claw[0],right_front_claw[1],right_front_claw[2],s=250,c='#F57C00',marker='^',ec = 'black',alpha = 1,zorder=20)
        ax.scatter(left_hind_claw[0],left_hind_claw[1],left_hind_claw[2],s=250,c='#00E676',marker='^',ec = 'black',alpha = 1,zorder=30)
        ax.scatter(right_hind_claw[0],right_hind_claw[1],right_hind_claw[2],s=250,c='#FFFF00',marker='^',ec = 'black',alpha = 1,zorder=20)
        
        
        ax.plot([nose[0],left_ear[0]],[nose[1],left_ear[1]],[nose[2],left_ear[2]],c='black',alpha = 0.8,lw=5)
        ax.plot([nose[0],right_ear[0]],[nose[1],right_ear[1]],[nose[2],right_ear[2]],c='black',alpha = 0.8,lw=5)
        ax.plot([left_ear[0],right_ear[0]],[left_ear[1],right_ear[1]],[left_ear[2],right_ear[2]],c='black',alpha = 0.8,lw=5)
        
        ax.plot([left_front_limb[0],left_hind_limb[0]],[left_front_limb[1],left_hind_limb[1]],[left_front_limb[2],left_hind_limb[2]],c='black',alpha = 0.8,lw=5)
        ax.plot([left_hind_limb[0],right_hind_limb[0]],[left_hind_limb[1],right_hind_limb[1]],[left_hind_limb[2],right_hind_limb[2]],c='black',alpha = 0.8,lw=5)
        ax.plot([right_hind_limb[0],right_front_limb[0]],[right_hind_limb[1],right_front_limb[1]],[right_hind_limb[2],right_front_limb[2]],c='black',alpha = 0.8,lw=5)
        ax.plot([right_front_limb[0],left_front_limb[0]],[right_front_limb[1],left_front_limb[1]],[right_front_limb[2],left_front_limb[2]],c='black',alpha = 0.8,lw=5)
        
        ax.plot([left_ear[0],neck[0]],[left_ear[1],neck[1]],[left_ear[2],neck[2]],c='black',alpha = 0.8,lw=5)
        ax.plot([right_ear[0],neck[0]],[right_ear[1],neck[1]],[right_ear[2],neck[2]],c='black',alpha = 0.8,lw=5)
        ax.plot([left_front_limb[0],neck[0]],[left_front_limb[1],neck[1]],[left_front_limb[2],neck[2]],c='black',alpha = 0.8,lw=5)
        ax.plot([right_front_limb[0],neck[0]],[right_front_limb[1],neck[1]],[right_front_limb[2],neck[2]],c='black',alpha = 0.8,lw=5)
        ax.plot([left_hind_limb[0],root_tail[0]],[left_hind_limb[1],root_tail[1]],[left_hind_limb[2],root_tail[2]],c='black',alpha = 0.8,lw=5)
        ax.plot([right_hind_limb[0],root_tail[0]],[right_hind_limb[1],root_tail[1]],[right_hind_limb[2],root_tail[2]],c='black',alpha = 0.8,lw=5)
        
        ax.plot([root_tail[0],mid_tail[0]],[root_tail[1],mid_tail[1]],[root_tail[2],mid_tail[2]],c='black',alpha = 0.8,lw=5)
        #ax.plot([mid_tail[0],tip_tail[0]],[mid_tail[1],tip_tail[1]],c='black')
        
    
        ax.plot([left_front_limb[0],back[0]],[left_front_limb[1],back[1]],[left_front_limb[2],back[2]],c='black',alpha = 0.8,lw=5)
        ax.plot([right_front_limb[0],back[0]],[right_front_limb[1],back[1]],[right_front_limb[2],back[2]],c='black',alpha = 0.8,lw=5)
        ax.plot([left_hind_limb[0],back[0]],[left_hind_limb[1],back[1]],[left_hind_limb[2],back[2]],c='black',alpha = 0.8,lw=5)
        ax.plot([right_hind_limb[0],back[0]],[right_hind_limb[1],back[1]],[right_hind_limb[2],back[2]],c='black',alpha = 0.8,lw=5)
        
        ax.plot([left_front_limb[0],left_front_claw[0]],[left_front_limb[1],left_front_claw[1]],[left_front_limb[2],left_front_claw[2]],c='black',alpha = 0.8,lw=5)
        ax.plot([right_front_limb[0],right_front_claw[0]],[right_front_limb[1],right_front_claw[1]],[right_front_limb[2],right_front_claw[2]],c='black',alpha = 0.8,lw=5)
        ax.plot([left_hind_limb[0],left_hind_claw[0]],[left_hind_limb[1],left_hind_claw[1]],[left_hind_limb[2],left_hind_claw[2]],c='black',alpha = 0.8,lw=5)
        ax.plot([right_hind_limb[0],right_hind_claw[0]],[right_hind_limb[1],right_hind_claw[1]],[right_hind_limb[2],right_hind_claw[2]],c='black',alpha = 0.8,lw=5)
        
        ax.set_xlim(-100,100)
        ax.set_ylim(-100,100)
        ax.set_zlim(-70,170)
        plt.title(i,fontsize = 25)
        ax.view_init(20, 150)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.grid(False)
        plt.axis('off')
        #ax.scatter(df_singleLocomotion.loc[:,'left_front_claw_x'], df_singleLocomotion.loc[:,'left_front_claw_y'], df_singleLocomotion.loc[:,'left_front_claw_z'],color = '#FF1744',alpha=0.2,s=0.5)
        #ax.scatter(df_singleLocomotion.loc[:,'right_front_claw_x'], df_singleLocomotion.loc[:,'right_front_claw_y'],df_singleLocomotion.loc[:,'right_front_claw_z'], color = '#F57C00',alpha=0.2,s=0.5)
        #ax.scatter(df_singleLocomotion.loc[:,'left_hind_claw_x'], df_singleLocomotion.loc[:,'left_hind_claw_y'], df_singleLocomotion.loc[:,'left_hind_claw_z'],color ='#00E676',alpha=0.2,s=0.5)
        #ax.scatter(df_singleLocomotion.loc[:,'right_hind_claw_x'], df_singleLocomotion.loc[:,'right_hind_claw_y'], df_singleLocomotion.loc[:,'right_hind_claw_z'],color ='#00E5FF',alpha=0.2,s=0.5)
        plt.savefig('{}/{}_skeleton_3DsideView.png'.format(output_dir,i),dpi=300,transparent=True)
plot_3DsideView_skeleton(df_select)

#%% plot torso (spine)

exploration_list = ['sniffing','rising','hunching','rearing','climbing']   # excluded jumping

df_select_list = []
for MV in exploration_list:
    df_singleLocomotion = all_df[all_df['revised_movement_label']==MV]
    if len(df_singleLocomotion) < 1000:
        df_frame_select = df_singleLocomotion
    else:
        df_frame_select = df_singleLocomotion.sample(n=1000, random_state=2024) # weights='locomotion_speed_smooth',
    print(MV,len(df_frame_select))
    df_select_list.append(df_frame_select)

df_select = pd.concat(df_select_list)
df_select.reset_index(drop=True,inplace=True)



def plot_angle_sideview(df_select):         ### for stand up
    angle_info_dict = {'movement_label':[],'nose-neck-back':[],'neck-back-tail':[],'back-tail1-tail2':[]}
    for i in df_select['revised_movement_label'].unique():
        df_singleLocomotion = df_select[df_select['revised_movement_label'] == i]
        df_singleLocomotion = df_singleLocomotion.iloc[:,4:]
        df_singleLocomotion.reset_index(drop=True,inplace=True)
        df_singleLocomotion.loc[len(df_singleLocomotion.index)] = df_singleLocomotion.mean(axis=0)
        
        nose_average = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_z']]
        neck_average = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_z']]
        back_average = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_z']]
        root_tail_average = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_z']]
        mid_tail_average = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'mid_tail_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'mid_tail_z']]
        
        ## 计算角度
        angle1 = cal_ang(nose_average, neck_average, back_average)
        angle2 = cal_ang(neck_average, back_average,root_tail_average)
        angle3 = cal_ang(back_average,root_tail_average,mid_tail_average)
        angle_info_dict['movement_label'].append(i)
        angle_info_dict['nose-neck-back'].append(angle1)
        angle_info_dict['neck-back-tail'].append(angle2)
        angle_info_dict['back-tail1-tail2'].append(angle3)
        
        fig = plt.figure(figsize=(9,9),constrained_layout=True,dpi=300)
        ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
        for j in df_singleLocomotion.index[0:-2]:
            nose = [-df_singleLocomotion.loc[j,'nose_y'],df_singleLocomotion.loc[j,'nose_z']]
            neck = [-df_singleLocomotion.loc[j,'neck_y'],df_singleLocomotion.loc[j,'neck_z']]
            back = [-df_singleLocomotion.loc[j,'back_y'],df_singleLocomotion.loc[j,'back_z']]
            root_tail = [-df_singleLocomotion.loc[j,'root_tail_y'],df_singleLocomotion.loc[j,'root_tail_z']]
            mid_tail = [-df_singleLocomotion.loc[j,'mid_tail_y'],df_singleLocomotion.loc[j,'mid_tail_z']]
            
            if (mid_tail[0] >25) & (mid_tail[1] >-10) :
                ax.scatter(nose[0],nose[1],s=40,c='#9C27B0',alpha = 0.1,zorder=0)
                ax.scatter(neck[0],neck[1],s=40,c='#03A9F4',alpha = 0.1,zorder=0)
                ax.scatter(back[0],back[1],s=40,c='#66BB6A',alpha = 0.1,zorder=0)
                ax.scatter(root_tail[0],root_tail[1],s=40,c='#009688',alpha = 0.1,zorder=0)
                ax.scatter(mid_tail[0],mid_tail[1],s=40,c='#9E9D24',alpha = 0.1,zorder=0)
                
                ax.plot([nose[0],neck[0]],[nose[1],neck[1]],c='#90A4AE',alpha = 0.1,lw=1,zorder=0)
                ax.plot([neck[0],back[0]],[neck[1],back[1]],c='#90A4AE',alpha = 0.1,lw=1,zorder=0)
                ax.plot([back[0],root_tail[0]],[back[1],root_tail[1]],c='#90A4AE',alpha = 0.1,lw=1,zorder=0)
                ax.plot([root_tail[0],mid_tail[0]],[root_tail[1],mid_tail[1]],c='#90A4AE',alpha = 0.1,lw=1,zorder=0)
        
        ax.scatter(nose_average[0],nose_average[1],s=260,c='#9C27B0',ec='black',alpha = 1,zorder=2)
        ax.scatter(neck_average[0],neck_average[1],s=260,c='#03A9F4',ec='black',alpha = 1,zorder=2)
        ax.scatter(back_average[0],back_average[1],s=260,c='#66BB6A',ec='black',alpha = 1,zorder=2)
        ax.scatter(root_tail_average[0],root_tail_average[1],s=260,ec='black',c='#009688',alpha = 1,zorder=2)
        ax.scatter(mid_tail_average[0],mid_tail_average[1],s=260,c='#9E9D24',ec='black',alpha = 1,zorder=2)
        
        ax.plot([nose_average[0],neck_average[0]],[nose_average[1],neck_average[1]],c='black',alpha = 1,lw=3,zorder=1)
        ax.plot([neck_average[0],back_average[0]],[neck_average[1],back_average[1]],c='black',alpha = 1,lw=3,zorder=1)
        ax.plot([back_average[0],root_tail_average[0]],[back_average[1],root_tail_average[1]],c='black',alpha = 1,lw=3,zorder=1)
        ax.plot([root_tail_average[0],mid_tail_average[0]],[root_tail_average[1],mid_tail_average[1]],c='black',alpha = 1,lw=3,zorder=1)
        plt.title(i,fontsize = 25)
        
        ax.set_xlim(-100,100)
        ax.set_ylim(-25,175)
        plt.axis('off')
        plt.savefig('{}/{}_sideview_torso_angle.png'.format(output_dir,i),dpi=600,transparent=True)
    df = pd.DataFrame(angle_info_dict)
    print(df)
        
        
plot_angle_sideview(df_select)      


#%% plot turning spine curvature


turning_list = ['running','trotting','left_turning','right_turning','walking','stepping']   # excluded jumping

df_select_list = []
for MV in turning_list:
    df_singleLocomotion = all_df[all_df['revised_movement_label']==MV]
    if len(df_singleLocomotion) < 1000:
        df_frame_select = df_singleLocomotion
    else:
        df_frame_select = df_singleLocomotion.sample(n=1000, random_state=2024) # weights='locomotion_speed_smooth',
    print(MV,len(df_frame_select))
    df_select_list.append(df_frame_select)

df_select = pd.concat(df_select_list)
df_select.reset_index(drop=True,inplace=True)


def plot_angle(df_select):
    angle_info_dict = {'movement_label':[],'nose-neck-back':[],'neck-back-tail':[],'back-tail1-tail2':[]}
    for i in df_select['revised_movement_label'].unique():
        df_singleLocomotion = df_select[df_select['revised_movement_label'] == i]
    #for i in df_select['OriginalDigital_label'].unique():       
    #    df_singleLocomotion = df_select[df_select['OriginalDigital_label'] == i]
        df_singleLocomotion = df_singleLocomotion.iloc[:,3:]
        df_singleLocomotion.reset_index(drop=True,inplace=True)
        df_singleLocomotion.loc[len(df_singleLocomotion.index)-1] = df_singleLocomotion.mean(axis=0)
        
        nose_average = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_y']]
        neck_average = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_y']]
        back_average = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_y']]
        root_tail_average = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_y']]
        mid_tail_average = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'mid_tail_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'mid_tail_y']]
        
        ## 计算角度
        angle1 = cal_ang(nose_average, neck_average, back_average)
        angle2 = cal_ang(neck_average, back_average,root_tail_average)
        angle3 = cal_ang(back_average,root_tail_average,mid_tail_average)
        angle_info_dict['movement_label'].append(i)
        angle_info_dict['nose-neck-back'].append(angle1)
        angle_info_dict['neck-back-tail'].append(angle2)
        angle_info_dict['back-tail1-tail2'].append(angle3)
        
        fig = plt.figure(figsize=(10,10),constrained_layout=True,dpi=300)
        ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    
        for j in df_singleLocomotion.index[0:-2]:
            nose = [df_singleLocomotion.loc[j,'nose_x'],df_singleLocomotion.loc[j,'nose_y']]
            neck = [df_singleLocomotion.loc[j,'neck_x'],df_singleLocomotion.loc[j,'neck_y']]
            back = [df_singleLocomotion.loc[j,'back_x'],df_singleLocomotion.loc[j,'back_y']]
            root_tail = [df_singleLocomotion.loc[j,'root_tail_x'],df_singleLocomotion.loc[j,'root_tail_y']]
            mid_tail = [df_singleLocomotion.loc[j,'mid_tail_x'],df_singleLocomotion.loc[j,'mid_tail_y']]
            
            ax.scatter(nose[0],nose[1],s=60,c='#9C27B0',alpha = 0.1)
            ax.scatter(neck[0],neck[1],s=60,c='#F57C00',alpha = 0.1)
            ax.scatter(back[0],back[1],s=60,c='#00E676',alpha = 0.1)
            ax.scatter(root_tail[0],root_tail[1],s=60,c='#00E5FF',alpha = 0.1)
            ax.scatter(mid_tail[0],mid_tail[1],s=60,c='#607D5B',alpha = 0.1)
            
            ax.plot([nose[0],neck[0]],[nose[1],neck[1]],c='#90A4AE',alpha = 0.1,lw=1)
            ax.plot([neck[0],back[0]],[neck[1],back[1]],c='#90A4AE',alpha = 0.1,lw=1)
            ax.plot([back[0],root_tail[0]],[back[1],root_tail[1]],c='#90A4AE',alpha = 0.1,lw=1)
            ax.plot([root_tail[0],mid_tail[0]],[root_tail[1],mid_tail[1]],c='#90A4AE',alpha = 0.1,lw=1)
        ax.scatter(nose_average[0],nose_average[1],s=260,c='#9C27B0',ec='black',alpha = 1)
        ax.scatter(neck_average[0],neck_average[1],s=260,c='#03A9F4',ec='black',alpha = 1)
        ax.scatter(back_average[0],back_average[1],s=260,c='#66BB6A',ec='black',alpha = 1)
        ax.scatter(root_tail_average[0],root_tail_average[1],s=260,ec='black',c='#009688',alpha = 1)
        ax.scatter(mid_tail_average[0],mid_tail_average[1],s=260,c='#9E9D24',ec='black',alpha = 1)
        
        ax.plot([nose_average[0],neck_average[0]],[nose_average[1],neck_average[1]],c='black',alpha = 1,lw=3)
        ax.plot([neck_average[0],back_average[0]],[neck_average[1],back_average[1]],c='black',alpha = 1,lw=3)
        ax.plot([back_average[0],root_tail_average[0]],[back_average[1],root_tail_average[1]],c='black',alpha = 1,lw=3)
        ax.plot([root_tail_average[0],mid_tail_average[0]],[root_tail_average[1],mid_tail_average[1]],c='black',alpha = 1,lw=3)
        plt.title(i,fontsize = 25)
        ax.set_xlim(-100,100)
        ax.set_ylim(-100,100)
        plt.axis('off')
        plt.savefig('{}/{}_TopView_torso_angle.png'.format(output_dir,i),dpi=600,transparent=True)
    df = pd.DataFrame(angle_info_dict)
    print(df)
plot_angle(df_select)


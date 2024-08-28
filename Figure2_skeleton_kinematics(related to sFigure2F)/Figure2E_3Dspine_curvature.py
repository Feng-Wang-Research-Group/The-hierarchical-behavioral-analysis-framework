# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:14:00 2024

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
import scipy.signal as signal
import math
import joypy


InputData_path_dir =  r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\01_BehaviorAtlas_collated_data"
InputData_path_dir2 = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure2_skeleton_kinematics(related to sFigure2F)\movement_parametter\spine_angle3D'
if not os.path.exists(output_dir):                                    
    os.mkdir(output_dir)


def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-')) & (file_name.endswith(content)):
            video_index = int(file_name.split('-')[1])
            file_path_dict.setdefault(video_index,file_dir+'\\'+file_name)
    return(file_path_dict)

Movement_Label_path = get_path(InputData_path_dir2,'revised_Movement_Labels.csv')
Feature_space_path = get_path(InputData_path_dir2,'Feature_Space.csv')
Skeleton_path = get_path(InputData_path_dir,'normalized_skeleton_XYZ.csv')
#Skeleton_path = get_path(InputData_path_dir2,'Cali_Data3d.csv')

skip_file_list = [1,3,28,29,110,122] 
animal_info_csv = animal_info_csv = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Table_S1_animal_information.csv'             
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

exploration = ['sniffing','rising','hunching','rearing','climbing']

grooming = []

def get_color(name):
    if name.startswith('Morning'):
        color = '#F5B25E'
    elif name.startswith('Afternoon'):
        color = '#936736'
    elif name.startswith('Night_lightOn'):
        color = '#398FCB'
    elif name.startswith('Night_lightOff'):
        color = '#003960'
    elif name.startswith('Stress') | name.startswith('stress'):
        color = '#f55e6f'
    else:
        print(name)
    
    return(color)
        

plot_dataset = Morning_lightOn_info
dataset1_name = 'Morning_lightOn'
plot_dataset_male = plot_dataset[plot_dataset['gender']=='male']
plot_dataset_female = plot_dataset[plot_dataset['gender']=='female']
dataset1_color = get_color(dataset1_name)


plot_dataset2 = stress_animal_info
dataset2_name = 'stress_animal'
plot_dataset2_male = plot_dataset2[plot_dataset2['gender']=='male']
plot_dataset2_female = plot_dataset2[plot_dataset2['gender']=='female']
dataset2_color = get_color(dataset2_name)


def cal_ang(point_1, point_2, point_3):
    a=math.sqrt((point_2[0]-point_3[0])*(point_2[0]-point_3[0])+(point_2[1]-point_3[1])*(point_2[1] - point_3[1]))
    b=math.sqrt((point_1[0]-point_3[0])*(point_1[0]-point_3[0])+(point_1[1]-point_3[1])*(point_1[1] - point_3[1]))
    c=math.sqrt((point_1[0]-point_2[0])*(point_1[0]-point_2[0])+(point_1[1]-point_2[1])*(point_1[1]-point_2[1]))
    A=math.degrees(math.acos((a*a-b*b-c*c)/(-2*b*c)))
    B=math.degrees(math.acos((b*b-a*a-c*c)/(-2*a*c)))
    C=math.degrees(math.acos((c*c-a*a-b*b)/(-2*a*b)))
    return (B)

def cal_3D_angle(df):
    df_copy = df.copy()
    for index in df.index:
        
        nose_x = df.loc[index,'nose_x']
        nose_y = df.loc[index,'nose_y']
        nose_z = df.loc[index,'nose_z']
        
        neck_x = df.loc[index,'neck_x']
        neck_y = df.loc[index,'neck_y']
        neck_z = df.loc[index,'neck_z']
        
        back_x = df.loc[index,'back_x']
        back_y = df.loc[index,'back_y']
        back_z = df.loc[index,'back_z']
        
        root_tail_x = df.loc[index,'root_tail_x']
        root_tail_y = df.loc[index,'root_tail_y']
        root_tail_z = df.loc[index,'root_tail_z']
    
    
        ## calculate angle
        angle1 = cal_ang([nose_x,nose_y,nose_z], [neck_x,neck_y,neck_z], [back_x,back_y,back_z])
        angle2 = cal_ang([neck_x,neck_y,neck_z], [back_x,back_y,back_z],[root_tail_x,root_tail_y,root_tail_z])
        #angle3 = cal_ang(back_average,root_tail_average,mid_tail_average)
        
        df_copy.loc[index,'nose-neck-back'] = angle1
        df_copy.loc[index,'neck-back-tail'] = angle2
        #df_copy.loc[index,'back-tail1-tail2'] = angle3
        

    return(df_copy)
        

def average_FeA(FeA,ske):
    
    FeA_copy = FeA.copy()
    ske_copy = ske.copy()
    
    num = 0
    for i in FeA_copy.index:
        mv = FeA_copy.loc[i,'revised_movement_label']
        start = FeA_copy.loc[i,'segBoundary_start']
        end = FeA_copy.loc[i,'segBoundary_end']
        
        average_nose_x = ske.loc[start:end,'nose_x'].mean()
        average_nose_y = ske.loc[start:end,'nose_y'].mean()
        average_nose_z = ske.loc[start:end,'nose_z'].mean()
    
        average_neck_x = ske.loc[start:end,'neck_x'].mean()
        average_neck_y = ske.loc[start:end,'neck_y'].mean()
        average_neck_z = ske.loc[start:end,'neck_z'].mean()
        
        average_back_x = ske.loc[start:end,'back_x'].mean()
        average_back_y = ske.loc[start:end,'back_y'].mean()
        average_back_z = ske.loc[start:end,'back_z'].mean()
    
        average_root_tail_x = ske.loc[start:end,'root_tail_x'].mean()
        average_root_tail_y = ske.loc[start:end,'root_tail_y'].mean()
        average_root_tail_z = ske.loc[start:end,'root_tail_z'].mean()
                
        #average_mid_tail_x = ske.loc[start:end,'mid_tail_x'].mean()
        #average_mid_tail_y = ske.loc[start:end,'mid_tail_y'].mean()
        
        angle1 = cal_ang([average_nose_x,average_nose_y,average_nose_z], [average_neck_x,average_neck_y,average_neck_z], [average_back_x,average_back_y,average_back_z])
        angle2 = cal_ang([average_neck_x,average_neck_y,average_neck_z], [average_back_x,average_back_y,average_back_z],[average_root_tail_x,average_root_tail_y,average_root_tail_z])
        
        FeA_copy.loc[i,'average_back_x'] = average_back_x
        FeA_copy.loc[i,'average_back_y'] = average_back_y
        
        FeA_copy.loc[i,'average_nose_x'] = average_nose_x
        FeA_copy.loc[i,'average_nose_y'] = average_nose_y
        
        FeA_copy.loc[i,'average_neck_x'] = average_neck_x
        FeA_copy.loc[i,'average_neck_y'] = average_neck_y
        
        FeA_copy.loc[i,'average_root_tail_x'] = average_root_tail_x
        FeA_copy.loc[i,'average_root_tail_y'] = average_root_tail_y
        
        #FeA_copy.loc[i,'average_mid_tail_x'] = average_mid_tail_x
        #FeA_copy.loc[i,'average_mid_tail_y'] = average_mid_tail_y
        
        FeA_copy.loc[i,'nose_neck_back'] = angle1
        FeA_copy.loc[i,'neck_back_tail'] = angle2
    #df_output = FeA_copy[['revised_movement_label','nose_neck_back','neck_back_tail']]
    return(FeA_copy)


ske_data_list = []
for index in plot_dataset.index:
    video_index = plot_dataset.loc[index,'video_index']
    ExperimentCondition = plot_dataset.loc[index,'ExperimentCondition']
    LightingCondition = plot_dataset.loc[index,'LightingCondition']
    gender = plot_dataset.loc[index,'gender']
    ske_data = pd.read_csv(Skeleton_path[video_index])
    FeA_data = pd.read_csv(Feature_space_path[video_index])
    average_FeA_Ske = average_FeA(FeA_data,ske_data)
    average_FeA_Ske['gender'] = gender
    average_FeA_Ske['group'] = ExperimentCondition + '_' +  LightingCondition
    ske_data_list.append(average_FeA_Ske)


for index in plot_dataset2.index:
    video_index = plot_dataset2.loc[index,'video_index']
    ExperimentCondition = plot_dataset2.loc[index,'ExperimentCondition']
    LightingCondition = plot_dataset2.loc[index,'LightingCondition']
    gender = plot_dataset2.loc[index,'gender']
    ske_data = pd.read_csv(Skeleton_path[video_index])
    FeA_data = pd.read_csv(Feature_space_path[video_index])
    average_FeA_Ske = average_FeA(FeA_data,ske_data)
    average_FeA_Ske['gender'] = gender
    average_FeA_Ske['group'] = ExperimentCondition + '_' +  LightingCondition
    ske_data_list.append(average_FeA_Ske)

all_df = pd.concat(ske_data_list,axis=0)
all_df.reset_index(drop=True,inplace=True)

Nose_neck_back_Angle_dict = {}
for mv in exploration:
    mv_df = all_df[all_df['revised_movement_label']==mv]
    for group in mv_df['group'].unique():
        col_name =  mv + '_' + group
        values = mv_df[mv_df['group']==group]['nose_neck_back'].values
        Nose_neck_back_Angle_dict.setdefault(col_name,values)

Nose_neck_back_Angle = pd.DataFrame.from_dict(Nose_neck_back_Angle_dict,orient='index').T
Nose_neck_back_Angle.to_csv(r'{}\Nose_neck_back_Angle.csv'.format(output_dir))

Neck_back_tail_Angle_dict = {}
for mv in exploration:
    mv_df = all_df[all_df['revised_movement_label']==mv]
    for group in mv_df['group'].unique():
        col_name =  mv + '_' + group
        values = mv_df[mv_df['group']==group]['neck_back_tail'].values
        Neck_back_tail_Angle_dict.setdefault(col_name,values)

Neck_back_tail_Angle = pd.DataFrame.from_dict(Nose_neck_back_Angle_dict,orient='index').T
Neck_back_tail_Angle.to_csv(r'{}\Neck_back_tail_Angle.csv'.format(output_dir))



fig, ax = plt.subplots(ncols=1,nrows=2,figsize=(10,9),dpi=300,sharex=True)

sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color':'black', 'ls': '-', 'lw': 4,'label': '_mean_'},
            medianprops={'visible': False},
            whiskerprops={'visible': True},
            #hue_order=['naive','stress'],
            order=exploration,
            zorder=10,
            x="revised_movement_label",
            y="nose_neck_back",
            hue='group',
            data=all_df,
            showfliers=False,
            showbox=True,
            showcaps=True,
            legend=False,
            palette=[dataset1_color,dataset2_color],
            linewidth = 3,
            linecolor= 'black',
            width = 0.5,
            ax=ax[0])

sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color':'black', 'ls': '-', 'lw': 2,'label': '_mean_'},
            medianprops={'visible': False},
            whiskerprops={'visible': True},
            #hue_order=['naive','stress'],
            order=exploration,
            zorder=10,
            x="revised_movement_label",
            y="neck_back_tail",
            hue='group',
            data=all_df,
            showfliers=False,
            showbox=True,
            showcaps=True,
            legend=False,
            palette=[dataset1_color,dataset2_color],
            linewidth = 3,
            linecolor= 'black',
            width = 0.5,
            ax=ax[1])


ax[0].set_xlabel('')
ax[0].set_ylabel('')
ax[1].set_xlabel('')
ax[1].set_ylabel('')


ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].spines['bottom'].set_visible(False)
ax[0].spines['left'].set_linewidth(3)
ax[0].set_yticks(np.arange(120,181,30))
ax[0].yaxis.set_major_formatter(plt.NullFormatter())
ax[0].tick_params(length=7,width=3,axis='y')
ax[0].tick_params(length=0,width=0,axis='x')


ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].spines['bottom'].set_visible(False)
ax[1].spines['left'].set_linewidth(3)
ax[1].set_yticks(np.arange(120,181,30))
ax[1].yaxis.set_major_formatter(plt.NullFormatter())
ax[1].tick_params(length=7,width=3,axis='y')
ax[1].tick_params(length=0,width=0,axis='x')

ax[1].xaxis.set_ticklabels([])

fig.tight_layout()
plt.subplots_adjust(wspace =0, hspace = 1.5)

plt.savefig('{}/body_angle.png'.format(output_dir),transparent=True,dpi=300)





























# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:43:40 2024

@author: Jialin Ye
@institution: SIAT
@Contact_email: jl.ye@siat.ac.cn

"""

import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as signal
import math
import joypy
import mpl_scatter_density
import matplotlib.colors as mcolors

InputData_path_dir =  r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\01_BehaviorAtlas_collated_data"
InputData_path_dir2 = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure2_skeleton_kinematics(related to sFigure2F)\movement_parametter\grooming'
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
        

plot_dataset = stress_animal_info
dataset_name = 'stress'
plot_dataset_male = plot_dataset[plot_dataset['gender']=='male']
plot_dataset_female = plot_dataset[plot_dataset['gender']=='female']
dataset_color = get_color(dataset_name)


def cal_ang(point_1, point_2, point_3):
    a=math.sqrt((point_2[0]-point_3[0])*(point_2[0]-point_3[0])+(point_2[1]-point_3[1])*(point_2[1] - point_3[1]))
    b=math.sqrt((point_1[0]-point_3[0])*(point_1[0]-point_3[0])+(point_1[1]-point_3[1])*(point_1[1] - point_3[1]))
    c=math.sqrt((point_1[0]-point_2[0])*(point_1[0]-point_2[0])+(point_1[1]-point_2[1])*(point_1[1]-point_2[1]))
    A=math.degrees(math.acos((a*a-b*b-c*c)/(-2*b*c)))
    B=math.degrees(math.acos((b*b-a*a-c*c)/(-2*a*c)))
    C=math.degrees(math.acos((c*c-a*a-b*b)/(-2*a*b)))
    return (B)

def Srotate(angle,valuex,valuey,pointx,pointy):  ### 顺时针
    valuex = np.array(valuex)  
    valuey = np.array(valuey)  
    sRotatex = (valuex-pointx)*math.cos(angle) + (valuey-pointy)*math.sin(angle) + pointx  
    sRotatey = (valuey-pointy)*math.cos(angle) - (valuex-pointx)*math.sin(angle) + pointy  
    return((sRotatex,sRotatey))
def Nrotate(angle,valuex,valuey,pointx,pointy):  
    valuex = np.array(valuex)  
    valuey = np.array(valuey)  
    nRotatex = (valuex-pointx)*math.cos(angle) - (valuey-pointy)*math.sin(angle) + pointx
    nRotatey = (valuex-pointx)*math.sin(angle) + (valuey-pointy)*math.cos(angle) + pointy
    return((nRotatex,nRotatey))


def cal_nose_distribution(df):
    df_copy = df.copy()
    
    for index in df_copy.index:
        
        nose_x = df_copy.loc[index,'nose_x']
        nose_y = df_copy.loc[index,'nose_y']
        #nose_z = df_copy.loc[index,'nose_y']
        
        neck_x = df_copy.loc[index,'neck_x']
        neck_y = df_copy.loc[index,'neck_y']
        #neck_z = df.loc[index,'neck_z']
        
        back_x = df_copy.loc[index,'back_x']
        back_y = df_copy.loc[index,'back_y']
        #back_z = df.loc[index,'back_z']
        
        root_tail_x = df_copy.loc[index,'root_tail_x']
        root_tail_y = df_copy.loc[index,'root_tail_y']

        
        ## calculate angle
        if nose_x > 0:
            angle1 = 180 - cal_ang([nose_x,nose_y], [back_x,back_y],[root_tail_x,root_tail_y])
        else:
            angle1 = 180  + cal_ang([nose_x,nose_y], [back_x,back_y],[root_tail_x,root_tail_y])
        #angle2 = cal_ang([neck_x,neck_y,neck_z], [back_x,back_y,back_z],[root_tail_x,root_tail_y,root_tail_z])
        #angle3 = cal_ang(back_average,root_tail_average,mid_tail_average)
        
        df_copy.loc[index,'nose-back-tail'] = angle1
        #df_copy.loc[index,'neck-back-tail'] = angle2
        #df_copy.loc[index,'back-tail1-tail2'] = angle3
    return(df_copy)
        

    
ske_data_list = []
for index in plot_dataset.index:
    video_index = plot_dataset.loc[index,'video_index']
    ExperimentCondition = plot_dataset.loc[index,'ExperimentCondition']
    LightingCondition = plot_dataset.loc[index,'LightingCondition']
    gender = plot_dataset.loc[index,'gender']
    ske_data = pd.read_csv(Skeleton_path[video_index])
    #ske_data = flitter(ske_data)
    Mov_data = pd.read_csv(Movement_Label_path[video_index],usecols=['original_label','revised_movement_label','smooth_speed'])
    conbime_data = pd.concat([Mov_data,ske_data],axis=1)
    conbime_data['gender'] = gender
    conbime_data['group'] = ExperimentCondition + '_' +  LightingCondition
    ske_data_list.append(conbime_data)

dataset_skeleton = pd.concat(ske_data_list,axis=0)
dataset_skeleton.reset_index(drop=True,inplace=True)


grooming_skeleton = dataset_skeleton[dataset_skeleton['revised_movement_label']=='grooming']
grooming_skeleton = cal_nose_distribution(grooming_skeleton)


def count_distribution_range(df,step):
    
    df_count = pd.DataFrame()
    num = 0
    start_angle = 0
    end_angle = 0 
    for i in range(step,361,step):
        end_angle = i
        #print(start_angle,end_angle)
        value = len(df[(df['nose-back-tail']>start_angle) & (df['nose-back-tail']<end_angle)])
        df_count.loc[num,'angle'] = i
        df_count.loc[num,'count'] = value
        num += 1
        start_angle = end_angle
    df_count['percentage'] = df_count['count']/df_count['count'].sum()
    
    return(df_count)
        

step = 10
dataset1_angle_distribution = count_distribution_range(grooming_skeleton,step)

angles=np.arange(0,2*np.pi,2*np.pi/dataset1_angle_distribution.shape[0])
radius=np.array(dataset1_angle_distribution['percentage'])

fig=plt.figure(figsize=(10,10),dpi=300)
fig = fig.add_subplot(projection='polar')
fig.grid(axis='both', linewidth =5,zorder=0,color='#424242')
fig.set_theta_offset(np.pi/2)
fig.set_theta_direction(-1)
fig.set_rlabel_position(90)

plt.bar(angles,radius,width=2*np.pi/dataset1_angle_distribution.shape[0],edgecolor='black',lw=2,label=dataset1_angle_distribution['angle'],color=dataset_color,zorder=3)
plt.ylim(0,0.08)
plt.yticks(np.arange(0,0.08,0.02))

fig.yaxis.set_major_formatter(plt.NullFormatter())
fig.xaxis.set_major_formatter(plt.NullFormatter())
fig.spines['polar'].set_linewidth(10)
plt.savefig('{}/{}_grooming_direction.png'.format(output_dir,dataset_name),transparent=True,dpi=300)


nose_y_max = np.sort(grooming_skeleton['nose_y'],axis=-1)[::-1][:1000].mean()                     ####
root_tail_y_min = np.sort(grooming_skeleton['root_tail_y'],axis=-1)[:1000].mean()

left_hind_limb_x_min = np.sort(grooming_skeleton['left_hind_limb_x'],axis=-1)[:1000].mean()
left_hind_limb_y_average = grooming_skeleton['left_hind_limb_y'].mean()
left_hind_limb_z_average = grooming_skeleton['left_hind_limb_z'].mean()


right_hind_limb_x_max = np.sort(grooming_skeleton['right_hind_limb_x'])[::-1][:1000].mean()
right_hind_limb_y_average = grooming_skeleton['right_hind_limb_y'].mean()
right_hind_limb_z_average = grooming_skeleton['right_hind_limb_z'].mean()

nose_z_average = grooming_skeleton['nose_z'].mean()

back_x_average = grooming_skeleton['back_x'].mean()
back_y_average = grooming_skeleton['back_y'].mean()
back_z_average = grooming_skeleton['back_z'].mean()

root_tail_x_average = grooming_skeleton['root_tail_x'].mean()
root_tail_y_average = grooming_skeleton['root_tail_y'].mean()
root_tail_z_average = grooming_skeleton['root_tail_z'].mean()


##color

nose_color = '#9C27B0'
root_tail_color = '#009688'
back_color = '#66BB6A'

left_limb_color = '#546E7A'
right_limb_color = '#546E7A'


### X,Y
fig = plt.figure(figsize=(4,4),constrained_layout=True,dpi=300)
ax = fig.add_subplot(1,1,1, projection='scatter_density')
#norm = mcolors.TwoSlopeNorm(vmin=-1, vmax =3000, vcenter=0)
ax.scatter_density(grooming_skeleton['nose_x'],grooming_skeleton['nose_y'], color=dataset_color,alpha=1)
ax.set_xlim(-81,81)
ax.set_ylim(-81,81)
ax.plot([left_hind_limb_x_min,right_hind_limb_x_max],[(left_hind_limb_y_average+right_hind_limb_y_average)/2,(left_hind_limb_y_average+right_hind_limb_y_average)/2],lw=6,color='black')
ax.scatter(left_hind_limb_x_min,(left_hind_limb_y_average+right_hind_limb_y_average)/2,zorder=3,c=left_limb_color,s=100)
ax.scatter(right_hind_limb_x_max,(left_hind_limb_y_average+right_hind_limb_y_average)/2,zorder=3,c=right_limb_color,s=100)

ax.plot([0,0],[root_tail_y_min,nose_y_max],lw=6,color='black')
ax.scatter(0,root_tail_y_min,zorder=3,c=root_tail_color,s=100)
ax.scatter(0,nose_y_max,zorder=3,c=nose_color,s=100)
for i in range(-80,81,20):
    ax.axhline(i,lw=2,c='#90A4AE',zorder=-1,linestyle='--',)
for i in range(-80,81,20):
    ax.axvline(i,lw=2,c='#90A4AE',zorder=-1,linestyle='--',)

plt.axis('off')
plt.savefig('{}/{}_grooming_nose_XY_distribution.png'.format(output_dir,dataset_name),transparent=True,dpi=300)


### Y,Z

fig = plt.figure(figsize=(4,4),constrained_layout=True,dpi=300)
ax = fig.add_subplot(1,1,1, projection='scatter_density')
ax.grid(lw=2,linestyle = '--',zorder=0,alpha=0.5)
#norm = mcolors.TwoSlopeNorm(vmin=-1, vmax =3000, vcenter=0)
ax.scatter_density(grooming_skeleton['nose_y'],grooming_skeleton['nose_z'], color=dataset_color,alpha=1)
ax.set_xlim(-81,81)
ax.set_ylim(-61,101)

ax.plot([back_y_average,nose_y_max],[back_z_average,nose_z_average],lw=6,color='black')
ax.plot([back_y_average,root_tail_y_min],[back_z_average,root_tail_z_average],lw=6,color='black')

ax.scatter(nose_y_max,nose_z_average,zorder=3,c=nose_color,s=100)
ax.scatter(back_y_average,back_z_average,zorder=3,c=back_color,s=100)
ax.scatter(root_tail_y_min,root_tail_z_average,zorder=3,c=root_tail_color,s=100)
for i in range(-60,101,20):
    ax.axhline(i,lw=2,c='#90A4AE',zorder=-1,linestyle='--',)
for i in range(-80,81,20):
    ax.axvline(i,lw=2,c='#90A4AE',zorder=-1,linestyle='--',)

plt.axis('off')
plt.savefig('{}/{}_grooming_nose_YZ_distribution.png'.format(output_dir,dataset_name),transparent=True,dpi=300)





### X,Z

fig = plt.figure(figsize=(4,4),constrained_layout=True,dpi=300)
ax = fig.add_subplot(1,1,1, projection='scatter_density',zorder=10)

#norm = mcolors.TwoSlopeNorm(vmin=-1, vmax =3000, vcenter=0)
ax.scatter_density(grooming_skeleton['nose_x'],grooming_skeleton['nose_z'], color=dataset_color,alpha=1)
ax.set_xlim(-81,81)
ax.set_ylim(-61,101)

ax.plot([left_hind_limb_x_min,right_hind_limb_x_max],[(left_hind_limb_z_average+right_hind_limb_z_average)/2,(left_hind_limb_z_average+right_hind_limb_z_average)/2],lw=6,color='black')

ax.scatter(left_hind_limb_x_min,(left_hind_limb_z_average+right_hind_limb_z_average)/2,zorder=3,c=left_limb_color,s=100)
ax.scatter(right_hind_limb_x_max,(left_hind_limb_z_average+right_hind_limb_z_average)/2,zorder=3,c=right_limb_color,s=100)

ax.plot([back_x_average,root_tail_x_average],[back_z_average,root_tail_z_average],lw=6,color='black')

ax.scatter(back_x_average,back_z_average,zorder=3,c=back_color,s=100)
ax.scatter(root_tail_x_average,root_tail_z_average,zorder=3,c=root_tail_color,s=100)

for i in range(-20,101,20):
    ax.axhline(i,lw=2,c='#90A4AE',zorder=-1,linestyle='--',)

for i in range(-80,101,20):
    ax.plot((i,i),(-20,100),lw=2,c='#90A4AE',zorder=-1,linestyle='--',)

plt.axis('off')
plt.savefig('{}/{}_grooming_nose_XZ_distribution.png'.format(output_dir,dataset_name),transparent=True,dpi=300)


































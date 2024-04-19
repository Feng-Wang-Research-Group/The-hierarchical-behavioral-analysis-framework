# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:27:13 2024

@author: Jialin Ye
@institution: SIAT
@Contact_email: jl.ye@siat.ac.cn

"""

import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import joypy
from scipy.optimize import curve_fit
from scipy.stats import bootstrap


InputData_path_dir =  r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\01_BehaviorAtlas_collated_data"
InputData_path_dir2 = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure2_skeleton_kinematics(related to sFigure2)\movement_parametter\turning'
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
Feature_space_path = get_path(InputData_path_dir2,'revised_Feature_Space.csv')
Skeleton_path = get_path(InputData_path_dir,'normalized_skeleton_XYZ.csv')
loc_path = get_path(InputData_path_dir,'normalized_coordinates_back_XY.csv')
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


turning = ['left_turning','right_turning']

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
dataset_name = 'Morning_lightOn'
plot_dataset_male = plot_dataset[plot_dataset['gender']=='male']
plot_dataset_female = plot_dataset[plot_dataset['gender']=='female']
dataset_color = get_color(dataset_name)

plot_dataset2 = stress_animal_info
dataset_name2 = 'stress_animal'
plot_dataset_male2 = plot_dataset2[plot_dataset2['gender']=='male']
plot_dataset_female2 = plot_dataset2[plot_dataset2['gender']=='female']
dataset_color2 = get_color(dataset_name2)

##### From: Bézier Interpolation - Create smooth shapes using Bézier curves - 2024.01.31
# find the a & b points
def get_bezier_coef(points):
    # since the formulas work given that we have n+1 points
    # then n must be this:
    n = len(points) - 1

    # build coefficents matrix
    C = 4 * np.identity(n)
    np.fill_diagonal(C[1:], 1)
    np.fill_diagonal(C[:, 1:], 1)
    C[0, 0] = 2
    C[n - 1, n - 1] = 7
    C[n - 1, n - 2] = 2

    # build points vector
    P = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
    P[0] = points[0] + 2 * points[1]
    P[n - 1] = 8 * points[n - 1] + points[n]

    # solve system, find a & b
    A = np.linalg.solve(C, P)
    B = [0] * n
    for i in range(n - 1):
        B[i] = 2 * points[i + 1] - A[i + 1]
    B[n - 1] = (A[n - 1] + points[n]) / 2

    return A, B

# returns the general Bezier cubic formula given 4 control points
def get_cubic(a, b, c, d):
    return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d

# return one cubic curve for each consecutive points
def get_bezier_cubic(points):
    A, B = get_bezier_coef(points)
    return [
        get_cubic(points[i], A[i], B[i], points[i + 1])
        for i in range(len(points) - 1)
    ]

# evalute each cubic curve on the range [0, 1] sliced in n points
def evaluate_bezier(points, n):
    curves = get_bezier_cubic(points)
    return np.array([fun(t) for fun in curves for t in np.linspace(0, 1, n)])


def cal_ang(point_1, point_2, point_3):
    a=math.sqrt((point_2[0]-point_3[0])*(point_2[0]-point_3[0])+(point_2[1]-point_3[1])*(point_2[1] - point_3[1]))
    b=math.sqrt((point_1[0]-point_3[0])*(point_1[0]-point_3[0])+(point_1[1]-point_3[1])*(point_1[1] - point_3[1]))
    c=math.sqrt((point_1[0]-point_2[0])*(point_1[0]-point_2[0])+(point_1[1]-point_2[1])*(point_1[1]-point_2[1]))
    A=math.degrees(math.acos((a*a-b*b-c*c)/(-2*b*c)))
    B=math.degrees(math.acos((b*b-a*a-c*c)/(-2*a*c)))
    C=math.degrees(math.acos((c*c-a*a-b*b)/(-2*a*b)))
    return (B)

def add_turning_point(FeA,ske,loc,mv_type):
    
    if mv_type == 'left_turning':
        direction = 'descending'
    elif mv_type == 'right_turning':
        direction = 'ascending'
    
    FeA_copy = FeA.copy()
    ske_copy = ske.copy()
    loc_copy = loc.copy()
    
    num = 0
    for i in FeA_copy.index:
        mv = FeA_copy.loc[i,'revised_movement_label']
        start = FeA_copy.loc[i,'segBoundary_start']
        end = FeA_copy.loc[i,'segBoundary_end']
              
        average_back_x = ske.loc[start:end,'back_x'].mean()
        average_back_y = ske.loc[start:end,'back_y'].mean()
    
        average_nose_x = ske.loc[start:end,'nose_x'].mean()
        average_nose_y = ske.loc[start:end,'nose_y'].mean()
    
        average_neck_x = ske.loc[start:end,'neck_x'].mean()
        average_neck_y = ske.loc[start:end,'neck_y'].mean()
        
        average_root_tail_x = ske.loc[start:end,'root_tail_x'].mean()
        average_root_tail_y = ske.loc[start:end,'root_tail_y'].mean()
        
        average_mid_tail_x = ske.loc[start:end,'mid_tail_x'].mean()
        average_mid_tail_y = ske.loc[start:end,'mid_tail_y'].mean()
        
        if average_nose_x == 0:
            pass
        
        else:
        
            if average_nose_x < 0:
                point3 = [-1,0]
                angle1 = cal_ang([average_nose_x,average_nose_y],[average_back_x,average_back_y],point3)
            else:
                point3 = [1,0]
                angle1 = cal_ang([average_nose_x,average_nose_y],[average_back_x,average_back_y],point3)
            
            x_start = loc_copy.loc[start,'back_x']
            x_end = loc_copy.loc[end-1,'back_x']
            x_speed = abs(x_end-x_start)/((end-start+1)/30)                             ### v = S/t  mm/s
            
            y_start = loc_copy.loc[start,'back_y']
            y_end = loc_copy.loc[end-1,'back_y']
            y_speed = abs(y_end-y_start)/((end-start+1)/30)
        
            FeA_copy.loc[i,'average_back_x'] = average_back_x
            FeA_copy.loc[i,'average_back_y'] = average_back_y
            
            FeA_copy.loc[i,'average_nose_x'] = average_nose_x
            FeA_copy.loc[i,'average_nose_y'] = average_nose_y
            
            FeA_copy.loc[i,'average_neck_x'] = average_neck_x
            FeA_copy.loc[i,'average_neck_y'] = average_neck_y
            
            FeA_copy.loc[i,'average_root_tail_x'] = average_root_tail_x
            FeA_copy.loc[i,'average_root_tail_y'] = average_root_tail_y
            
            FeA_copy.loc[i,'average_mid_tail_x'] = average_mid_tail_x
            FeA_copy.loc[i,'average_mid_tail_y'] = average_mid_tail_y
            
            FeA_copy.loc[i,'x_speed'] = x_speed
            FeA_copy.loc[i,'y_speed'] = y_speed
            
            FeA_copy.loc[i,'turning_angle'] = angle1
    return(FeA_copy)


def cal_distribution(arr):
    data = (arr,)
    #res = bootstrap(data, np.std, confidence_level=0.9,random_state=rng)
    res = bootstrap(data, np.mean, axis=-1, confidence_level=0.95, n_resamples=100, random_state=1)
    ci_l, ci_u = res.confidence_interval
    resample_values = res.bootstrap_distribution
    return(ci_l, ci_u,resample_values)


left_turning_df_list = []
right_turning_df_list = []

for index in plot_dataset.index:
    video_index = plot_dataset.loc[index,'video_index']
    ExperimentCondition = plot_dataset.loc[index,'ExperimentCondition']
    LightingCondition = plot_dataset.loc[index,'LightingCondition']
    gender = plot_dataset.loc[index,'gender']
    ske_data = pd.read_csv(Skeleton_path[video_index])
    loc_data = pd.read_csv(loc_path[video_index],index_col=0)
    FeA_data = pd.read_csv(Feature_space_path[video_index])
    
    left_turning_df = FeA_data[FeA_data['revised_movement_label']=='left_turning']
    left_turning_df = add_turning_point(left_turning_df,ske_data,loc_data,'left_turning')
    left_turning_df_list.append(left_turning_df)

    right_turning_df = FeA_data[FeA_data['revised_movement_label']=='right_turning']
    right_turning_df = add_turning_point(right_turning_df,ske_data,loc_data,'right_turning')
    right_turning_df_list.append(right_turning_df)


all_left_turning_df = pd.concat(left_turning_df_list,axis=0)
all_left_turning_df.reset_index(drop=True,inplace=True)

all_right_turning_df = pd.concat(right_turning_df_list,axis=0)
all_right_turning_df.reset_index(drop=True,inplace=True)


left_turning_df_list2 = []
right_turning_df_list2 = []

for index in plot_dataset2.index:
    video_index = plot_dataset2.loc[index,'video_index']
    ExperimentCondition = plot_dataset2.loc[index,'ExperimentCondition']
    LightingCondition = plot_dataset2.loc[index,'LightingCondition']
    gender = plot_dataset2.loc[index,'gender']
    ske_data = pd.read_csv(Skeleton_path[video_index])
    loc_data = pd.read_csv(loc_path[video_index],index_col=0)
    FeA_data = pd.read_csv(Feature_space_path[video_index])
    
    left_turning_df2 = FeA_data[FeA_data['revised_movement_label']=='left_turning']
    left_turning_df2 = add_turning_point(left_turning_df2,ske_data,loc_data,'left_turning')
    left_turning_df_list2.append(left_turning_df2)

    right_turning_df2 = FeA_data[FeA_data['revised_movement_label']=='right_turning']
    right_turning_df2 = add_turning_point(right_turning_df2,ske_data,loc_data,'right_turning')
    right_turning_df_list2.append(right_turning_df2)


all_left_turning_df2 = pd.concat(left_turning_df_list2,axis=0)
all_left_turning_df2.reset_index(drop=True,inplace=True)

all_right_turning_df2 = pd.concat(right_turning_df_list2,axis=0)
all_right_turning_df2.reset_index(drop=True,inplace=True)


output_df = pd.concat([all_left_turning_df,all_right_turning_df,all_left_turning_df2,all_right_turning_df2],axis=0)
output_df.to_csv(r'{}/{}_{}_turning_para.csv'.format(output_dir,dataset_name,dataset_name2))


nose_color = '#9C27B0'
neck_color = '#B0BEC5'
root_tail_color = '#009688'
back_color = '#66BB6A'

left_limb_color = '#546E7A'
right_limb_color = '#546E7A'


def plot(plot_df,dataset_name,line_color,mv_type):
    fig,ax = plt.subplots(figsize=(10,10),dpi=300)
    
    #for i in all_left_turning_df.index:
    back_x = plot_df['average_back_x'].mean()
    back_y = plot_df['average_back_y'].mean()
    
    neck_x = plot_df['average_neck_x'].mean()
    neck_y = plot_df['average_neck_y'].mean()
    
    nose_x = plot_df['average_nose_x'].mean()
    nose_y = plot_df['average_nose_y'].mean()
    
    root_tail_x = plot_df['average_root_tail_x'].mean()
    root_tail_y = plot_df['average_root_tail_y'].mean()
    
    x_speed = plot_df['x_speed'].mean()
    y_speed = plot_df['y_speed'].mean()
    angle_nose_back = plot_df['turning_angle'].mean()
    
    print(dataset_name,',',mv_type)
    print(x_speed,y_speed,angle_nose_back)
    
    ax.scatter(nose_x,nose_y,c=nose_color,s=200,zorder = 10)
    ax.scatter(back_x,back_y,c=back_color,s=200,zorder = 10)
    ax.scatter(neck_x,neck_y,c=neck_color,s=200,zorder = 10)
    ax.scatter(root_tail_x,root_tail_y,c=root_tail_color,s=200,zorder = 10)
    
    ax.plot([nose_x,back_x],[nose_y,back_y],c='black',lw=4,zorder = 9)
    
    
    ax.arrow(0,0,0,y_speed,length_includes_head=True,head_width=1, fc='black', ec='black',lw=8)
    
    if mv_type == 'left_turning':
        ax.arrow(0,0,-x_speed,0,length_includes_head=True,head_width=1, fc='black', ec='black',lw=8)
    elif mv_type == 'right_turning':
        ax.arrow(0,0,x_speed,0,length_includes_head=True,head_width=1, fc='black', ec='black',lw=8)
    
    # =============================================================================
    # ci_l, ci_u,resample_values = cal_distribution(all_left_turning_df['x_speed'].values)
    # ax.plot([-ci_l, -ci_u],[2,2],c='#F8BBD0',lw=4,linestyle='-',zorder=1,alpha=0.9)
    # 
    # ci_l2, ci_u2,resample_values2 = cal_distribution(all_left_turning_df['y_speed'].values)
    # ax.plot([-2,-2],[ci_l2, ci_u2],c='#F8BBD0',lw=4,linestyle='-',zorder=1,alpha=0.9)
    # 
    # ax.scatter(-2,y_speed,c='grey')
    # ax.scatter(-x_speed,2,c='pink')
    # =============================================================================
    
    
    
    points = np.array([[nose_x,nose_y],[neck_x,neck_y],[back_x,back_y],[root_tail_x,root_tail_y]])
    
    # fit the points with Bezier interpolation
    # use 50 points between each consecutive points to draw the curve
    path = evaluate_bezier(points, 50)
    
    # extract x & y coordinates of points
    x, y = points[:,0], points[:,1]
    px, py = path[:,0], path[:,1]
    
    
    plt.plot(px, py, color = line_color,lw=10)
    plt.plot(x, y, 'ro')
    
    #ax.boxplot(all_left_turning_df['y_speed'].values,positions=[10],widths=8,meanline=True,autorange=True,bootstrap =10000,showmeans=True,showfliers=False)
    #ax.boxplot(-all_left_turning_df['x_speed'].values,positions=[-10],widths=8,meanline=True,autorange=True,vert =False,bootstrap =10000)
    
    ax.set_xlim(-100,100)
    ax.set_ylim(-100,100)
    plt.axis('off')
    plt.savefig('{}/{}_{}.png'.format(output_dir,dataset_name,mv_type),transparent=True,dpi=300)


def plot_BodyBendAngle_topview(plot_df,dataset_name,mv_type):
    
    select_df = plot_df.sample(n=1000, random_state=22)
    select_df = select_df[['average_nose_x','average_nose_y','average_neck_x','average_neck_y','average_back_x','average_back_y','average_root_tail_x','average_root_tail_y','average_mid_tail_x','average_mid_tail_y']]
    select_df.reset_index(drop=True,inplace=True)
    select_df.loc[len(select_df.index)-1] = select_df.mean(axis=0)
    
    nose_average = [select_df.loc[len(select_df.index)-1,'average_nose_x'],select_df.loc[len(select_df.index)-1,'average_nose_y']]
    neck_average = [select_df.loc[len(select_df.index)-1,'average_neck_x'],select_df.loc[len(select_df.index)-1,'average_neck_y']]
    back_average = [select_df.loc[len(select_df.index)-1,'average_back_x'],select_df.loc[len(select_df.index)-1,'average_back_y']]
    root_tail_average = [select_df.loc[len(select_df.index)-1,'average_root_tail_x'],select_df.loc[len(select_df.index)-1,'average_root_tail_y']]
    mid_tail_average = [select_df.loc[len(select_df.index)-1,'average_mid_tail_x'],select_df.loc[len(select_df.index)-1,'average_mid_tail_y']]
    
    fig,ax = plt.subplots(figsize=(10,10),dpi=300)
    for i in select_df.index:
        
        nose_x = select_df.loc[i,'average_nose_x']
        nose_y = select_df.loc[i,'average_nose_y']
        neck_x = select_df.loc[i,'average_neck_x']
        neck_y = select_df.loc[i,'average_neck_y']
        back_x = select_df.loc[i,'average_back_x']
        back_y = select_df.loc[i,'average_back_y']
        root_tail_x = select_df.loc[i,'average_root_tail_x']
        root_tail_y = select_df.loc[i,'average_root_tail_y']
        mid_tail_x = select_df.loc[i,'average_mid_tail_x']
        mid_tail_y = select_df.loc[i,'average_mid_tail_y']
        
        ax.scatter(nose_x,nose_y,s=60,c='#9C27B0',alpha = 0.1)
        ax.scatter(neck_x,neck_y,s=60,c='#03A9F4',alpha = 0.1)
        ax.scatter(back_x,back_y,s=60,c='#66BB6A',alpha = 0.1)
        ax.scatter(root_tail_x,root_tail_y,s=60,c='#009688',alpha = 0.1)
        ax.scatter(mid_tail_x,mid_tail_y,s=60,c='#9E9D24',alpha = 0.1)
        
        ax.plot([nose_x,neck_x],[nose_y,neck_y],c='#90A4AE',alpha = 0.1,lw=1)
        ax.plot([neck_x,back_x],[neck_y,back_y],c='#90A4AE',alpha = 0.1,lw=1)
        ax.plot([back_x,root_tail_x],[back_y,root_tail_y],c='#90A4AE',alpha = 0.1,lw=1)
        ax.plot([root_tail_x,mid_tail_x],[root_tail_y,mid_tail_y],c='#90A4AE',alpha = 0.1,lw=1)
        
        
    
    ax.scatter(nose_average[0],nose_average[1],s=260,c='#9C27B0',ec='black',alpha = 1,zorder=10)
    ax.scatter(neck_average[0],neck_average[1],s=260,c='#03A9F4',ec='black',alpha = 1,zorder=10)
    ax.scatter(back_average[0],back_average[1],s=260,c='#66BB6A',ec='black',alpha = 1,zorder=10)
    ax.scatter(root_tail_average[0],root_tail_average[1],s=260,ec='black',c='#009688',alpha = 1,zorder=10)
    ax.scatter(mid_tail_average[0],mid_tail_average[1],s=260,c='#9E9D24',ec='black',alpha = 1,zorder=10)
    
    ax.plot([nose_average[0],neck_average[0]],[nose_average[1],neck_average[1]],c='black',alpha = 1,lw=4,zorder=10)
    ax.plot([neck_average[0],back_average[0]],[neck_average[1],back_average[1]],c='black',alpha = 1,lw=4,zorder=10)
    ax.plot([back_average[0],root_tail_average[0]],[back_average[1],root_tail_average[1]],c='black',alpha = 1,lw=4,zorder=10)
    ax.plot([root_tail_average[0],mid_tail_average[0]],[root_tail_average[1],mid_tail_average[1]],c='black',alpha = 1,lw=4,zorder=10)
    #plt.title(i,fontsize = 25)
    ax.set_xlim(-100,100)
    ax.set_ylim(-100,100)
    plt.axis('off')
    plt.savefig('{}/{}_{}__skeleton_Hangle.png'.format(output_dir,dataset_name,mv_type),transparent=True,dpi=300) 
    plt.show()


plot(all_left_turning_df,dataset_name,dataset_color,'left_turning')
plot(all_right_turning_df,dataset_name,dataset_color,'right_turning')

plot(all_left_turning_df2,dataset_name2,dataset_color2,'left_turning')
plot(all_right_turning_df2,dataset_name2,dataset_color2,'right_turning')


plot_BodyBendAngle_topview(all_left_turning_df,dataset_name,'left_turning')
plot_BodyBendAngle_topview(all_right_turning_df,dataset_name,'right_turning')

plot_BodyBendAngle_topview(all_left_turning_df2,dataset_name2,'left_turning')
plot_BodyBendAngle_topview(all_right_turning_df2,dataset_name2,'right_turning')
































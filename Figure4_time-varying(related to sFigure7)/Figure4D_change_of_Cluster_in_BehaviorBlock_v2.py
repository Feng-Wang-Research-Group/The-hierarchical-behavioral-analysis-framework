# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 20:53:56 2024

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


InputData_path_dir =  r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure4_time-varying(related to sFigure7)'


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


movement_order = ['running','trotting','left_turning','right_turning','walking','stepping','sniffing',
                  'rising','hunching','rearing','climbing','jumping','grooming','scratching','pausing',]

cluster_order = ['locomotion','exploration','maintenance','nap']    


cluster_color_dict={'locomotion':'#DC2543',                     
                     'exploration':'#009688',
                     'maintenance':'#973C8D',
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



movement_frequency_each_mice = []

dataset1 = Morning_lightOn_info
datasetname1 = 'Morning_lightOn'

count_variable = 'movement_cluster_label'   #revised_movement_label  movement_cluster_label

if count_variable == 'movement_cluster_label':
    count_order = cluster_order
    color_dict = cluster_color_dict

elif count_variable == 'revised_movement_label':
    count_order = movement_order
    color_dict = movement_color_dict



if datasetname1.startswith('Morning'):
    branch_point = [0,15,45,60]   # morning
elif datasetname1.startswith('stress') | datasetname1.startswith('Stress'):
    branch_point = [0,5,25,60]      #stress
elif datasetname1.startswith('Afternoon'):
    branch_point = [0,20,45,60]  #afternoon
elif datasetname1.startswith('Night_lightOn'):
    branch_point = [0,15,40,60]       #light_lightOn
elif datasetname1.startswith('Night_lightOff'):
    branch_point = [0,15,60]      #night_lightOff


def label_count(df,count_variable):
    data = df.copy()
    label_number = data[count_variable].value_counts()

    df_output = pd.DataFrame()
    num = 0
    for mv in count_order:
        df_output.loc[num,count_variable] = mv
        if not mv in label_number.index:
            df_output.loc[num,'label_frequency'] = 0
        else:
            df_output.loc[num,'label_frequency'] = label_number[mv] / label_number.values.sum()
        df_output.loc[num,'accumulative_time'] = df_output.loc[num,'label_frequency']*(time_window2-time_window1)
        num += 1
    return(df_output)

for index in range(1,len(branch_point)):
    x1 = index - 1 
    x2 = index
    time_window1 = branch_point[x1]
    time_window2 = branch_point[x2]
    
    for i in dataset1.index:
        video_index = dataset1.loc[i,'video_index']
        Movement_Label_file = pd.read_csv(Movement_Label_path[video_index])
        
        Movement_Label_file = Movement_Label_file.iloc[time_window1*30*60:time_window2*30*60,:]
        
        df_count = label_count(Movement_Label_file,count_variable)
        df_count['time'] = time_window2 #min_i
        movement_frequency_each_mice.append(df_count)
    #time_window1 = time_window2

df_out = pd.concat(movement_frequency_each_mice)


df_statistic = pd.DataFrame()


#for min_i in range(15,61,15):
for min_i in branch_point[1:]:
    for j in count_order:
        value = df_out.loc[(df_out[count_variable]==j)&(df_out['time']==min_i),'label_frequency'].values.mean()
        df_statistic.loc[j,min_i] = value

df_statistic.to_csv('{}/{}_temporal_{}_fraction.csv'.format(output_dir,datasetname1,count_variable))



length = np.diff(branch_point)

fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(12,4))

x1 = 0
x2 = 0
y1 = 0
y2 = 0


x1_list = []
x2_list = []
y_list = []

num = 0
for i in range(1,len(branch_point)):
    x2 = x1 + length[i-1]
    x1_list.append(x1)
    x2_list.append(x2)
    y_list.append(y2)
    for label in count_order:        
        value = df_statistic.loc[label,branch_point[i]]
        y2 -= value
        color = color_dict[label]
        ax.fill_between([x1,x2],[y1,y1],[y2,y2],color=color,ec='black',lw=2,zorder=3)
        
        x1_list.append(x1)
        x2_list.append(x2)
        y_list.append(y2)
        y1=y2      
    x1 = x2 + 20
    y1 = 0
    y2 = 0
    num += 1


def split_list(lst, chunk_size):
    return list(zip(*[iter(lst)] * chunk_size))

chunk_size = len(count_order)+1
x1_boundary = split_list(x1_list, chunk_size)
x2_boundary = split_list(x2_list, chunk_size)
y_boundary = split_list(y_list, chunk_size)


for num in range(1,len(y_boundary)):
    x1_boundary1 = x1_boundary[num-1]
    x1_boundary2 = x1_boundary[num]
    
    x2_boundary1 = x2_boundary[num-1]
    x2_boundary2 = x2_boundary[num]
    
    y_boundary1 = y_boundary[num-1]
    y_boundary2 = y_boundary[num]
    
    for i in range(1,len(x1_boundary1)):
        loc1 = i-1
        loc2 = i
        
        xa = x1_boundary2[loc1]
        xb = x2_boundary1[loc1]
        ya = y_boundary1[loc1]
        yb = y_boundary1[loc2]
        yc = y_boundary2[loc1]
        yd = y_boundary2[loc2]
        ax.fill_between([xa,xb],[yc,ya],[yd,yb],color=color_dict[count_order[i-1]],ec='black',lw=0,alpha=0.65,zorder=0)
        #diff_Value = '{:.2}%'.format(((yc-yd)-(ya-yb))/(yc-yd)*100)
        #ax.text((xa+xb)/3,(ya+yb+yc+yd)/4,s=diff_Value,size=20,)


ax.set_xlim(-5,105)
     
plt.axis('off')
plt.savefig('{}/{}_{}_variances{}.png'.format(output_dir,datasetname1,count_variable,branch_point),transparent=True,dpi=600)
plt.show()

    























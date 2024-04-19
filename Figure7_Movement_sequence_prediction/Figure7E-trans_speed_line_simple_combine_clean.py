# -*- coding: utf-8 -*-
"""
Created on Thu May  4 17:22:55 2023

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
import scipy.stats
from scipy.interpolate import make_interp_spline
import matplotlib.patheffects as pe


new_generate_data_day = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure7_Movement_sequence_prediction\predicted_movement_sequences\new_sequence-MOTP_MODP'
new_generate_data_stress = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure7_Movement_sequence_prediction\predicted_movement_sequences\new_sequence-ASTP_ASDP'

InputData_path_dir = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure7_Movement_sequence_prediction\movement_transition_frequency'


skip_file_list = [1,3,28,29,110,122] 

def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-'))&(file_name.endswith(content)):
            USN = int(file_name.split('-')[1])
            file_path_dict.setdefault(USN,file_dir+'\\'+file_name)
    return(file_path_dict)

def get_path2(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-'))&(file_name.endswith(content)):
            USN = file_name.split('-')[1]
            file_path_dict.setdefault(USN,file_dir+'\\'+file_name)
    return(file_path_dict)


Feature_space_path = get_path(InputData_path_dir,'Feature_Space.csv')
#Feature_space_path_predict_day = get_path2(new_generate_data_day,'Feature_Space.csv')
#Feature_space_path_predict_night = get_path2(new_generate_data_night,'Feature_Space.csv')

Movement_label_path = get_path(InputData_path_dir,'Movement_Labels.csv')
#Movement_label_predict_day = get_path2(new_generate_data_day,'Movement_Labels.csv')
#Movement_label_predict_night = get_path2(new_generate_data_night,'Movement_Labels.csv')


animal_info_csv = r'F:\spontaneous_behavior\04返修阶段\Table_S1_animal_information_clean.csv'              
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


def calculate_entroy(df):
    count_df = df['revised_movement_label'].value_counts()
    mv_entroy = pd.DataFrame()
    for mv in movement_order:        
        if mv in count_df.index:
            mv_count = count_df[mv]
        else:
            mv_count = 0
        mv_entroy.loc[mv,'count'] = mv_count
    entroy = scipy.stats.entropy(mv_entroy,base=2)
    return(entroy)

def FeA2MoV(FeA):
    mov_df_dict = {'revised_movement_label':[]}
    for index in FeA.index:
        mov = FeA.loc[index,'revised_movement_label']
        mov_length = FeA.loc[index,'length']
        mov_df_dict['revised_movement_label'].extend([mov]*mov_length)
    output_df = pd.DataFrame(mov_df_dict)
    #print(output_df)
        
def MoV2FeA(df):
    MoV_file = df.copy()
    FeA_file = {'revised_movement_label':[],'segBoundary':[],'segBoundary_start':[],'segBoundary_end':[],'length':[]}
    last_boundary = 0
    for pos in range(1,MoV_file.shape[0]):
        x1 = pos -1
        x2 = pos
        movement_label1 = MoV_file.loc[x1,'revised_movement_label']
        movement_label2 = MoV_file.loc[x2,'revised_movement_label']
        if x2 != MoV_file.shape[0]-1:
            if movement_label1 != movement_label2:
                
                FeA_file['revised_movement_label'].append(movement_label1)
                FeA_file['segBoundary'].append(x1+1)
                FeA_file['segBoundary_start'].append(last_boundary)
                FeA_file['segBoundary_end'].append(x1+1)
                FeA_file['length'].append(x1+1-last_boundary)
                last_boundary = x1+1
        else:
            FeA_file['revised_movement_label'].append(movement_label2)
            FeA_file['segBoundary'].append(x2+1)
            FeA_file['segBoundary_start'].append(last_boundary)
            FeA_file['segBoundary_end'].append(x2+1)
            FeA_file['length'].append(x2+1-last_boundary)
    FeA_df = pd.DataFrame(FeA_file)
    return(FeA_df)


def get_color(name):
    if name.startswith('Morning'):
        color = '#F5B25E'
    elif name.startswith('Afternoon'):
        color = '#936736'
    elif name.startswith('Night_Light-on'):
        color = '#398FCB'
    elif name.startswith('Night_Light-off'):
        color = '#003960'
    elif (name.startswith('Stress')) | (name.startswith('stress')):
        color = '#f55e6f'
    else:
        print(name)
    
    return(color)



df_trans_speed = pd.DataFrame()
num = 0
for index in animal_info.index:
    start = 0
    end = 0
    video_index = animal_info.loc[index,'video_index']
    ExperimentCondition = animal_info.loc[index,'ExperimentCondition']
    LightingCondition = animal_info.loc[index,'LightingCondition']    
    Mov_data = pd.read_csv(Movement_label_path[video_index])
    for i in range(5,61,5):
        end = i *30*60
        select_df = Mov_data.iloc[start:end,:]
        select_df.reset_index(drop=True,inplace=True)
        select_FeA = MoV2FeA(select_df)

        df_trans_speed.loc[num,'video_index'] = index
        df_trans_speed.loc[num,'ExperimentCondition'] = ExperimentCondition+'_'+LightingCondition
        df_trans_speed.loc[num,'time_tag'] = i
        df_trans_speed.loc[num,'trans_speed'] = len(select_FeA) /5 # (len(temp_df) /1) * entroy        

        num += 1
        start = end

    
def average_data(df,count_varible):
    mv_arr = df[count_varible].values
    mv_average = np.mean(mv_arr)
    mv_sem = np.std(mv_arr,ddof=1) / np.sqrt(len(mv_arr))
    return(mv_average,mv_sem)



average_df = pd.DataFrame()
num = 0   
for i in range(5,61,5):
    for k in df_trans_speed['ExperimentCondition'].unique(): #,dataset3_name,dataset4_name
        temp_df = df_trans_speed[(df_trans_speed['time_tag']==i) &(df_trans_speed['ExperimentCondition']==k) ]
        
        average_trans, trans_sem = average_data(temp_df,'trans_speed')


        average_df.loc[num,'time_tag'] = i
        average_df.loc[num,'ExperimentCondition'] = k
        average_df.loc[num,'average_trans_speed'] = average_trans
        average_df.loc[num,'trans_sem'] = trans_sem

        num += 1



fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(15,10),dpi=300)
for k in  average_df['ExperimentCondition'].unique():
    ExperimentCondition_df = average_df[average_df['ExperimentCondition']==k]
    x = ExperimentCondition_df['time_tag']
    y = ExperimentCondition_df['average_trans_speed']
    y1 = ExperimentCondition_df['trans_sem']
    sem_up = y+y1
    sem_down = y-y1
    x_smooth = np.linspace(x.min(), x.max(), 300)  # np.linspace 等差数列,从x.min()到x.max()生成300个数，便于后续插值
    
    y_smooth = make_interp_spline(x, y)(x_smooth)
    #y_smooth_sem = make_interp_spline(x, y1)(x_smooth)
    
    smooth_sem_up = make_interp_spline(x, sem_up)(x_smooth)
    smooth_sem_down = make_interp_spline(x, sem_down)(x_smooth)
    
    color = get_color(k) 
    ax.plot(x_smooth, y_smooth,color = color,lw=3,path_effects=[pe.Stroke(linewidth=5, foreground=color,alpha=1), pe.Normal()],zorder=3)
    ax.fill_between(x=x_smooth,y1=smooth_sem_up,y2=smooth_sem_down,color=color,alpha=0.3,lw=1,zorder=2)
    
ax.set_xlabel('Time(min)',fontsize=15)
ax.set_ylabel('Average movement trans number',fontsize=15)
ax.set_ylim(15,60)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)  
ax.spines['bottom'].set_linewidth(5)
ax.spines['left'].set_linewidth(5)
ax.set_xticks(np.arange(2.5,63,15))
#ax.set_xticklabels([])
ax.set_ylim(10,61)
ax.set_yticks(range(10,61,10))
#ax.set_yticklabels([])
plt.tick_params(which='major',width=4,length=10)

for i in range(10,61,10):
    ax.plot([2.5,63],[i,i],lw=3,linestyle='--',color='#9E9E9E',zorder=-1,alpha=0.6)

plt.savefig('{}/trans_number_all.png'.format(output_dir),dpi=300)


        
        
        
        
        
        
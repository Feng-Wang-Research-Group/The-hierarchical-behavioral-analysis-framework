# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 11:09:35 2023

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

from statsmodels.tsa.stattools import adfuller
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from arch import arch_model

import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import bootstrap
from scipy.stats import norm
import matplotlib.ticker as ticker


##### calculate_movement_duration_probability

InputData_path_dir = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure7_Movement_sequence_prediction\movement_temporal_duration_distribution'

def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if file_name.startswith('rec-') & file_name.endswith(content):
            USN = int(file_name.split('-')[1])
            #date = i.split('-')[3][0:8]
            #file_name = 'rec-{0}-G1-{1}'.format(USN,date)
            file_path_dict.setdefault(USN,file_dir+'/'+file_name)
    return(file_path_dict)

Mov_path = get_path(InputData_path_dir,'Movement_Labels.csv')
FeA_path = get_path(InputData_path_dir,'Feature_Space.csv')



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


# select the dataset you want to plot

dataset = stress_animal_info
dataset_name = 'stress_animal'


def add_category(df):
    df_copy = df[['new_label']].copy()
    big_category_dict4 = {'locomotion':['running','walking','left_turning','right_turning','stepping'],
                         'exploration':['climbing','rearing','hunching','rising','sniffing','jumping'],
                         'maintenance':['grooming','scratching'],
                         'inactive':['pausing']
                         }

    df_copy.loc[df_copy['new_label'].isin(big_category_dict4['locomotion']),'category4'] = 'locomotion'
    df_copy.loc[df_copy['new_label'].isin(big_category_dict4['exploration']),'category4'] = 'exploration'
    df_copy.loc[df_copy['new_label'].isin(big_category_dict4['maintenance']),'category4'] = 'maintenance' 
    df_copy.loc[df_copy['new_label'].isin(big_category_dict4['inactive']),'category4'] = 'inactive'
    return(df_copy)

def countFragementLength(df,time_segement):
    start = 0
    end = 0
    
    df_output = pd.DataFrame()
    num = 0
    for i in range(time_segement,61,time_segement):
        end = i*60*30
        temp_FeA = df[(df['segBoundary_start']>start) & (df['segBoundary_end']<=end)]
        for mv in movement_order:
            temp_FeA_mv = temp_FeA[temp_FeA['movement_label']==mv]
            if len(temp_FeA_mv) > 0:
                average_duration = np.mean(temp_FeA_mv['frame_length'])
            else:
                average_duration = 0
            df_output.loc[num,'time'] = i
            df_output.loc[num,'movement_label'] = mv
            df_output.loc[num,'average_duration'] = average_duration
            num += 1
        start = end
    return(df_output)


def countFragementLengthEach(df,time_segement):
    start = 0
    end = 0
    
    df_output = pd.DataFrame()
    num = 0
    for i in range(time_segement,61,time_segement):
        end = i*60*30
        temp_FeA = df[(df['segBoundary_start']>start) & (df['segBoundary_end']<=end)]
        for mv in movement_order:
            temp_FeA_mv = temp_FeA[temp_FeA['revised_movement_label']==mv]
            if len(temp_FeA_mv) > 0:
                for length in temp_FeA_mv['frame_length']:
                    df_output.loc[num,'time'] = i
                    df_output.loc[num,'movement_label'] = mv
                    df_output.loc[num,'duration'] = length
                    num += 1
        start = end
    return(df_output)         
        

MV_duration_list = []

for i in dataset.index:
    video_index = dataset.loc[i,'video_index']
    ExperimentCondition = dataset.loc[i,'ExperimentCondition']
    gender = dataset.loc[i,'gender']
        
    data = pd.read_csv(Mov_path[video_index])
    FeA_data = pd.read_csv(FeA_path[video_index])
    
    MV_duration = countFragementLengthEach(FeA_data,1)
    MV_duration_list.append(MV_duration)
     
movement_duration_data = pd.concat(MV_duration_list)   


rng = np.random.default_rng()
step = 5

movement_duration_data['time_stage'] = (((movement_duration_data['time']-1) // step)+1) * step

df_mv_duration_range = pd.DataFrame()
num = 0
for i in range(step,61,step):    
    time_df = movement_duration_data[movement_duration_data['time_stage']==i]
    for mv in movement_order:
        time_mv_df = time_df[time_df['movement_label']==mv]
        if len(time_mv_df) > 10:
            arr = np.array(time_mv_df['duration'])
            data = (arr,)
            res = bootstrap(data, np.mean, axis=-1, confidence_level=0.95, n_resamples=10000, random_state=rng)
            ci_l, ci_u = res.confidence_interval
            average_value = np.mean(res.bootstrap_distribution)
            median = np.median(res.bootstrap_distribution)
            df_mv_duration_range.loc[num,'time'] = i
            df_mv_duration_range.loc[num,'movement_label'] = mv
            df_mv_duration_range.loc[num,'ci_l'] = ci_l
            df_mv_duration_range.loc[num,'ci_u'] = ci_u
            df_mv_duration_range.loc[num,'average_value'] = average_value
            df_mv_duration_range.loc[num,'median'] = median
            df_mv_duration_range.loc[num,'std_sample'] =  np.std(data)
            df_mv_duration_range.loc[num,'scale'] = res.standard_error
            num += 1

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


for mv in movement_order:
    movement_df = df_mv_duration_range[df_mv_duration_range['movement_label'] == mv]
    color = movement_color_dict[mv]

    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,10),dpi=300)

    for i in range(step,61,step):
        if len(movement_df.loc[movement_df['time']==i]) >0:
            ci_l = movement_df.loc[movement_df['time']==i,'ci_l'].values[0]
            ci_u = movement_df.loc[movement_df['time']==i,'ci_u'].values[0]
            average_value =  movement_df.loc[movement_df['time']==i,'average_value'].values[0]
            median =  movement_df.loc[movement_df['time']==i,'median'].values[0]
            std_sample = movement_df.loc[movement_df['time']==i,'std_sample'].values[0]
            scale = movement_df.loc[movement_df['time']==i,'scale'].values[0]
            x = np.linspace(ci_l, ci_u)
            #std_sample = np.std(data)
            pdf = norm.pdf(x, loc=average_value, scale=scale)
            ax.plot([ci_l,ci_u],[i,i],c='#CFD8DC',lw=5,linestyle='-',zorder=1,alpha=0.9)
            ax.plot(x, (pdf*5)+i,lw=2,color=color)
            ax.fill(x, (pdf*5)+i,color=color,alpha=0.5)
            ax.scatter(ci_l,i,c='#90A4AE',s=25,zorder=2)
            ax.scatter(ci_u,i,c='#90A4AE',s=25,zorder=2)
            ax.scatter(average_value,i,c='#D50000',s=100,zorder=2,marker='|')
            ax.scatter(median,i,c='#1976D2',s=100,zorder=2,marker='|')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.set_title(mv,fontsize=20)

    plt.savefig(r'{}\{}_{}_temporal_duration_distribution.png'.format(output_dir,dataset_name,mv),dpi=300,transparent=True)  


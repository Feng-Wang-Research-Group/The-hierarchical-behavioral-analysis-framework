# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 20:42:26 2023

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
from tkinter import _flatten
import networkx as nx
import scipy.stats as stats


InputData_path_dir = r"F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Figure6_Movement_transition\Cluster_transition'


skip_file_list = [1,3,28,29,110,122] 

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
#Speed_distance_path =  



animal_info_csv = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Table_S1_animal_information.csv'
animal_info = pd.read_csv(animal_info_csv)
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)]

Morning_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Morning') & (animal_info['LightingCondition']=='Light-on')]
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-off')]



dataset1 = Night_lightOn_info
dataset_name1 = 'Night_lightOn'

dataset2 = Night_lightOff_info
dataset_name2 = 'Night_lightOff'

start = 0
end = 60



# =============================================================================
# movement_order = ['running','trotting','walking','right_turning','left_turning','stepping',
#                   'jumping','climbing','rearing','hunching','rising','sniffing',
#                   'grooming','pausing']
# =============================================================================

movement_order = ['locomotion','exploration','maintenance','nap']

# =============================================================================
# movement_color_dict = {'running':'#DD2C00',
#                        'trotting':'#EC407A',
#                        'walking':'#FF5722',
#                        'left_turning':'#FFAB91',
#                        'right_turning':'#FFCDD2',
#                        'stepping':'#BCAAA4',
#                        'sniffing':'#26A69A',
#                        'climbing':'#2E7D32',
#                        'rearing':'#66BB6A',
#                        'hunching':'#0288D1',
#                        'rising':'#9CCC65',
#                        'jumping':'#FFB74D',
#                        'grooming':'#AB47BC',
#                        'pausing':'#90A4AE',} 
# =============================================================================


movement_color_dict ={'locomotion':'#DC2543',                     
                     'exploration':'#028C6A',
                     'maintenance':'#A13E97',
                     'nap':'#D4D4D4'}



def find_indices(list_to_check,item_to_find):
    return([idx for idx, value in enumerate(list_to_check) if value == item_to_find])

def count_percentage(label_assemble):
    df = pd.DataFrame()
    num = 0
    for key in movement_order:
        count = label_assemble.count(key)
        df.loc[num,'later_label'] = key
        df.loc[num,'count'] = count
        
        num += 1
    df['percentage'] = df['count']/df['count'].sum()
    
    return(df)

def count_percentage2(label_assemble,key):
    while '' in label_assemble:
        label_assemble.remove('')
    df = pd.DataFrame()
    num = 0
    for key in movement_order:
        count = label_assemble.count(key)
        df.loc[num,'later_label'] = key
        df.loc[num,'count'] = count
        
        if count / len(label_assemble) <0.05:
            df.loc[num,'count'] = 0
        
        num += 1
    df['percentage'] = df['count']/df['count'].sum()
    
    return(df)

def add_Cluster_label(df):
    big_cluster_dict4 = {'locomotion':['running','trotting','walking','left_turning','right_turning','stepping'],
                         'exploration':['climbing','rearing','hunching','rising','sniffing','jumping'],
                         'maintenance':['grooming'],
                         'nap':['pausing']
                         }   
    df_copy = df.copy()
    for mv in big_cluster_dict4.keys():
        df_copy.loc[df_copy['revised_movement_label'].isin(big_cluster_dict4[mv]),'movement_cluster_label'] = mv
    return(df_copy)

def add_color(df):
    for i in df.index:
        df.loc[i,'color'] = movement_color_dict[df.loc[i,'later_label']]
    return(df)



def count_trans_probablities(sentences):
    df_list = []
    for key in movement_order:
        label_behind_key_sum = []
        
        for sentence in sentences:
            if key in sentence:
                index_ID = np.array(find_indices(sentence,key))
                if len(sentence)-1 in index_ID:
                    index_ID = np.delete(index_ID, find_indices(index_ID,len(sentence)-1)[0])
        
                    #del index_ID[find_indices(index_ID,len(sentence)-1)[0]]
                label_behind_key = np.array(sentence)[index_ID+1]
            else:
                label_behind_key = ['']
            
            #index_ID = np.array(find_indices(sentence,key))
            #if 0 in index_ID:
            #    index_ID = np.delete(index_ID, find_indices(index_ID,0)[0])
            #label_before_key = np.array(sentence)[index_ID-1]
            
            label_behind_key_sum.extend(label_behind_key)
            while '' in label_behind_key_sum:
                label_behind_key_sum.remove('')
            if len(label_behind_key_sum) > 10:
                df_count = count_percentage2(label_behind_key_sum,key)
                df_count['previous_label'] = key
                df_count = df_count[['previous_label','later_label','count','percentage']]
                df_list.append(df_count)
        
    all_df = pd.concat(df_list,axis=0)
    all_df.reset_index(drop=True,inplace=True)

    return(all_df)


def get_trans_df(dataset):
    df_singleMice_trans_list = []
    for key in dataset['video_index']:
        sentences = []
        df = pd.read_csv(Feature_space_path[key])
        df = add_Cluster_label(df)
        temp_df = df[(df['segBoundary_start']>=start*30*60) & (df['segBoundary_end']<end*30*60)]
        sentences.append(list(temp_df['movement_cluster_label'].values))
        df_singleMice_trans = count_trans_probablities(sentences)
        df_singleMice_trans_list.append(df_singleMice_trans)
    #mv_percentage = count_percentage2(list(_flatten(sentences)))    ## 占比
    #mv_percentage = add_color(mv_percentage)
    all_df = pd.concat(df_singleMice_trans_list,axis=0)
    all_df.reset_index(drop=True,inplace=True)
    
    return(all_df)
    


MVcluster_trans_df_dataset1 =  get_trans_df(dataset1)
MVcluster_trans_df_dataset1['group'] = dataset_name1
MVcluster_trans_df_dataset2 =  get_trans_df(dataset2)
MVcluster_trans_df_dataset2['group'] = dataset_name2

all_df = pd.concat([MVcluster_trans_df_dataset1,MVcluster_trans_df_dataset2])

info_dict = {'previous_label':[],'later_label':[],'count':[],'dataset1_mean':[],'dataset2_mean':[],'diff':[],'h':[],'p':[],'significant':[]}

for previous_label in movement_order:
    for later_label in movement_order:
        arr1 = np.array(all_df[(all_df['previous_label']==previous_label)&(all_df['later_label']==later_label)&(all_df['group']==dataset_name1)]['percentage'])
        arr2 = np.array(all_df[(all_df['previous_label']==previous_label)&(all_df['later_label']==later_label)&(all_df['group']==dataset_name2)]['percentage']) ##day-pm #night_lightOn #night_lightOff
        
        count = np.sum(all_df[(all_df['previous_label']==previous_label)&(all_df['later_label']==later_label)&(all_df['group']=='dataset_name1')]['count'])
    
        arr1 = np.nan_to_num(arr1)
        arr2 = np.nan_to_num(arr2)
        
        _,arr1_norm = stats.shapiro(arr1)
        _,arr2_norm = stats.shapiro(arr2)
        
        arr1_mean = np.mean(arr1)
        arr2_mean = np.mean(arr2)
        
        diff = arr2_mean - arr1_mean
        
        try:
            #h, p = kruskalwallis(arr1,arr2)
            if (arr1_norm > 0.05) & (arr2_norm > 0.05):
                h, p = stats.ttest_ind(arr1, arr2)
            else:
                h, p = stats.mannwhitneyu(arr1, arr2, use_continuity=True)
                p *= 2
        except:
            h = p = 1

        info_dict['previous_label'].append(previous_label)
        info_dict['later_label'].append(later_label)
        info_dict['count'].append(count)
        info_dict['dataset1_mean'].append(arr1_mean)
        info_dict['dataset2_mean'].append(arr2_mean)
        
        info_dict['diff'].append(diff)
        
        info_dict['h'].append(h)
        info_dict['p'].append(p)
        
        if p < 0.05:
            info_dict['significant'].append('significant')
        else:
            info_dict['significant'].append('Not significant')

df = pd.DataFrame(info_dict)


df_heatmap1 = pd.DataFrame()
for previous_label in df['previous_label']:
    for later_label in df['later_label']:
        df_heatmap1.loc[previous_label,later_label] = df[(df['previous_label']==previous_label)&(df['later_label']==later_label)]['dataset1_mean'].values[0]

df_heatmap2 = pd.DataFrame()
for previous_label in df['previous_label']:
    for later_label in df['later_label']:
        df_heatmap2.loc[previous_label,later_label] = df[(df['previous_label']==previous_label)&(df['later_label']==later_label)]['dataset2_mean'].values[0]

df_heatmap_diff = pd.DataFrame()
for previous_label in df['previous_label']:
    for later_label in df['later_label']:
        df_heatmap_diff.loc[previous_label,later_label] = df[(df['previous_label']==previous_label)&(df['later_label']==later_label)]['diff'].values[0]


fig,ax = plt.subplots(nrows=1,ncols=1,figsize = (10,10),dpi=300)
sns.heatmap(df_heatmap1,cmap='RdYlBu_r',ax=ax,square=True,annot=True,fmt='.2f',annot_kws={'fontsize':20,'weight':'bold',},linewidths=1,linecolor='black', cbar=True,vmin=0,vmax=1,cbar_kws={'shrink': 0.8,'ticks': np.arange(0,1.1,0.1)})
plt.savefig('{}/{}_MovClusterTrans_network_v2.png'.format(output_dir,dataset_name1),dpi=600,transparent=True)

fig,ax = plt.subplots(nrows=1,ncols=1,figsize = (10,10),dpi=300)
sns.heatmap(df_heatmap2,cmap='RdYlBu_r',ax=ax,square=True,annot=True,fmt='.2f',annot_kws={'fontsize':20,'weight':'bold',},linewidths=1,linecolor='black', cbar=True,vmin=0,vmax=1,cbar_kws={'shrink': 0.8,'ticks': np.arange(0,1.1,0.1)})
plt.savefig('{}/{}_MovClusterTrans_network_v2.png'.format(output_dir,dataset_name2),dpi=600,transparent=True)

fig,ax = plt.subplots(nrows=1,ncols=1,figsize = (10,10),dpi=300)
sns.heatmap(df_heatmap_diff,cmap='RdBu_r',ax=ax,square=True,annot=True,fmt='.2f',annot_kws={'fontsize':25,'weight':'bold',},linewidths=1,linecolor='black', cbar=True,vmin=-0.2,vmax=0.2,cbar_kws={'shrink': 0.8,'ticks': np.arange(0,1.1,0.1)})#,vmin=-0.03,vmax=0.03,cbar=True)

#plt.axis('off')
plt.savefig('{}/{}&{}_MovClusterTrans_diffheatmap.png'.format(output_dir,dataset_name1,dataset_name2),dpi=600,transparent=True)







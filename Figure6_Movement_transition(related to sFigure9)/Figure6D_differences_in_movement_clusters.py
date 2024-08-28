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


InputData_path_dir = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure6_Movement_transition(related to sFigure9)\Clusters'


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


# =============================================================================
# movement_order = ['running','trotting','walking','right_turning','left_turning','stepping',
#                   'rising','hunching','rearing','climbing','jumping','sniffing','grooming','scratching','pausing',]
# =============================================================================

movement_order = ['locomotion','exploration','maintenance','nap']
movement_color_dict ={'locomotion':'#DC2543',                     
                     'exploration':'#009688',
                     'maintenance':'#A13E97',
                     'nap':'#B0BEC5'}

movement_color_dict_list = ['#DC2543',
                            '#009688',
                            '#A13E97',
                            '#B0BEC5',
                            ]

dataset1 = Morning_lightOn_info
dataset_name1 = 'Morning_lightOn'

dataset2 = stress_animal_info
dataset_name2 = 'stress_animal'

start1 = 0
end1 = 60

start2 = 0
end2 = 60

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
    if len(label_assemble) > 0:
        for key in movement_order:
            count = label_assemble.count(key)
            df.loc[num,'later_label'] = key
            df.loc[num,'count'] = count
            
            if count / len(label_assemble) <0.05:
                df.loc[num,'count'] = 0
            
            num += 1
        df['percentage'] = df['count']/df['count'].sum()
    else:
        for key in movement_order:
            df.loc[num,'later_label'] = key
            df.loc[num,'count'] = 0
            df['percentage'] = 0
    return(df)

def add_Cluster_label(df):
    big_cluster_dict4 = {'locomotion':['running','trotting','walking','left_turning','right_turning','stepping'],
                         'exploration':['climbing','rearing','hunching','rising','sniffing','jumping'],
                         'maintenance':['grooming','scratching',],
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
            #while '' in label_behind_key_sum:
            #    label_behind_key_sum.remove('')
            #if len(label_behind_key_sum) > 10:
            df_count = count_percentage2(label_behind_key_sum,key)
            df_count['previous_label'] = key
            df_count = df_count[['previous_label','later_label','count','percentage']]
            df_list.append(df_count)
    
    all_df = pd.concat(df_list,axis=0)
    all_df.reset_index(drop=True,inplace=True)

    return(all_df)


def get_trans_df(dataset,start=0,end=60):
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
    


MVcluster_trans_df_dataset1 =  get_trans_df(dataset1,start1,end1)
MVcluster_trans_df_dataset1['group'] = dataset_name1
MVcluster_trans_df_dataset2 =  get_trans_df(dataset2,start2,end2)
MVcluster_trans_df_dataset2['group'] = dataset_name2

all_df = pd.concat([MVcluster_trans_df_dataset1,MVcluster_trans_df_dataset2])

info_dict = {'previous_label':[],'later_label':[],'dataset1_mean':[],'dataset2_mean':[],'statistic_diff':[],'CI(permuted_differences)':[],'p_value':[],'significant':[]}

num = 0
for previous_label in movement_order:
    for later_label in movement_order:
        num += 1

        arr1 = np.array(all_df[(all_df['previous_label']==previous_label)&(all_df['later_label']==later_label)&(all_df['group']==dataset_name1)]['percentage'])
        arr2 = np.array(all_df[(all_df['previous_label']==previous_label)&(all_df['later_label']==later_label)&(all_df['group']==dataset_name2)]['percentage'])
        
        arr1 = np.nan_to_num(arr1)
        arr2 = np.nan_to_num(arr2)
                
        arr1_mean = np.mean(arr1)
        arr2_mean = np.mean(arr2)
        
        diff = arr2_mean - arr1_mean
        
        observed_diff = np.mean(arr2) - np.mean(arr1)        
        # Number of bootstrap samples
        num_bootstraps = 10000
        combined_data = np.concatenate((arr1, arr2))
        bootstrap_means_group1 = np.zeros(num_bootstraps)
        bootstrap_means_group2 = np.zeros(num_bootstraps)
        
        permuted_statistics = np.zeros(num_bootstraps)
        # Perform bootstrap resampling
        for i in range(num_bootstraps):
            # Randomly select indices with replacement
            a = len(arr1)
            b = len(arr2)
            if a >= b :
                sample_num = b
            else:
                sample_num = a
            sample_indices = np.random.choice(sample_num, size=sample_num, replace=True)
            
            # Create bootstrap samples from each distribution
            bootstrap_sample1 = arr1[sample_indices]
            bootstrap_sample2 = arr2[sample_indices]
            
            # Calculate the difference in means for the bootstrap samples
            bootstrap_means_group1[i] = np.mean(bootstrap_sample1)
            bootstrap_means_group2[i] = np.mean(bootstrap_sample2)
            
            np.random.shuffle(combined_data)
            permuted_group1 = combined_data[:len(arr1)]
            permuted_group2 = combined_data[len(arr1):]
            permuted_statistic = np.mean(permuted_group2) - np.mean(permuted_group1)
            permuted_statistics[i] = permuted_statistic
        
        
        mean_difference = np.mean(bootstrap_means_group2) - np.mean(bootstrap_means_group1)
        confidence_interval = np.percentile(bootstrap_means_group2 - bootstrap_means_group1, [2.5, 97.5])              #0.95 two_tail
        p_value = (np.abs(permuted_statistics) >= np.abs(observed_diff)).mean()
        info_dict['previous_label'].append(previous_label)
        info_dict['later_label'].append(later_label)
        info_dict['dataset1_mean'].append(np.mean(bootstrap_means_group1))
        info_dict['dataset2_mean'].append(np.mean(bootstrap_means_group2))        
        info_dict['statistic_diff'].append(mean_difference)            
        info_dict['CI(permuted_differences)'].append(confidence_interval)  
        info_dict['p_value'].append(p_value)
        if p_value < 0.05:
            info_dict['significant'].append('significant')
        else:
            info_dict['significant'].append('Not significant')

df = pd.DataFrame(info_dict)
df.to_csv('{}/{}&{}_MovClusterTrans_diff.csv'.format(output_dir,dataset_name1,dataset_name2))

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
        if df.loc[(df['previous_label']==previous_label)&(df['later_label']==later_label),'significant'].values[0] == 'significant':
            df_heatmap_diff.loc[previous_label,later_label] = df.loc[(df['previous_label']==previous_label)&(df['later_label']==later_label),'statistic_diff'].values[0]
        else:
            df_heatmap_diff.loc[previous_label,later_label] = 0



fig,ax = plt.subplots(nrows=1,ncols=1,figsize = (10,10),dpi=300)
sns.heatmap(df_heatmap1,cmap='RdYlBu_r',ax=ax,square=True,annot=True,fmt='.2f',annot_kws={'fontsize':20,'weight':'bold',},linewidths=1,linecolor='black', cbar=False,vmin=0,vmax=1,cbar_kws={'shrink': 0.8,'ticks': np.arange(0,1.1,0.1)},lw=4)
plt.axis('off')
plt.savefig('{}/{}_MovClusterTrans_network_v2.png'.format(output_dir,dataset_name1),dpi=600,transparent=True)

fig,ax = plt.subplots(nrows=1,ncols=1,figsize = (10,10),dpi=300)
sns.heatmap(df_heatmap2,cmap='RdYlBu_r',ax=ax,square=True,annot=True,fmt='.2f',annot_kws={'fontsize':20,'weight':'bold',},linewidths=1,linecolor='black', cbar=False,vmin=0,vmax=1,cbar_kws={'shrink': 0.8,'ticks': np.arange(0,1.1,0.1)},lw=4)
plt.axis('off')
plt.savefig('{}/{}_MovClusterTrans_network_v2.png'.format(output_dir,dataset_name2),dpi=600,transparent=True)

fig,ax = plt.subplots(nrows=1,ncols=1,figsize = (10,10),dpi=300)
sns.heatmap(df_heatmap_diff,cmap='RdBu_r',ax=ax,square=True,annot=True,fmt='.2f',annot_kws={'fontsize':25,'weight':'bold'},linewidths=1,linecolor='black', cbar=False,vmin=-0.15,vmax=0.15,cbar_kws={'shrink': 0.8,'ticks': np.arange(0,1.1,0.1)},lw=4)#,vmin=-0.03,vmax=0.03,cbar=True)
plt.axis('off')
#plt.axis('off')
plt.savefig('{}/{}&{}_MovClusterTrans_diffheatmap.png'.format(output_dir,dataset_name1,dataset_name2),dpi=600,transparent=True)



fig,ax = plt.subplots(nrows=1,ncols=1,figsize = (10,10),dpi=300)
Drawing_uncolored_circle = plt.Circle(((0,0)), radius=1,color='#ECEFF1',zorder=0)
ax.set_aspect( 1 )
ax.add_artist( Drawing_uncolored_circle)
#G = nx.Graph()
G = nx.DiGraph()
for i in df_heatmap_diff.index:
    for j in df_heatmap_diff.columns:
        G.add_edge(i, j, weight=df_heatmap_diff.loc[i,j])

    
#pos = pos_13movement2
pos = nx.circular_layout(G)
#pos = nx.spring_layout(G)

nx.draw_networkx_nodes(G,pos,alpha=1,node_size = 1000,node_color = movement_color_dict_list,edgecolors='black',linewidths=2)  # movement_color_dict_list
for (u,v,d) in G.edges(data=True):
    #color = movement_color_dict[u]
    weight = d['weight']
    level_1 = [(u,v) for (u,v,d) in G.edges(data=True) if d['weight']>=0.2]
    level_2 = [(u,v) for (u,v,d) in G.edges(data=True) if (d['weight']<0.2) & (d['weight']>=0.1)]
    #level_3 = [(u,v) for (u,v,d) in G.edges(data=True) if (d['weight']<0.1) & (d['weight']>=0.01)]
    #level_3 =  [(u,v) for (u,v,d) in G.edges(data=True) if (d['weight']>=-0.1) & (d['weight']<0.1)]
    level_1_2 =  [(u,v) for (u,v,d) in G.edges(data=True) if d['weight']<= -0.2]
    level_2_2 = [(u,v) for (u,v,d) in G.edges(data=True) if (d['weight']>-0.2) & (d['weight']<=-0.1)]
    #level_3_2 = [(u,v) for (u,v,d) in G.edges(data=True) if (d['weight']>-0.1) & (d['weight']<=-0.01)]
    
    
    nx.draw_networkx_edges(G,pos,edgelist=level_1,connectionstyle='Arc3, rad = 0.15',width=5,alpha=0.7,edge_color='#E91E63',arrows=True,arrowsize=20,min_target_margin=22)
    nx.draw_networkx_edges(G,pos,edgelist=level_2,connectionstyle='Arc3, rad = 0.15',width=3,alpha=0.7,edge_color='#E91E63',arrows=True,arrowsize=15,min_target_margin=22)
    #nx.draw_networkx_edges(G,pos,edgelist=level_3,connectionstyle='Arc3, rad = 0.15',width=1,alpha=0.5,edge_color='red',arrows=True,arrowsize=15,min_target_margin=22,style='--')
    
    nx.draw_networkx_edges(G,pos,edgelist=level_1_2,connectionstyle='Arc3, rad = 0.15',width=6,alpha=0.7,edge_color='#3C4EA0',arrows=True,arrowsize=20,min_target_margin=22)
    nx.draw_networkx_edges(G,pos,edgelist=level_2_2,connectionstyle='Arc3, rad = 0.15',width=3,alpha=0.7,edge_color='#3C4EA0',arrows=True,arrowsize=15,min_target_margin=22)
    #nx.draw_networkx_edges(G,pos,edgelist=level_3_2,connectionstyle='Arc3, rad = 0.15',width=1,alpha=0.5,edge_color='blue',arrows=True,arrowsize=15,min_target_margin=22,style='--')


plt.axis('off')
plt.savefig('{}/{}&{}_MovTrans_diff_{}-{}VS{}-{}min.png'.format(output_dir,dataset_name1,dataset_name2,start1,end1,start2,end2),dpi=1200,transparent=True)




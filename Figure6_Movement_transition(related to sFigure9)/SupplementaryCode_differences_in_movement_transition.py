# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 21:00:46 2023

@author: Jialin Ye
@institution: SIAT
@Contact_email: jl.ye@siat.ac.cn

"""

import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import _flatten
import networkx as nx
import scipy.stats as stats
from scipy.stats import bootstrap

InputData_path_dir = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure6_Movement_transition(related to sFigure9)\Transition_differences'

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

animal_info_csv = r'F:\spontaneous_behavior\04返修阶段\Table_S1_animal_information_clean.csv'
animal_info = pd.read_csv(animal_info_csv)
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)]


Morning_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Morning') & (animal_info['LightingCondition']=='Light-on')]
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-off')]
Stress_info = animal_info[(animal_info['ExperimentCondition']=='Stress')]

Morning_lightOn_info_male = animal_info[(animal_info['ExperimentCondition']=='Morning') & (animal_info['LightingCondition']=='Light-on') & (animal_info['gender']=='male')]
Morning_lightOn_info_female = animal_info[(animal_info['ExperimentCondition']=='Morning') & (animal_info['LightingCondition']=='Light-on')& (animal_info['gender']=='female')]
Afternoon_lightOn_info_male = animal_info[(animal_info['ExperimentCondition']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')& (animal_info['gender']=='male')]
Afternoon_lightOn_info_female = animal_info[(animal_info['ExperimentCondition']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')& (animal_info['gender']=='female')]
Night_lightOn_info_male = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-on')& (animal_info['gender']=='male')]
Night_lightOn_info_female = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-on')& (animal_info['gender']=='female')]
Night_lightOff_info_male = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-off')& (animal_info['gender']=='male')]
Night_lightOff_info_female = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-off')& (animal_info['gender']=='female')]
Stress_info_male = Stress_info[Stress_info['gender']=='male']
Stress_info_female = Stress_info[Stress_info['gender']=='female']


dataset1 = Morning_lightOn_info
dataset_name1 = 'Morning_lightOn'

dataset2 = Stress_info
dataset_name2 = 'Stress'

start1 = 0
end1 = 60

start2 = 0
end2 = 60

movement_order = ['running','trotting','left_turning','right_turning','walking','stepping','sniffing',
                  'rising','hunching','rearing','climbing','jumping','grooming','scratching','pausing',]

#movement_order = ['locomotion','exploration','maintenance','nap']

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

movement_color_dict_list = ['#FF3030',
                            '#E15E8A',
                            '#F6BBC6',
                            '#F8C8BA',
                            '#EB6148',                       
                            '#C6823F',  
                            '#2E8BBE',
                            '#84CDD9',
                            '#D4DF75',
                            '#88AF26',                           
                            '#2E7939',
                            '#24B395',                       
                            '#973C8D',
                            '#EADA33',
                            '#B0BEC5']


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

def count_percentage2(label_assemble):
    while '' in label_assemble:
        label_assemble.remove('')
    df = pd.DataFrame()
    num = 0
    #print(label_assemble)
    for key in movement_order:
        if len(label_assemble) == 0:
            df.loc[num,'later_label'] = key
            df.loc[num,'count'] = 0
            num += 1
        else:
            if key in label_assemble:
                count = label_assemble.count(key)
                df.loc[num,'later_label'] = key
                df.loc[num,'count'] = count
    
                if count / len(label_assemble) <0.05:
                    df.loc[num,'count'] = 0
                num += 1
            else:
                df.loc[num,'later_label'] = key
                df.loc[num,'count'] = 0
                num += 1
    if (df['count'].sum()) == 0:
        df['percentage'] = 0
    else:
        df['percentage'] = df['count']/df['count'].sum()
    return(df)


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
                label_behind_key = []
            
            #index_ID = np.array(find_indices(sentence,key))
            #if 0 in index_ID:
            #    index_ID = np.delete(index_ID, find_indices(index_ID,0)[0])
            #label_before_key = np.array(sentence)[index_ID-1]
            
            label_behind_key_sum.extend(label_behind_key)
        df_count = count_percentage2(label_behind_key_sum)
        df_count['previous_label'] = key
        df_count = df_count[['previous_label','later_label','count','percentage']]
        df_list.append(df_count)
        
    all_df = pd.concat(df_list,axis=0)
    all_df.reset_index(drop=True,inplace=True)

    return(all_df)


def get_trans_df1(dataset):
    df_singleMice_trans_list = []
    for key in dataset['video_index']:
        sentences = []
        random_seq = []
        df = pd.read_csv(Feature_space_path[key])
        temp_df = df[(df['segBoundary_start']>=start1*30*60) & (df['segBoundary_end']<end1*30*60)]
        
        sentences.append(list(temp_df['revised_movement_label'].values))
        #print(sentences)
        
        df_singleMice_trans = count_trans_probablities(sentences)
        df_singleMice_trans_list.append(df_singleMice_trans)

    mv_percentage = count_percentage2(list(_flatten(sentences)))    ## 占比
    mv_percentage = add_color(mv_percentage)
    all_df = pd.concat(df_singleMice_trans_list,axis=0)
    all_df.reset_index(drop=True,inplace=True)
    
    return(all_df,mv_percentage)

def get_trans_df2(dataset):
    df_singleMice_trans_list = []
    for key in dataset['video_index']:
        sentences = []
        df = pd.read_csv(Feature_space_path[key])
        temp_df = df[(df['segBoundary_start']>=start2*30*60) & (df['segBoundary_end']<end2*30*60)]
        sentences.append(list(temp_df['revised_movement_label'].values))
        #print(sentences)
        
        df_singleMice_trans = count_trans_probablities(sentences)
        df_singleMice_trans_list.append(df_singleMice_trans)

    mv_percentage = count_percentage2(list(_flatten(sentences)))    
    mv_percentage = add_color(mv_percentage)
    all_df = pd.concat(df_singleMice_trans_list,axis=0)
    all_df.reset_index(drop=True,inplace=True)
    
    return(all_df,mv_percentage)
        
    
def plot_MVcluster(all_df,mv_percentage):

    trans_df = pd.DataFrame()
    for i in all_df.index:
        previous_label = all_df.loc[i,'previous_label']
        later_label = all_df.loc[i,'later_label']
        trans_df.loc[previous_label,later_label] = all_df.loc[i,'percentage']
    
    
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize = (10,10),dpi=600)
    sns.heatmap(trans_df,cmap='flare',square=True,cbar=False,linecolor='white',linewidths=1,)
    plt.axis('off')
    #plt.savefig('{}/{}_Movtrans_heatmap.png'.format(output_dir,dataset_name),dpi=600)
    plt.show()
    
    
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize = (10,10),dpi=600)
    #G = nx.Graph()
    G = nx.MultiDiGraph()
    for i in all_df['previous_label'].unique():
        for j in all_df['later_label'].unique():
            weight=all_df[(all_df['previous_label']==i) & (all_df['later_label']==j)]['percentage'].values[0]
            G.add_edge(i, j, weight=all_df[(all_df['previous_label']==i) & (all_df['later_label']==j)]['percentage'].values[0])
        
    #pos = pos_13movement
    pos = nx.circular_layout(G)
    pos2 = {'locomotion': np.array([1.2, -0.2]),
     'exploration': np.array([0,  1]),
     'maintenance': np.array([-1, 0]),
     'nap': np.array([0, -1])}
    nx.draw_networkx_nodes(G,pos,alpha=1,node_size=mv_percentage.percentage * 10000,node_color=mv_percentage.color,edgecolors='black',linewidths=3)

    for (u,v,d) in G.edges(data=True):
        color = movement_color_dict[u]
        weight = d['weight']
        if u == v:
            nx.draw_networkx_edges(G,pos2,edgelist=[(u,v)],connectionstyle='Arc3, rad = 100',width=weight*20,alpha=0.7,edge_color=color,node_size=1000,arrows=False)
        else:
            nx.draw_networkx_edges(G,pos,edgelist=[(u,v)],connectionstyle='Arc3, rad = 0.1',width=weight*20,alpha=0.7,edge_color=color,arrows=True,arrowsize=1)
    
    plt.axis('off')


MVcluster_trans_df_dataset1,mv_percentage1 =  get_trans_df1(dataset1)
MVcluster_trans_df_dataset1['group'] = dataset_name1
MVcluster_trans_df_dataset2,mv_percentage2 =  get_trans_df2(dataset2)
MVcluster_trans_df_dataset2['group'] = dataset_name2

all_df = pd.concat([MVcluster_trans_df_dataset1,MVcluster_trans_df_dataset2])

info_dict = {'previous_label':[],'later_label':[],'dataset1_mean':[],'dataset2_mean':[],'statistic_diff':[],'CI(permuted_differences)':[],'p_value':[],'significant':[]}
for previous_label in movement_order:
    for later_label in movement_order:

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
            sample_indices = np.random.choice(len(arr1), size=len(arr1), replace=True)
            
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
sns.heatmap(df_heatmap1,cmap='RdYlBu_r',ax=ax,square=True,annot=False,fmt='.2f',annot_kws={'size':20,'weight':'bold',},linewidths=1,linecolor='black', cbar=False)
plt.savefig('{}/{}_MovTrans_heatmap.png'.format(output_dir,dataset_name1),dpi=300,transparent=True)

fig,ax = plt.subplots(nrows=1,ncols=1,figsize = (10,10),dpi=300)
sns.heatmap(df_heatmap2,cmap='RdYlBu_r',ax=ax,square=True,annot=False,fmt='.2f',annot_kws={'size':20,'weight':'bold',},linewidths=1,linecolor='black', cbar=False)
plt.savefig('{}/{}_MovTrans_heatmap.png'.format(output_dir,dataset_name2),dpi=300,transparent=True)

fig,ax = plt.subplots(nrows=1,ncols=1,figsize = (10,10),dpi=300)
sns.heatmap(df_heatmap_diff,cmap='RdBu_r',ax=ax,square=True,annot=True,fmt='.1f',annot_kws={'size':10,'weight':'bold',},linewidths=2,linecolor='black',cbar=False,vmin=-0.2,vmax=0.2)
plt.savefig('{}/{}&{}_MovTrans_diff_heatmap.png'.format(output_dir,dataset_name1,dataset_name2),dpi=300,transparent=True)

fig,ax = plt.subplots(nrows=1,ncols=1,figsize = (10,10),dpi=300)
Drawing_uncolored_circle = plt.Circle(((0,0)), radius=1,color='#ECEFF1',zorder=0)
ax.set_aspect( 1 )
ax.add_artist( Drawing_uncolored_circle)
#G = nx.Graph()
G = nx.DiGraph()
for i in df_heatmap_diff.index:
    for j in df_heatmap_diff.columns:
        G.add_edge(i, j, weight=df_heatmap_diff.loc[i,j])
        #if df.loc[(df['previous_label']==i) & (df['later_label']==j),'significant'].values[0] == 'significant':
        #    G.add_edge(i, j, weight=df[(df['previous_label']==i) & (df['later_label']==j)]['statistic_diff'].values[0])
        #else:
        #    G.add_edge(i, j, weight=0)
    
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














































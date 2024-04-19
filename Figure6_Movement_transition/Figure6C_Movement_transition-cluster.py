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
import seaborn as sns
from tkinter import _flatten
import networkx as nx
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import random

InputData_path_dir = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure6_Movement_transition\Clusters'


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


movement_order = ['locomotion','exploration','maintenance','nap']
movement_color_dict ={'locomotion':'#DC2543',                     
                     'exploration':'#009688',
                     'maintenance':'#A13E97',
                     'nap':'#B0BEC5'}


dataset = Morning_lightOn_info
dataset_name = 'Morning_lightOn'

output_dir = output_dir +'\\' +dataset_name

if not os.path.exists(output_dir):                                    
    os.mkdir(output_dir) 



start = 0
end = 60


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
                         'maintenance':['grooming','scratching'],
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




sentences = []
random_seq = []
for key in dataset['video_index']:
    df = pd.read_csv(Feature_space_path[key])
    df = add_Cluster_label(df)
    temp_df = df[(df['segBoundary_start']>=start*30*60) & (df['segBoundary_end']<end*30*60)]
    
    sentences.append(list(temp_df['movement_cluster_label'].values))
    sentences_copy = list(temp_df['movement_cluster_label'].values)
    random.shuffle(sentences_copy)
    random_seq.append(sentences_copy)
mv_percentage = count_percentage(list(_flatten(sentences)))    ## 占比
mv_percentage = add_color(mv_percentage)


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
    df_count = count_percentage2(label_behind_key_sum,key)
    df_count['previous_lable'] = key
    df_count = df_count[['previous_lable','later_label','count','percentage']]
    df_list.append(df_count)
    
all_df = pd.concat(df_list,axis=0)
all_df.reset_index(drop=True,inplace=True)



trans_df = pd.DataFrame()
for i in all_df.index:
    previous_lable = all_df.loc[i,'previous_lable']
    later_label = all_df.loc[i,'later_label']
    trans_df.loc[previous_lable,later_label] = all_df.loc[i,'percentage']

aspect = 20
pad_fraction = 0.5
fig,ax = plt.subplots(nrows=1,ncols=1,figsize = (10,10),dpi=600)
sns.heatmap(trans_df,cmap='RdYlBu_r',square=True,cbar=False,linecolor='white',linewidths=1,annot=True,annot_kws={"fontsize":20},vmin=0,vmax=1,cbar_kws={'shrink': 0.8,'ticks': np.arange(0,1.1,0.1)})
#plt.axis('off')

plt.savefig('{}/{}_Movtrans_heatmap.png'.format(output_dir,dataset_name),dpi=600)
plt.show()


fig,ax = plt.subplots(nrows=1,ncols=1,figsize = (10,10),dpi=600)
#G = nx.Graph()
G = nx.MultiDiGraph()
for i in all_df['previous_lable'].unique():
    for j in all_df['later_label'].unique():
        weight=all_df[(all_df['previous_lable']==i) & (all_df['later_label']==j)]['percentage'].values[0]
        G.add_edge(i, j, weight=all_df[(all_df['previous_lable']==i) & (all_df['later_label']==j)]['percentage'].values[0])
    
#pos = pos_13movement
pos = nx.circular_layout(G)
pos2 = {'locomotion': np.array([1.2, -0.2]),
 'exploration': np.array([0,  1]),
 'maintenance': np.array([-1, 0]),
 'nap': np.array([0, -1])}
nx.draw_networkx_nodes(G,pos,alpha=1,node_size=mv_percentage.percentage * 10000,node_color=mv_percentage.color,edgecolors='black',linewidths=3)
print(mv_percentage)

for (u,v,d) in G.edges(data=True):
    color = movement_color_dict[u]
    weight = d['weight']
    if u == v:
        nx.draw_networkx_edges(G,pos2,edgelist=[(u,v)],connectionstyle='Arc3, rad = 100',width=weight*20,alpha=0.7,edge_color=color,node_size=1000,arrows=False)
    else:
        nx.draw_networkx_edges(G,pos,edgelist=[(u,v)],connectionstyle='Arc3, rad = 0.1',width=weight*20,alpha=0.7,edge_color=color,arrows=True,arrowsize=1)
#edge_labels = nx.draw_networkx_edge_labels(G, pos=pos2,font_size=50,)
plt.axis('off')
plt.savefig('{}/{}_MovClusterTrans_network.png'.format(output_dir,dataset_name),dpi=600,transparent=True)





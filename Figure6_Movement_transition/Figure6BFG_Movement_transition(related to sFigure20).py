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
import random


InputData_path_dir =r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure6_Movement_transition'


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


movement_order = ['running','trotting','left_turning','right_turning','walking','stepping','sniffing',
                  'rising','hunching','rearing','climbing','jumping','grooming','scratching','pausing',]

cluster_color_dict={'locomotion':'#DC2543',                     
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
        
        num += 1
    df['percentage'] = df['count']/df['count'].sum()
    
    return(df)



def add_color(df):
    for i in df.index:
        df.loc[i,'color'] = movement_color_dict[df.loc[i,'later_label']]
    return(df)



sentences = []
for key in dataset['video_index']:
    df = pd.read_csv(Feature_space_path[key])
    temp_df = df[(df['segBoundary_start']>=start*30*60) & (df['segBoundary_end']<end*30*60)]
    sentences.append(list(temp_df['revised_movement_label'].values))
    
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
            label_behind_key = []
        
        #index_ID = np.array(find_indices(sentence,key))
        #if 0 in index_ID:
        #    index_ID = np.delete(index_ID, find_indices(index_ID,0)[0])
        #label_before_key = np.array(sentence)[index_ID-1]
        
        label_behind_key_sum.extend(label_behind_key)
    if len(label_behind_key_sum) > 0:
        df_count = count_percentage2(label_behind_key_sum,key)
        df_count['previous_label'] = key
        df_count = df_count[['previous_label','later_label','count','percentage']]
        df_list.append(df_count)
    else:
        df_count = pd.DataFrame(data=movement_order,columns=['later_label'],index=range(len(movement_order)))
        df_count['count'] = 0
        df_count['percentage'] = 0
        df_count['previous_label'] = key
        df_count = df_count[['previous_label','later_label','count','percentage']]
        df_list.append(df_count)
    
all_df = pd.concat(df_list,axis=0)
all_df.reset_index(drop=True,inplace=True)

rising_df = all_df[all_df['previous_label'] == 'rising']
print(rising_df)
fig,ax = plt.subplots(nrows=1,ncols=1,figsize = (10,10),dpi=300)
sns.heatmap(rising_df[['count']],cmap='flare',square=True,cbar=False,linecolor='white',linewidths=1,annot=True,annot_kws={'size':10},fmt='.0f')
plt.savefig('{}/{}_next_Mov_of_Sniffing.png'.format(output_dir,dataset_name),dpi=300)
plt.axis('off')
plt.show()


trans_df = pd.DataFrame()
for i in all_df.index:
    previous_label = all_df.loc[i,'previous_label']
    later_label = all_df.loc[i,'later_label']
    trans_df.loc[previous_label,later_label] = all_df.loc[i,'percentage']


fig,ax = plt.subplots(nrows=1,ncols=1,figsize = (10,10),dpi=300)
sns.heatmap(trans_df,cmap='flare',square=True,cbar=False,linecolor='white',linewidths=1,)
plt.axis('off')
plt.savefig('{}/{}_Movtrans_heatmap.png'.format(output_dir,dataset_name),dpi=300)
plt.show()


fig,ax = plt.subplots(nrows=1,ncols=1,figsize = (10,10),dpi=300)
#G = nx.Graph()
G = nx.MultiDiGraph()
for i in all_df['previous_label'].unique():
    for j in all_df['later_label'].unique():
        weight=all_df[(all_df['previous_label']==i) & (all_df['later_label']==j)]['percentage'].values[0]
        G.add_edge(i, j, weight=all_df[(all_df['previous_label']==i) & (all_df['later_label']==j)]['percentage'].values[0])
    
#pos = pos_13movement
pos = nx.circular_layout(G)
nx.draw_networkx_nodes(G,pos,alpha=1,node_size=mv_percentage.percentage * 10000,node_color=mv_percentage.color,edgecolors='black',linewidths=3)#mv_percentage.color) 

for (u,v,d) in G.edges(data=True):
    color = movement_color_dict[u]
    weight = d['weight']
    if u == v:
        nx.draw_networkx_edges(G,pos,edgelist=[(u,v)],connectionstyle='Arc3, rad = 100',width=weight*20,alpha=0.7,edge_color=color,node_size=10,arrows=True)
    else:
        if weight > 0.05:
            nx.draw_networkx_edges(G,pos,edgelist=[(u,v)],connectionstyle='Arc3, rad = 0.1',width=weight*20,alpha=0.7,edge_color=color,arrows=True,arrowsize=1)

plt.axis('off')
plt.savefig('{}/{}_Movtrans_network.png'.format(output_dir,dataset_name),dpi=300,)#transparent=True)



G2 = nx.MultiDiGraph()
for i in all_df['previous_label'].unique():
    for j in all_df['later_label'].unique():
        weight=all_df[(all_df['previous_label']==i) & (all_df['later_label']==j)]['percentage'].values[0]
        if weight > 0.05:
            G2.add_edge(i, j, weight=all_df[(all_df['previous_label']==i) & (all_df['later_label']==j)]['percentage'].values[0])
d1 = nx.degree_centrality(G2)
df1 = pd.DataFrame(data=d1,index=['degree_centrality'])

d2 = nx.closeness_centrality(G2)
df2 = pd.DataFrame(data=d2,index=['closeness_centrality'])

d3 = nx.betweenness_centrality(G2)
df3 = pd.DataFrame(data=d3,index=['betweenness_centrality'])

df_centrality1 = pd.concat([df1,df2,df3]).T
df_centrality1['movement_label'] = df_centrality1.index
df_centrality1['color'] = df_centrality1['movement_label'].map(movement_color_dict)
df_centrality1['experiment_time'] = 'day'



fig = plt.figure(figsize=(10,10),dpi=600)
ax = fig.add_subplot(projection='3d')

for index in df_centrality1.index:
    x = df_centrality1.loc[index,'closeness_centrality']
    y = df_centrality1.loc[index,'degree_centrality']
    z = df_centrality1.loc[index,'betweenness_centrality']
    c = df_centrality1.loc[index,'color']
    mv = df_centrality1.loc[index,'movement_label']
    e_time= df_centrality1.loc[index,'experiment_time']
    ax.scatter(xs=x,ys=y,zs=z,c=c,ec='black',linewidth=2,alpha=0.8,s=1200)

ax.set_xlim(0,1.1)
ax.set_ylim(0,1.5)
ax.set_zlim(0,0.3)

ax.set_xticks(np.arange(0,1.1,0.25),np.arange(0,1.1,0.25))
ax.set_yticks(np.arange(0,1.6,0.5),np.arange(0,1.6,0.5))
ax.set_zticks(np.arange(0,0.31,0.1),np.arange(0,0.31,0.1))

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

plt.savefig('{}/{}_Movtrans_3Dcentrality.png'.format(output_dir,dataset_name),dpi=600)



#df_centrality1['degree_centrality'] = (df_centrality1['degree_centrality']/df_centrality1['degree_centrality'].sum())*100
df_centrality1['degree_centrality'] = (df_centrality1['degree_centrality']/df_centrality1['degree_centrality'].sum())*100
#df_centrality1['closeness_centrality'] = (df_centrality1['closeness_centrality']/df_centrality1['closeness_centrality'].sum())*100
#df_centrality1['betweenness_centrality'] = (df_centrality1['betweenness_centrality']/df_centrality1['betweenness_centrality'].sum())*100

df_centrality1['sum'] = df_centrality1['degree_centrality'] +  df_centrality1['closeness_centrality'] + df_centrality1['betweenness_centrality']
df_centrality1.reset_index(drop=True,inplace=True)

print(df_centrality1)
df_centrality1.sort_values(by=['betweenness_centrality'],ascending=False,inplace=True)
key_nodes = df_centrality1.head(5)['movement_label'].values
print(key_nodes)
rank_num = 0
for key_node in key_nodes:
    rank_num +=1
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize = (10,10),dpi=300)

    G = nx.MultiDiGraph()
    for i in all_df['previous_label'].unique():
        if i == key_node:
            for j in all_df['later_label'].unique():
                weight =all_df[(all_df['previous_label']==key_node) & (all_df['later_label']==j)]['percentage'].values[0]
                if (weight > 0.05):
                    G.add_edge(i, j, weight=all_df[(all_df['previous_label']==i) & (all_df['later_label']==j)]['percentage'].values[0])

    color_list = []
    for node2 in G.nodes._nodes.keys():
        mv = node2
        color = movement_color_dict[mv]
        color_list.append(color)
        

    pos = nx.kamada_kawai_layout(G)

    nx.draw_networkx_nodes(G,pos,alpha=0.8,node_size=1500,node_color=color_list,edgecolors='black',linewidths=3) 

    for (u,v,d) in G.edges(data=True):
        color = movement_color_dict[u]
        weight = d['weight']
        nx.draw_networkx_edges(G,pos,edgelist=[(u,v)],connectionstyle='Arc3, rad = 0.1',width=weight*80,alpha=0.7,edge_color=color,arrows=True,arrowsize=0.1,min_target_margin=20,min_source_margin=20)

    ax.spines['left'].set_linewidth(7)
    ax.spines['right'].set_linewidth(7)
    ax.spines['top'].set_linewidth(7)
    ax.spines['bottom'].set_linewidth(7)
    plt.savefig('{}/{}_{}_rank{}_connectivities.png'.format(output_dir,key_node,rank_num,dataset_name),transparent = True,dpi=600)


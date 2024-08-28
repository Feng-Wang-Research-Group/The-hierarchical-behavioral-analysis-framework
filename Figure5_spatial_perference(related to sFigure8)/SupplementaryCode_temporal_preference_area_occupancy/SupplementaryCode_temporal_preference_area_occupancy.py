# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 10:14:38 2023

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


InputData_path_dir = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure5_spatial_perference\SupplementaryCode_temporal_preference_area_occupancy'


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
#Speed_distance_path =  


skip_file_list = [1,3,28,29,110,122] 

animal_info_csv = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Table_S1_animal_information.csv'
animal_info = pd.read_csv(animal_info_csv)
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)]


Morning_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Morning') & (animal_info['LightingCondition']=='Light-on')]
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-off')]

Stress_info = animal_info[(animal_info['ExperimentCondition']=='Stress')]


def add_location(Mov_loc_data_copy,boundary):
    Mov_loc_data_copy['data_driven_location'] = 'perimeter'   
    Mov_loc_data_copy.loc[(Mov_loc_data_copy['back_x']>(250-boundary))&(Mov_loc_data_copy['back_x']<(250+boundary))&(Mov_loc_data_copy['back_y']>(250-boundary))&(Mov_loc_data_copy['back_y']<(250+boundary)),'data_driven_location'] = 'center'
    Mov_loc_data_copy.loc[(Mov_loc_data_copy['back_x']<=(250-boundary))&(Mov_loc_data_copy['back_y']<=(250-boundary)),'data_driven_location'] = 'corner'
    Mov_loc_data_copy.loc[(Mov_loc_data_copy['back_x']<=(250-boundary))&(Mov_loc_data_copy['back_y']>=(250+boundary)),'data_driven_location'] = 'corner'
    Mov_loc_data_copy.loc[(Mov_loc_data_copy['back_x']>=(250+boundary))&(Mov_loc_data_copy['back_y']>=(250+boundary)),'data_driven_location'] = 'corner'
    Mov_loc_data_copy.loc[(Mov_loc_data_copy['back_x']>=(250+boundary))&(Mov_loc_data_copy['back_y']<=(250-boundary)),'data_driven_location'] = 'corner'
    
# =============================================================================
#     fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,10),dpi=300,constrained_layout=True,sharex=True)
#     sns.scatterplot(data = Mov_loc_data_copy,x='back_x',y='back_y',hue='data_driven_location')
#     plt.show()
# =============================================================================
    return(Mov_loc_data_copy)

def add_traditional_location(Mov_loc_data,boundary=125):
    Mov_loc_data_copy = Mov_loc_data.copy()
    #Mov_loc_data_copy['back_x'] = Mov_loc_data_copy['back_x'] - 250
    #Mov_loc_data_copy['back_y'] = Mov_loc_data_copy['back_y'] - 250
    
    Mov_loc_data_copy['traditional_location'] = 'perimeter'
    
    Mov_loc_data_copy.loc[(Mov_loc_data_copy['back_x']>125) & (Mov_loc_data_copy['back_x']<375) & (Mov_loc_data_copy['back_y']>125) & (Mov_loc_data_copy['back_y']<375),'traditional_location'] = 'center'
    return(Mov_loc_data_copy)


dataset = Morning_lightOn_info
dataset_name = 'Morning_lightOn'

output_dir = output_dir + '/{}'.format(dataset_name) 
if not os.path.exists(output_dir):                                                                       
    os.mkdir(output_dir)

MV_loc_data_list = []
for index in dataset.index:
    video_index = dataset.loc[index,'video_index']
    gender = dataset.loc[index,'gender']
    
    MV_loc_data = pd.read_csv(Movement_Label_path[video_index])  
    MV_loc_data = add_location(MV_loc_data,175)
    MV_loc_data = add_traditional_location(MV_loc_data)
    MV_loc_data['frame'] = MV_loc_data.index +1
    MV_loc_data_list.append(MV_loc_data)
    
df_combine = pd.concat(MV_loc_data_list,axis=0)
df_combine.reset_index(drop=True,inplace=True)


start = 0
end = 0
location_color_dict = {'center':'#DBEDF4',
                       'perimeter':'#A5D2DC',
                       'corner':'#657999',}

tranditional_location_color_dict = {'center':'#E87E78',
                                    'perimeter':'#E85850',
                                    'corner':'#D0241A',}

location_order = location_order = {'center':1,
                  'perimeter':2,
                  'corner':3}

spatial_preference_ratio = pd.DataFrame()
    
num = 0
for i in range(10,61,10):             ## calculate by per 10 min
    end = i
    temp_df = df_combine.loc[(df_combine['frame']>=start*30*60) & (df_combine['frame']<=end*30*60),:]
    count_df = temp_df.value_counts('data_driven_location').to_frame(name='count')
    count_df['percentage'] = count_df['count'] / count_df['count'].sum()
    
    count_df['color'] = count_df.index.map(location_color_dict)
    count_df['plot_order'] = count_df.index.map(location_order)
    count_df.sort_values(by=['plot_order'],ascending=True,inplace=True)

    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,10),dpi=600)
    ax.pie(count_df['percentage'],counterclock=True,normalize=False,startangle = 90,
            #explode = [0.08,0.08,0.08,0.08],
            #autopct= '%1.1f%%',pctdistance=1.5,
            #textprops={'fontsize':60, 'color': 'black','fontfamily':'arial','weight': 'bold'},
            colors = count_df['color'],
            radius = 0.8,
            wedgeprops = {'linewidth':5, 'edgecolor': "black"},)
            #hatch=['.', '.', '.', '.'])
            
    #center_ratio = count_df.loc['center','percentage']/count_df.loc['perimeter','percentage']
    ax.text(-1,1,'{:.1%}'.format(count_df['percentage'].values[0]),fontdict={'fontsize':70, 'color': 'black','fontfamily':'arial','weight': 'bold'})
    ax.text(0.9,0.3,'{:.1%}'.format(count_df['percentage'].values[1]),fontdict={'fontsize':70, 'color': 'black','fontfamily':'arial','weight': 'bold'})
    ax.text(-1,-1.2,'{:.1%}'.format(count_df['percentage'].values[2]),fontdict={'fontsize':70, 'color': 'black','fontfamily':'arial','weight': 'bold'})
    
    
    spatial_preference_ratio.loc[num,'time'] = i
    spatial_preference_ratio.loc[num,'method'] = 'data_driven'
    spatial_preference_ratio.loc[num,'center2perimeter'] = count_df.loc['center','percentage']/count_df.loc['perimeter','percentage']
    spatial_preference_ratio.loc[num,'center2corner'] = count_df.loc['center','percentage']/count_df.loc['corner','percentage']
    spatial_preference_ratio.loc[num,'perimeter2corner'] = count_df.loc['perimeter','percentage']/count_df.loc['corner','percentage']
    
    num += 1
    
    Per_Cor = count_df.loc['perimeter','percentage']/count_df.loc['corner','percentage']
    if Per_Cor >= 1:
        anno_color2 = '#C51162'
    else:
        anno_color2 = '#757575'
    
    ax.text(1.75,-1,'Per:Cor={:.2f}'.format(Per_Cor),fontdict={'fontsize':50, 'color':anno_color2,'fontfamily':'arial','weight': 'black'},    #9C27B0
            verticalalignment="top",horizontalalignment="right")
    plt.savefig(r'{}\{}_{}min_accumulate.png'.format(output_dir,dataset_name,i),dpi=600,transparent=True,bbox_inches='tight',pad_inches=0.1)
    
    

for i in range(10,61,10):             ## calculate by per 10 min## traditional
    end = i
    temp_df = df_combine.loc[(df_combine['frame']>=start*30*60) & (df_combine['frame']<=end*30*60),:]
    count_df = temp_df.value_counts('traditional_location').to_frame(name='count')
    count_df['percentage'] = count_df['count'] / count_df['count'].sum()
    count_df['color'] = count_df.index.map(tranditional_location_color_dict)
    count_df['plot_order'] = count_df.index.map(location_order)
    count_df.sort_values(by=['plot_order'],ascending=True,inplace=True)

    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,10),dpi=600)
    ax.pie(count_df['percentage'],counterclock=True,normalize=False,startangle = 90,
            #explode = [0.08,0.08,0.08,0.08],
            #autopct= '%1.1f%%',pctdistance=1.4,
            #textprops={'fontsize':60, 'color': 'black','fontfamily':'arial','weight': 'bold'},
            colors = count_df['color'],
            radius = 0.8,
            wedgeprops = {'linewidth':5, 'edgecolor': "black"},)
            #hatch=['.', '.', '.', '.'])
    
    ax.text(-0.5,1,'{:.1%}'.format(count_df['percentage'].values[0]),fontdict={'fontsize':70, 'color': 'black','fontfamily':'arial','weight': 'bold'})
    ax.text(0.7,-0.8,'{:.1%}'.format(count_df['percentage'].values[1]),fontdict={'fontsize':70, 'color': 'black','fontfamily':'arial','weight': 'bold'})   
    
    spatial_preference_ratio.loc[num,'time'] = i
    spatial_preference_ratio.loc[num,'method'] = 'triditional'
    spatial_preference_ratio.loc[num,'center2perimeter'] = count_df.loc['center','percentage']/count_df.loc['perimeter','percentage']
    
    num += 1
    
    plt.savefig(r'{}\{}_TraditionalCenter_{}min_accumulate.png'.format(output_dir,dataset_name,i),dpi=600,bbox_inches='tight',transparent=True,pad_inches=0.1)




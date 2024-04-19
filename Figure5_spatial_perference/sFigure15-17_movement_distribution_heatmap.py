# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 13:47:55 2023

@author: Jialin Ye
@institution: SIAT
@Contact_email: jl.ye@siat.ac.cn

"""

import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from scipy import stats
from matplotlib import colors
from matplotlib.colors import LogNorm
import matplotlib as mpl

InputData_path_dir = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure5_spatial_perference\sFigure15-17_movement_distribution_heatmap'


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
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-off')]


stress_animal_info = animal_info[animal_info['ExperimentCondition']=='Stress']

stress_animal_info_male = stress_animal_info[stress_animal_info['gender']=='male']
stress_animal_info_female = stress_animal_info[stress_animal_info['gender']=='female']


movement_order = ['running','trotting','left_turning','right_turning','walking','stepping','sniffing',
                  'rising','hunching','rearing','climbing','jumping','grooming','scratching','pausing',]


movement_color_dict = {'running':'#FF3030',
                       'trotting':'#F06292',
                       'walking':'#FF5722',
                       'right_turning':'#F29B78',
                       'left_turning':'#FFBFB4',                       
                       'stepping':'#A1887F',  
                       'rising':'#FFEA00',    #'#FFEA00'  ####FFEE58
                       'hunching':'#ECAD4F',
                       'rearing':'#C0CA33',
                       'climbing':'#2E7939',                           
                       'jumping':'#80DEEA',
                       'sniffing':'#2C93CB',                       
                       'grooming':'#A13E97',
                       'scratching':'#00ACC1',
                       'pausing':'#B0BEC5',}
 

dataset = stress_animal_info
dataset_info = 'stress_animal'

output_dir = output_dir + '\\' + dataset_info
if not os.path.exists(output_dir):                                    
    os.mkdir(output_dir) 

MoV_file_list = []
for file_name in list(dataset['video_index']):
    MoV_file = pd.read_csv(Movement_Label_path[file_name])
    MoV_file_list.append(MoV_file)
df_all = pd.concat(MoV_file_list,axis=0)


def plot(df,mv):
    fig,ax = plt.subplots(nrows=1, ncols=1,figsize=(10,10),dpi=300)
    sns.kdeplot(df['back_x'], df['back_y'], shade=True,color=movement_color_dict[mv])

def plot2(df,mv):
    fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10),constrained_layout=True,dpi=300)
    x = df['back_x']
    y = df['back_y']
    xmin = 0
    xmax = 500
    ymin = 0
    ymax = 500
    X, Y = np.mgrid[xmin:xmax:250j, ymin:ymax:250j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    
    ax.imshow(np.rot90(Z), cmap='RdBu_r',interpolation='gaussian',extent=[xmin, xmax, ymin, ymax],zorder=0)
    #ax.set_title('heatmap')
    
    ax.plot((500,0), (0,0),c='black',lw=3,zorder=2)
    ax.plot((0,0), (0,500),c='black',lw=3,zorder=2)
    ax.plot((500,0), (500,500),c='black',lw=3,zorder=2)
    ax.plot((500,500), (500,0),c='black',lw=3,zorder=2)
    ax.set_title(mv,fontsize=15)
    ax.axis('equal')
    #plt.suptitle(file_name)
    ax.set_xlim(0,500)
    ax.set_ylim(0,500)
    plt.axis('off')
    plt.savefig('{}/{}_{}_DensityMovementHeatmap.png'.format(output_dir,dataset_info,mv),dpi=300)



def plot1(df,mv):
    x = df['back_x']
    y = df['back_y']
    
    cmap = mpl.cm.RdBu_r
    norm = mpl.colors.Normalize(vmin=0.1, vmax=1)
    
    fig,ax = plt.subplots(nrows=1, ncols=1,figsize=(10,10),dpi=300,constrained_layout=True)
    im = ax.hist2d(x,y,bins=100,range=[[0,500], [0,500]],cmap='RdBu_r')#,vmin=0,vmax=600)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax,shrink=0.8)
    plt.title(mv,fontsize=25)
    
    plt.axis('off')
    plt.savefig('{}/{}_{}_movementHeatmap.png'.format(output_dir,dataset_info,mv),dpi=300,bbox_inches='tight')


for mv in movement_order:
    temp_df = df_all[df_all['revised_movement_label']==mv]
    plot1(temp_df,mv)
    #plot2(temp_df)
    
    
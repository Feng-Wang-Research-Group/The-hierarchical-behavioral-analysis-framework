# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 18:19:29 2023

@author: Jialin Ye
@institution: SIAT
@Contact_email: jl.ye@siat.ac.cn

"""

import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import statsmodels.api as sm
from scipy.stats import gaussian_kde

InputData_path_dir = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure5_spatial_perference\Occupancy_in_difference_regions'


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

def get_color(name):
    if name.startswith('Morning'):
        color = '#F5B25E'
    elif name.startswith('Afternoon'):
        color = '#936736'
    elif name.startswith('Night_lightOn'):
        color = '#398FCB'
    elif name.startswith('Night_lightOff'):
        color = '#003960'
    elif (name.startswith('Stress')) | (name.startswith('stress')):
        color = '#f55e6f'
    else:
        print(name)
    
    return(color)


dataset1 = Morning_lightOn_info
dataset1_name = 'Morning_lightOn'
dataset1_color = get_color(dataset1_name)

dataset2 = stress_animal_info
dataset2_name = 'stress_animal'
dataset2_color = get_color(dataset2_name)

varible = 'location'
location_list = ['center','perimeter','corner']


boundary10_Moring = 171
boundary10_Stress = 183

boundary1060_Moring = 175
boundary1060_Stress = 183


time_window1 = 10
time_window2 = 60

if time_window1 == 0:
    boundary1 = boundary10_Moring
    boundary2 = boundary10_Stress
elif time_window1 == 10:
    boundary1 = boundary1060_Moring
    boundary2 = boundary1060_Stress
else:
    print('wrong and check')

def count_MV(df):
    df_output = pd.DataFrame()
    count_df = df['revised_movement_label'].value_counts()
    num = 0
    for mv in movement_order:
        if mv in count_df.index:
            count = count_df[mv]
        else:
            count = 0
        df_output.loc[num,'revised_movement_label'] = mv
        df_output.loc[num,'count'] = count
        num += 1
    df_output['percentage'] = df_output['count']/df_output['count'].max()
    df_output = df_output.fillna(0)
    return(df_output)

def areaPointCount(df,boundary,loc_type):
    if loc_type == 'traditional_location':
        boundary = 125
        location_area = {'center':(boundary*2)*(boundary*2),
                        'perimeter':4*(250-boundary)*(boundary*2),
                        'corner':4*(250-boundary)*(250-boundary)} 
    else:
        location_area = {'center':(boundary*2)*(boundary*2),
                        'perimeter':4*(250-boundary)*(boundary*2),
                        'corner':4*(250-boundary)*(250-boundary)} 
 
    info_dict = {'location':[],'points':[],'density':[]} #,'speed':[],'movement_fraction':[]}
    
    speed_df_list = []
    mov_count_df_list = []
    
    for loc in location_list:
        points = (len(df[df[loc_type]==loc]) / len(df)) *100
        density = (points/location_area[loc]) * 10000               # %/mm2
        speed = df[df[loc_type]==loc]['smooth_speed'].values
        
        speed_df = pd.DataFrame(data=speed,columns=['speed'])
        speed_df['loc_type'] = loc_type
        speed_df['location'] = loc
        speed_df_list.append(speed_df)
        movement_fraction = count_MV(df[df[loc_type]==loc])
        movement_fraction['loc_type'] = loc_type
        movement_fraction['location'] = loc
        mov_count_df_list.append(movement_fraction)

        info_dict['location'].append(loc)
        info_dict['points'].append(points)
        info_dict['density'].append(density)

        
    df_points = pd.DataFrame(info_dict)
    df_speed = pd.concat(speed_df_list,axis=0)
    df_mov = pd.concat(mov_count_df_list,axis=0)
    return(df_points,df_speed,df_mov)
    
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

# =============================================================================
#     Mov_loc_data_copy.loc[(Mov_loc_data_copy['back_x']<=(250-boundary))&(Mov_loc_data_copy['back_y']<=(250-boundary)),'traditional_location'] = 'corner'
#     Mov_loc_data_copy.loc[(Mov_loc_data_copy['back_x']<=(250-boundary))&(Mov_loc_data_copy['back_y']>=(250+boundary)),'traditional_location'] = 'corner'
#     Mov_loc_data_copy.loc[(Mov_loc_data_copy['back_x']>=(250+boundary))&(Mov_loc_data_copy['back_y']>=(250+boundary)),'traditional_location'] = 'corner'
#     Mov_loc_data_copy.loc[(Mov_loc_data_copy['back_x']>=(250+boundary))&(Mov_loc_data_copy['back_y']<=(250-boundary)),'traditional_location'] = 'corner'
# =============================================================================
    
# =============================================================================
#     fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,10),dpi=300,constrained_layout=True,sharex=True)
#     sns.scatterplot(data = Mov_loc_data_copy,x='back_x',y='back_y',hue='traditional_location')
#     plt.show()
# =============================================================================
    return(Mov_loc_data_copy)

def normalize(arr):
    arr = np.array(arr)
    y_min = arr.min()
    y_max = arr.max()
    norm_arr = ((arr-y_min)/(y_max-y_min)) * 100
    return (norm_arr)

def get_nor_kde_value(arr):  
    xx = np.linspace(0,500,1000)
    kde = gaussian_kde(arr)
    yy = kde(xx)
    

    return(xx,yy)

def location_statistic(dataset,time_window1,time_window2,boundary):
    traditional_points_list = []
    data_driven_points_list = []
    
    traditional_speed_list = []
    data_driven_speed_list = []
      
    traditional_Mov_list = []
    data_driven_Mov_list = []
    

    for index in dataset.index:
        video_index = dataset.loc[index,'video_index']
        ExperimentCondition = dataset.loc[index,'ExperimentCondition']
        LightingCondition = dataset.loc[index,'LightingCondition']
        gender = dataset.loc[index,'gender']
        
        Mov_loc_data = pd.read_csv(Movement_Label_path[video_index])
        Mov_loc_data = add_traditional_location(Mov_loc_data,boundary=125)
        Mov_loc_data = add_location(Mov_loc_data,boundary=boundary)
        
        Mov_loc_data = Mov_loc_data.iloc[time_window1*30*60:time_window2*30*60,:]
        
        traditional_points_df,traditional_speed_df,traditional_movement_fraction =areaPointCount(Mov_loc_data,boundary,'traditional_location')
        
        traditional_points_df['statistic_type'] = 'traditional_location'
        traditional_points_df['video_index'] = video_index
        traditional_points_df['ExperimentCondition'] = ExperimentCondition + '_' + LightingCondition
        #traditional_statistic_df['LightingCondition'] = LightingCondition
        traditional_points_df['gender'] = gender
        traditional_points_list.append(traditional_points_df)
        
        traditional_speed_df['statistic_type'] = 'traditional_location'
        traditional_speed_df['video_index'] = video_index
        traditional_speed_df['ExperimentCondition'] = ExperimentCondition + '_' + LightingCondition
        #traditional_statistic_df['LightingCondition'] = LightingCondition
        traditional_speed_df['gender'] = gender
        traditional_speed_list.append(traditional_speed_df)
        
        traditional_movement_fraction['statistic_type'] = 'traditional_location'
        traditional_movement_fraction['video_index'] = video_index
        traditional_movement_fraction['ExperimentCondition'] = ExperimentCondition + '_' + LightingCondition
        #traditional_statistic_df['LightingCondition'] = LightingCondition
        traditional_movement_fraction['gender'] = gender
        traditional_Mov_list.append(traditional_movement_fraction)
        
        data_driven_points_df,data_driven_speed_df,data_driven_movement_fraction =areaPointCount(Mov_loc_data,boundary,'data_driven_location')
        data_driven_points_df['statistic_type'] = 'data_driven_location'
        data_driven_points_df['video_index'] = video_index
        data_driven_points_df['ExperimentCondition'] = ExperimentCondition + '_' + LightingCondition
        #data_driven_statistic_df['LightingCondition'] = LightingCondition
        data_driven_points_df['gender'] = gender
        data_driven_points_list.append(data_driven_points_df)
        
        data_driven_speed_df['statistic_type'] = 'data_driven_location'
        data_driven_speed_df['video_index'] = video_index
        data_driven_speed_df['ExperimentCondition'] = ExperimentCondition + '_' + LightingCondition
        #data_driven_statistic_df['LightingCondition'] = LightingCondition
        data_driven_speed_df['gender'] = gender
        data_driven_speed_list.append(data_driven_speed_df)
        
        data_driven_movement_fraction['statistic_type'] = 'data_driven_location'
        data_driven_movement_fraction['video_index'] = video_index
        data_driven_movement_fraction['ExperimentCondition'] = ExperimentCondition + '_' + LightingCondition
        #data_driven_statistic_df['LightingCondition'] = LightingCondition
        data_driven_movement_fraction['gender'] = gender
        data_driven_Mov_list.append(data_driven_movement_fraction)
    
    traditional_points = pd.concat(traditional_points_list)
    data_driven_points = pd.concat(data_driven_points_list)
    
    traditional_speed = pd.concat(traditional_speed_list)
    data_driven_speed = pd.concat(data_driven_speed_list)
    
    traditional_mov = pd.concat(traditional_Mov_list)
    data_driven_mov = pd.concat(data_driven_Mov_list)
    
    return(traditional_points,data_driven_points,traditional_speed,data_driven_speed,traditional_mov,data_driven_mov)


traditional_points1,data_driven_points1,traditional_speed1,data_driven_speed1,traditional_mov1,data_driven_mov1 = location_statistic(dataset1,time_window1,time_window2,boundary1)
traditional_points2,data_driven_points2,traditional_speed2,data_driven_speed2,traditional_mov2,data_driven_mov2 = location_statistic(dataset2,time_window1,time_window2,boundary2)

traditional_points_combine = pd.concat([traditional_points1,traditional_points2],axis=0)
traditional_points_combine.reset_index(drop=True,inplace=True)
data_driven_points_combine = pd.concat([data_driven_points1,data_driven_points2],axis=0)
data_driven_points_combine.reset_index(drop=True,inplace=True)
traditional_points_combine.to_csv('{}/points_TraditionalArea.csv'.format(output_dir))
data_driven_points_combine.to_csv('{}/points_DataDrivenArea.csv'.format(output_dir))


traditional_speed_combine = pd.concat([traditional_speed1,traditional_speed2],axis=0)
traditional_speed_combine.reset_index(drop=True,inplace=True)
data_driven_speed_combine = pd.concat([data_driven_speed1,data_driven_speed2],axis=0)
data_driven_speed_combine.reset_index(drop=True,inplace=True)

traditional_speed_combine.to_csv('{}/speed_TraditionalArea.csv'.format(output_dir))
data_driven_speed_combine.to_csv('{}/speed_DataDrivenArea.csv'.format(output_dir))

traditional_mov_combine = pd.concat([traditional_mov1,traditional_mov2],axis=0)
traditional_mov_combine.reset_index(drop=True,inplace=True)
data_driven_mov_combine = pd.concat([data_driven_mov1,data_driven_mov2],axis=0)
data_driven_mov_combine.reset_index(drop=True,inplace=True)

traditional_mov_combine.to_csv('{}/movement_percent_TraditionalArea.csv'.format(output_dir))
data_driven_mov_combine.to_csv('{}/movement_percent_Data_DrivenArea.csv'.format(output_dir))



fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(10,5),dpi=300,sharey=True)
sns.barplot(data=traditional_points_combine,x='location',y='density',order=['center','perimeter'],hue='ExperimentCondition',ax=ax[0])
ax[0].set_title('Traditional_center_region')
sns.barplot(data=data_driven_points_combine,x='location',y='density',hue='ExperimentCondition',ax=ax[1])
ax[1].set_title('Data-driven_center_region')
plt.savefig('{}/points_distribution_{}_{}min.png'.format(output_dir,time_window1,time_window2))

plot_speed_dataset = data_driven_speed_combine
fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(10,5),dpi=300,sharey=True)
sns.violinplot(data=plot_speed_dataset,x='location',y='speed',order=['center','perimeter'],hue='ExperimentCondition',ax=ax[0])
ax[0].set_title('Traditional_center_region')
sns.violinplot(data=plot_speed_dataset,x='location',y='speed',order=['center','perimeter','corner'],hue='ExperimentCondition',ax=ax[1])
ax[1].set_title('Data-driven_center_region')
plt.savefig('{}/speed_distribution_{}_{}min.png'.format(output_dir,time_window1,time_window2))


def plot_loc_speed_distribution(data):
    color_list = ['#9AC9DB','#2878B5','#F8AC8C','#C82423']
    for loc in location_list:
        fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(9,8),dpi=300,sharex=True)
        
        group1 = data[(data['ExperimentCondition'].str.contains('Morning'))&(data['location']==loc)]
        group2 = data[data['ExperimentCondition'].str.contains('Stress')&(data['location']==loc)]
        
        num = 0
        for con in ['male','female']:
            x1 = group1[group1['gender']==con]['speed'].values
            x1 = x1[~np.isnan(x1)]
            xx,yy = get_nor_kde_value(x1)
            yy_normalize = normalize(yy)
            #ax[0].hist(x1,bins=50,color=color_list[num],alpha=0.5, label=con,density=True)
            ax[0].plot(xx, yy_normalize, label=con,alpha=1,lw=5,color=color_list[num])
            
            xx = list(xx)
            yy_normalize = list(yy_normalize)
            xx.insert(0,0)
            yy_normalize.insert(0,0)
            
            ax[0].fill(xx, yy_normalize,alpha=0.6,color=color_list[num])
            
            ax[0].spines['top'].set_visible(False)
            ax[0].spines['right'].set_visible(False)
            ax[0].spines['bottom'].set_linewidth(3)
            ax[0].spines['left'].set_linewidth(3)
            ax[0].set_yticks(np.arange(0,101,25))
            #ax[0].xaxis.set_major_formatter(plt.NullFormatter())
            #ax[0].yaxis.set_major_formatter(plt.NullFormatter())
            ax[0].tick_params(length=7,width=3)
                                
            ax[0].legend()
            ax[0].set_ylim(0,105)
            num += 1
        
        for con2 in ['male','female']:
            x2 = group2[group2['gender']==con2]['speed'].values
            x2 = x2[~np.isnan(x2)]
            xx2,yy2 = get_nor_kde_value(x2)
            yy_normalize2 = normalize(yy2)
            ax[1].plot(xx2, yy_normalize2, label=con2,alpha=1,lw=5,color=color_list[num])
            
            xx2 = list(xx2)
            yy_normalize2 = list(yy_normalize2)
            xx2.insert(0,0)
            yy_normalize2.insert(0,0)
            
            ax[1].fill(xx2, yy_normalize2,alpha=0.6,color=color_list[num])
            ax[1].spines['bottom'].set_visible(False)
            ax[1].spines['right'].set_visible(False)
            ax[1].spines['top'].set_linewidth(3)
            ax[1].spines['left'].set_linewidth(3)
            ax[1].set_yticks(np.arange(0,101,25))
            #ax[1].xaxis.set_major_formatter(plt.NullFormatter())
            ax[1].xaxis.set_ticks_position('top')
            #ax[1].yaxis.set_major_formatter(plt.NullFormatter())
            ax[1].tick_params(length=7,width=3)
            ax[1].legend()
            ax[1].set_ylim(0,105)
            ax[1].invert_yaxis()
            num += 1
            
        plt.subplots_adjust(wspace =0, hspace =0)
        plt.suptitle('{}_smooth_speed'.format(loc),fontsize=20)
        #plt.xlim(-5,450)
        #plt.xticks([])
        #plt.yticks([])        
        #plt.axis('off')
        plt.savefig('{}/{}_speed_distribution_{}_{}min.png'.format(output_dir,loc,time_window1,time_window2))
        plt.show()

plot_loc_speed_distribution(data_driven_speed_combine)




def plot_movement(data):
    for dataset_name in data['ExperimentCondition'].unique():
        data_batch1 = data[data['ExperimentCondition']==dataset_name]
        for loc in location_list:
            temp_df = data_batch1[data_batch1['location']==loc]
            angles=np.arange(0,2*np.pi,2*np.pi/temp_df.shape[0])                    # 准备好半径
            radius=np.array(temp_df['average_percentage'])
            
            fig=plt.figure(figsize=(10,10),dpi=300)
            fig = fig.add_subplot(projection='polar')
            fig.grid(axis='both', linewidth =5,zorder=0,color='#424242')
            fig.set_theta_offset(np.pi/2)
            fig.set_theta_direction(-1)
            fig.set_rlabel_position(0)
            
            plt.bar(angles,radius,width=2*np.pi/temp_df.shape[0],edgecolor='white',lw=2,label=movement_order,color=movement_color_dict.values(),zorder=3)
            plt.xticks([])
            plt.tick_params(width=5)
            plt.ylim(0,1)
            fig.spines['polar'].set_linewidth(10)
            
            fig.xaxis.set_major_formatter(plt.NullFormatter())
            fig.yaxis.set_major_formatter(plt.NullFormatter())
            plt.title('{}: {}'.format(dataset_name,loc),family='arial',color='black', weight='normal', size =20,loc='left')
            plt.savefig('{}/{}_{}_movement_percent_{}_{}min.png'.format(output_dir,dataset_name,loc,time_window1,time_window2),transparent=True,dpi=300)


def Time_movemet(data):
    df_loc_movement = pd.DataFrame()
    num = 0
    for dataset_name in data['ExperimentCondition'].unique():
        data_batch1 = data[data['ExperimentCondition']==dataset_name]
        for loc in location_list:
            loc_df = data_batch1[data_batch1['location']==loc]
            for mv in movement_order:
                mv_df = loc_df[loc_df['revised_movement_label']==mv]
                value = np.mean(mv_df['percentage'].values)
                df_loc_movement.loc[num,'ExperimentCondition'] = dataset_name
                df_loc_movement.loc[num,'location'] = loc
                df_loc_movement.loc[num,'revised_movement_label'] = mv
                df_loc_movement.loc[num,'average_percentage'] = value
                num += 1
    return(df_loc_movement)

MLcount = Time_movemet(data_driven_mov_combine)
plot_movement(MLcount)



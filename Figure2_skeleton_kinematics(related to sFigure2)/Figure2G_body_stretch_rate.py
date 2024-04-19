# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 03:26:19 2024

@author: Jialin Ye
@institution: SIAT
@Contact_email: jl.ye@siat.ac.cn

"""

import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#### 

InputData_path_dir =  r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\01_BehaviorAtlas_collated_data"
InputData_path_dir2 = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure2_skeleton_kinematics(related to sFigure2)\movement_parametter'
if not os.path.exists(output_dir):                                    
    os.mkdir(output_dir) 

def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-')) & (file_name.endswith(content)):
            video_index = int(file_name.split('-')[1])
            file_path_dict.setdefault(video_index,file_dir+'\\'+file_name)
    return(file_path_dict)

Movement_Label_path = get_path(InputData_path_dir2,'revised_Movement_Labels.csv')
Feature_space_path = get_path(InputData_path_dir2,'Feature_Space.csv')
Skeleton_path = get_path(InputData_path_dir,'normalized_skeleton_XYZ.csv')
#Skeleton_path = get_path(InputData_path_dir2,'Cali_Data3d.csv')

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


color_list = {'nose':'#1e2c59',
              'left_ear':'#192887',
              'right_ear':'#1b3a95',
              'neck':'#204fa1',
              'left_front_limb':'#1974ba',
              'right_front_limb':'#1ea2ba',
              'left_hind_limb':'#42b799',
              'right_hind_limb':'#5cb979',
              'left_front_claw':'#7bbf57',
              'right_front_claw':'#9ec036',
              'left_hind_claw':'#beaf1f',
              'right_hind_claw':'#c08719',
              'back':'#bf5d1c',
              'root_tail':'#be3320',
              'mid_tail':'#9b1f24',
              'tip_tail':'#6a1517',}


movement_order = ['running','trotting','left_turning','right_turning','walking','stepping','sniffing',
                  'rising','hunching','rearing','climbing','jumping','grooming','scratching','pausing',]

exploration = ['sniffing','rising','hunching','rearing','climbing']

grooming = []

def get_color(name):
    if name.startswith('Morning'):
        color = '#F5B25E'
    elif name.startswith('Afternoon'):
        color = '#936736'
    elif name.startswith('Night_lightOn'):
        color = '#398FCB'
    elif name.startswith('Night_lightOff'):
        color = '#003960'
    elif name.startswith('Stress') | name.startswith('stress'):
        color = '#f55e6f'
    else:
        print(name)
    
    return(color)
        
plot_dataset = Morning_lightOn_info
dataset1_name = 'Morning_lightOn'
plot_dataset_male = plot_dataset[plot_dataset['gender']=='male']
plot_dataset_female = plot_dataset[plot_dataset['gender']=='female']
dataset1_color = get_color(dataset1_name)


plot_dataset2 = stress_animal_info
dataset2_name = 'stress_animal'
plot_dataset2_male = plot_dataset2[plot_dataset2['gender']=='male']
plot_dataset2_female = plot_dataset2[plot_dataset2['gender']=='female']
dataset2_color = get_color(dataset2_name)



def add_BodyPart_distance(df):
    df['nose-back_distance'] = np.sqrt(np.square(df['nose_x']-df['back_x'])+np.square(df['nose_y']-df['back_y'])+np.square(df['nose_z']-df['back_z']))
    df['neck-back_distance'] = np.sqrt(np.square(df['neck_x']-df['neck_x'])+np.square(df['neck_y']-df['back_y'])+np.square(df['neck_z']-df['back_z']))
    df['left_front_limb-back_distance'] = np.sqrt(np.square(df['left_front_limb_x']-df['back_x'])+np.square(df['left_front_limb_y']-df['back_y'])+np.square(df['left_front_limb_z']-df['back_z']))
    df['right_front_limb-back_distance'] = np.sqrt(np.square(df['right_front_limb_x']-df['back_x'])+np.square(df['right_front_limb_y']-df['back_y'])+np.square(df['right_front_limb_z']-df['back_z']))
    df['left_hind_limb-back_distance'] = np.sqrt(np.square(df['left_hind_limb_x']-df['back_x'])+np.square(df['left_hind_limb_y']-df['back_y'])+np.square(df['left_hind_limb_z']-df['back_z']))
    df['right_hind_limb-back_distance'] = np.sqrt(np.square(df['right_hind_limb_x']-df['back_x'])+np.square(df['right_hind_limb_y']-df['back_y'])+np.square(df['right_hind_limb_z']-df['back_z']))
    df['root_tail-back_distance'] = np.sqrt(np.square(df['root_tail_x']-df['back_x'])+np.square(df['root_tail_y']-df['back_y'])+np.square(df['root_tail_z']-df['back_z']))
    
    df['sum_body_distance'] = df[['nose-back_distance','neck-back_distance','left_front_limb-back_distance','right_front_limb-back_distance','left_hind_limb-back_distance','right_hind_limb-back_distance','root_tail-back_distance']].sum(axis=1)
    return(df)

def add_BodyPart_distance2(df):
    count_part = ['nose','neck','back','left_front_limb','right_front_limb','left_hind_limb','right_hind_limb','root_tail']
    count_done =[]
    new_col_num = 0
    for k in count_part:
        count_done.append(k)
        for l in count_part:
            if l in count_done:
                pass
            else:
                new_col_num += 1
                part1_name = k
                part2_name = l
                     
                part1_x = part1_name + '_x'
                part1_y = part1_name + '_y'
                part1_z = part1_name + '_z'
                        
                part2_x = part2_name + '_x'
                part2_y = part2_name + '_y'
                part2_z = part2_name + '_z'
                     

                df['{}-{}_distance'.format(part1_name,part2_name)] = np.sqrt(np.square(df[part1_x]-df[part2_x])+np.square(df[part1_y]-df[part2_y])+np.square(df[part1_z]-df[part2_z]))
                
    df['sum_body_distance'] = df.iloc[:,-new_col_num:-1].sum(axis=1)
    return(df)

ske_data_list = []
FeA_data_list = []
for index in plot_dataset.index:
    video_index = plot_dataset.loc[index,'video_index']
    ExperimentCondition = plot_dataset.loc[index,'ExperimentCondition']
    LightingCondition = plot_dataset.loc[index,'LightingCondition']
    gender = plot_dataset.loc[index,'gender']
    ske_data = pd.read_csv(Skeleton_path[video_index])
    Mov_data = pd.read_csv(Movement_Label_path[video_index],usecols=['revised_movement_label'])
    combine_df = pd.concat([Mov_data,ske_data],axis=1) 
    combine_df['gender'] = gender
    combine_df['group'] = ExperimentCondition + '_' +  LightingCondition
    ske_data_list.append(combine_df)


for index in plot_dataset2.index:
    video_index = plot_dataset2.loc[index,'video_index']
    ExperimentCondition = plot_dataset2.loc[index,'ExperimentCondition']
    LightingCondition = plot_dataset2.loc[index,'LightingCondition']
    gender = plot_dataset2.loc[index,'gender']
    ske_data = pd.read_csv(Skeleton_path[video_index])
    Mov_data = pd.read_csv(Movement_Label_path[video_index],usecols=['revised_movement_label'])
    combine_df = pd.concat([Mov_data,ske_data],axis=1)   
    combine_df['gender'] = gender
    combine_df['group'] = ExperimentCondition + '_' +  LightingCondition
    ske_data_list.append(combine_df)


all_df = pd.concat(ske_data_list,axis=0)
all_df = add_BodyPart_distance(all_df)
 
average_body_stretch_rate = all_df.loc[(all_df['group']=='Morning_Light-on')&(all_df['revised_movement_label']=='sniffing'),'sum_body_distance'].mean()
all_df['body_stretch_rate'] = (all_df['sum_body_distance']/average_body_stretch_rate) * 100
all_df.reset_index(drop=True,inplace=True)


# =============================================================================
# grooming_df_list = []
# for group_id in all_df['group'].unique():
#     group_df = all_df[all_df['group']==group_id]
#     grooming_df = group_df[group_df['revised_movement_label']=='grooming']
#     grooming_df = grooming_df.sample(10000)
#     grooming_df_list.append(grooming_df)
# 
# =============================================================================


df_select_list = []
for group_id in all_df['group'].unique():
    group_df = all_df[all_df['group']==group_id]
    for MV in movement_order:
        df_singleLocomotion = all_df[all_df['revised_movement_label']==MV]
        if len(df_singleLocomotion) < 1000:
            df_frame_select = df_singleLocomotion
        else:
            df_frame_select = df_singleLocomotion.sample(n=1000, random_state=2024) # weights='locomotion_speed_smooth',
        print(MV,len(df_frame_select))
        df_select_list.append(df_frame_select)


df_select = pd.concat(df_select_list,axis=0)
df_select.reset_index(drop=True,inplace=True)
df_select = df_select[['revised_movement_label','group','body_stretch_rate']]

df_select.to_csv('{}/body_stretch_rate.csv'.format(output_dir),index=False)

fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(10,5),dpi=300,sharex=True)
sns.violinplot(df_select,x="revised_movement_label",y="body_stretch_rate",
               hue='group',palette=[dataset1_color,dataset2_color],linewidth = 1,
               linecolor= 'black', width = 0.8,legend=False,)

ax.set_xlabel('Movement')
ax.set_ylabel('body_stretch_rate(%)')
ax.set_title('body_stretch_rate(%)',fontsize=25)
#ax.spines['bottom'].set_linewidth(4)
#ax.spines['left'].set_linewidth(4)
#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
ax.set_ylim(40,150)
ax.set_yticks(range(50,151,50))
ax.tick_params(width=1,length=2,axis='y')
plt.xticks(rotation=60)
#ax.yaxis.set_major_formatter(plt.NullFormatter())
#ax.xaxis.set_major_formatter(plt.NullFormatter())

#plt.savefig('{}/body_stretch_rate.png'.format(output_dir),transparent=True,dpi=300)
# =============================================================================
# sns.boxplot(showmeans=True,
#              meanline=True,
#              meanprops={'color':'black', 'ls': '-', 'lw': 4,'label': '_mean_'},
#              medianprops={'visible': False},
#              whiskerprops={'visible': False},
#              #hue_order=['naive','stress'],
#              #order=exploration,
#              zorder=10,
#              x="revised_movement_label",
#              y="body_stretch_rate",
#              hue='group',
#              data=grooming_df,
#              showfliers=False,
#              showbox=False,
#              showcaps=False,
#              legend=False,
#              palette=[dataset1_color,dataset2_color],
#              linewidth = 3,
#              linecolor= 'black',
#              width = 0.8,
#              ax=ax)
# =============================================================================

























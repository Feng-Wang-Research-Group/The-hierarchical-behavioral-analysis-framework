# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 14:08:46 2023

@author: Jialin Ye
@institution: SIAT
@Contact_email: jl.ye@siat.ac.cn

"""

import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split 
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from math import pi


new_generate_data_day = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure7_Movement_sequence_prediction\predicted_movement_sequences\new_sequence-MOTP_MODP'
new_generate_data_stress = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure7_Movement_sequence_prediction\predicted_movement_sequences\new_sequence-ASTP_ASDP'
InputData_path_dir = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure7_Movement_sequence_prediction\PCA_results'
if not os.path.exists(output_dir):                                                                       
    os.mkdir(output_dir)

def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-'))&(file_name.endswith(content)):
            USN = int(file_name.split('-')[1])
            file_path_dict.setdefault(USN,file_dir+'\\'+file_name)
    return(file_path_dict)

Movement_Label_path = get_path(InputData_path_dir,'revised_Movement_Labels.csv')
Feature_space_path = get_path(InputData_path_dir,'Feature_Space.csv')



def get_path2(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-'))&(file_name.endswith(content)):
            USN = file_name.split('_')[0]
            file_path_dict.setdefault(USN,file_dir+'\\'+file_name)
    return(file_path_dict)


Movement_Label_path_predict_day = get_path2(new_generate_data_day, 'Movement_Labels.csv')
Movement_Label_path_predict_stress = get_path2(new_generate_data_stress,'Movement_Labels.csv')


skip_file_list = [1,3,28,29,110,122] 

animal_info_csv = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Table_S1_animal_information.csv' 
animal_info = pd.read_csv(animal_info_csv)
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)]


Morning_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Morning') & (animal_info['LightingCondition']=='Light-on')]
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-off')]

Stress_info = animal_info[(animal_info['ExperimentCondition']=='Stress')]

movement_order = ['running','trotting','left_turning','right_turning','walking','stepping','sniffing',
                  'rising','hunching','rearing','climbing','jumping','grooming','scratching','pausing',]


def extract_segment(df):
    mv_dict = {}
    for mv in movement_order:
        mv_dict.setdefault(mv,0)
    count_df = pd.DataFrame(mv_dict,index=[0])

    label_count = df.value_counts('revised_movement_label')
    for count_mv in label_count.index:
        count_df[count_mv]  = label_count[count_mv]
    return(count_df)

def get_color(name):
    if name.startswith('Morning'):
        color = '#F5B25E'
    elif name.startswith('Afternoon'):
        color = '#936736'
    elif name.startswith('Night_lightOn'):
        color = '#398FCB'
    elif name.startswith('Night_lightOff'):
        color = '#003960'
    elif name.startswith('Stress'):
        color = '#f55e6f'
    else:
        print(name)
    
    return(color)

dataset1 = Morning_lightOn_info
dataset1_name = 'Morning_lightOn'
dataset1_color = get_color(dataset1_name)


dataset2 = Stress_info
dataset2_name =  'Stress_lightOn'
dataset2_color = get_color(dataset2_name)

all_matrix = []
#for file_name in list(Movement_Label_path.keys())[2:73]:
for index in dataset1.index:
    video_index = dataset1.loc[index,'video_index']
    gender =  dataset1.loc[index,'gender']
    ExperimentCondition =  dataset1.loc[index,'ExperimentCondition']
    LightingCondition = dataset1.loc[index,'LightingCondition']
    MoV_file = pd.read_csv(Movement_Label_path[video_index])
    #location_file = pd.read_csv(Location_path1[video_index],usecols=['location'])
    # MoV_file = pd.concat([MoV_file,location_file],axis=1)
    MoV_matrix = extract_segment(MoV_file)
    MoV_matrix['group_id'] = 0
    MoV_matrix['mouse_info'] = gender+'-' + ExperimentCondition+'-' + LightingCondition
    all_matrix.append(MoV_matrix)

for index in dataset2.index:
    video_index = dataset2.loc[index,'video_index']
    gender =  dataset2.loc[index,'gender']
    ExperimentCondition =  dataset2.loc[index,'ExperimentCondition']
    LightingCondition = dataset2.loc[index,'LightingCondition']
    MoV_file = pd.read_csv(Movement_Label_path[video_index])
    #location_file = pd.read_csv(Location_path2[video_index],usecols=['location'])
    #MoV_file = pd.concat([MoV_file,location_file],axis=1)
    MoV_matrix = extract_segment(MoV_file)
    MoV_matrix['group_id'] = 1
    MoV_matrix['mouse_info'] = gender+'-' + ExperimentCondition + '-' + LightingCondition
    all_matrix.append(MoV_matrix)

for key in Movement_Label_path_predict_day.keys():
    video_index = key
    gender =  'None'
    ExperimentCondition =  dataset1_name.split('_')[0]
    LightingCondition =dataset1_name.split('_')[1]
    MoV_file = pd.read_csv(Movement_Label_path_predict_day[video_index])
    #location_file = pd.read_csv(Location_path2[video_index],usecols=['location'])
    #MoV_file = pd.concat([MoV_file,location_file],axis=1)
    MoV_matrix = extract_segment(MoV_file)
    MoV_matrix['group_id'] = 0
    MoV_matrix['mouse_info'] = 'new_generate_data_day'
    all_matrix.append(MoV_matrix)

for key in Movement_Label_path_predict_stress.keys():
    video_index = key
    gender =  'None'
    ExperimentCondition =  dataset2_name.split('_')[0]
    LightingCondition =dataset2_name.split('_')[1]
    MoV_file = pd.read_csv(Movement_Label_path_predict_stress[video_index])
    #location_file = pd.read_csv(Location_path2[video_index],usecols=['location'])
    #MoV_file = pd.concat([MoV_file,location_file],axis=1)
    MoV_matrix = extract_segment(MoV_file)
    MoV_matrix['group_id'] = 1
    MoV_matrix['mouse_info'] = 'new_generate_data_night'
    all_matrix.append(MoV_matrix)

all_df = pd.concat(all_matrix)
all_df.reset_index(drop=True,inplace=True)


feat_cols = all_df.columns[:-2]
# Separating out the features
X = all_df[feat_cols].values
# Separating out the target
y = all_df['group_id']
# Standardizing the features
X = StandardScaler().fit_transform(X)


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
PC1 = principalComponents[:,0]
PC2 = principalComponents[:,1]
#PC3 = principalComponents[:,2]


svm = SVC(kernel='linear')
svm.fit(principalComponents, y)

w = svm.coef_[0]                       
a = -w[0]/w[1]                         
x = np.linspace(-6,6)                  
y = a * x -(svm.intercept_[0])/w[1]    

b = svm.support_vectors_[0]        
y_down = a * x + (b[1] - a*b[0])   
b = svm.support_vectors_[-1]       
y_up = a * x + (b[1] - a*b[0])


fig,ax= plt.subplots(nrows=1,ncols=1,figsize=(10,10),dpi=300)

plt.plot(x, y, 'k-', label='SVM Decision Boundary')

plt.plot(x,y_down,'k--')   
plt.plot(x,y_up,'k--')     


cross_vertices = [(-1,-2), # left
             (1,-2), # right
             (0,-1), #top
             (0,-3),] #bottom
 
cross_codes = [mpath.Path.MOVETO,mpath.Path.LINETO,mpath.Path.MOVETO,mpath.Path.LINETO]
cross_path = mpath.Path(cross_vertices, cross_codes)
 
circle = mpath.Path.unit_circle() 
verts = np.concatenate([circle.vertices, cross_path.vertices[::-1, ...]])
codes = np.concatenate([circle.codes, cross_path.codes])
female_marker = mpath.Path(verts, codes)

arrow_vertices = [(0.6,0.6), #end
                   (2.85,2.85), #tip
                   (1.2,2), #top
                   (2.85,2.85),
                   (2,1.2),] #bottom
arrow_codes = [mpath.Path.MOVETO,mpath.Path.LINETO,mpath.Path.LINETO,mpath.Path.MOVETO,mpath.Path.LINETO]
arrow_path = mpath.Path(arrow_vertices, arrow_codes)
verts2 = np.concatenate([circle.vertices, arrow_path.vertices[::-1, ...]])
codes2 = np.concatenate([circle.codes, arrow_path.codes])
male_marker = mpath.Path(verts2, codes2,closed=True)


principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, all_df[['mouse_info']]], axis = 1)
targets = all_df['mouse_info'].unique()

colors = [dataset1_color, dataset1_color, dataset2_color,dataset2_color,'#FFFF00','#b71726']  ### stimulated Naive '#FFFF00' & stimulated stress '#b71726',  


shapes = [female_marker,male_marker,male_marker,female_marker,'^','^']
for target, color,shape in zip(targets,colors,shapes):
     indicesToKeep = finalDf['mouse_info'] == target
     if target.startswith('new_generate_data_day'):
         ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                    , finalDf.loc[indicesToKeep, 'principal component 2']
                    , c = color
                    , ec = 'black'
                    , s = 200
                    , lw = 2
                    ,marker=shape
                    ,alpha=0.8)
     elif target.startswith('new_generate_data_night'):
         ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                    , finalDf.loc[indicesToKeep, 'principal component 2']
                    , c = color
                    , ec = 'black'
                    , s = 200
                    , lw = 2
                    ,marker=shape
                    ,alpha=0.8)    
     else:
         ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                    , finalDf.loc[indicesToKeep, 'principal component 2']
                    , c = color
                    , ec = 'black'
                    , s = 1500
                    , lw = 2
                    ,marker=shape
                    ,alpha=1)


ax.spines['bottom'].set_linewidth(5)
ax.spines['left'].set_linewidth(5)
ax.spines['top'].set_linewidth(5)
ax.spines['right'].set_linewidth(5)

plt.xlim(-8,8)
plt.ylim(-8,8)
plt.xticks([])
plt.yticks([])
plt.savefig('{}/MOTP_MODP&ASTP_ASDP.png'.format(output_dir),dpi=300)
# Show the plot
plt.show()


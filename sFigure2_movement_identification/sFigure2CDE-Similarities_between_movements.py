# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:12:08 2023

@author: Jialin Ye
@institution: SIAT
@Contact_email: jl.ye@siat.ac.cn

"""

import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist,squareform
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import umap
from matplotlib.colors import ListedColormap
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score, f1_score
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import StratifiedKFold

orginFeA_path_csv_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\01_BehaviorAtlas_collated_data'
anno_Mov_csv_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\01_BehaviorAtlas_collated_data'
output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\sFigure2_movement_identification'


def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-')) & (file_name.endswith(content)):
            USN = int(file_name.split('-')[1])
            file_path_dict.setdefault(USN,file_dir+'/'+file_name)
    return(file_path_dict)

orginFeA_path_dict = get_path(orginFeA_path_csv_dir,'Feature_Space.csv')
FeA_path_dict = get_path(anno_Mov_csv_dir,'Movement_core_features.csv')



skip_file_list = [1,3,28,29,110,122]
animal_info_csv = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Table_S1_animal_information.csv'              
animal_info = pd.read_csv(animal_info_csv)
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)]

lightOn_info = animal_info[animal_info['LightingCondition']=='Light-on']


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

calculated_dataset = lightOn_info


def add_movement_label_LightON(df):
    movement_dict = {'running':[29,28,13],                     # > 250 mm/s
                     'trotting':[14,22],                       # > 200 mm/s
                     'walking':[23,33,26,19],             # > 80 mm/s
                     'right_turning':[18,17,2,1],              # > 
                     'left_turning':[27,12],
                     'stepping':[9],                           # >50mm/s
                     'climbing':[31,32,25],	
                     'rearing':[16],
                     'hunching':[24],
                     'rising':[5,8,34],
                     'grooming':[37,40,15],
                     'sniffing':[10,11,30,35,36,6,38,4,3],
                     'pausing':[39,20,21],
                     'jumping':[7],
                     }
    df_copy = df.copy()
    for mv in movement_dict.keys():
        df_copy.loc[df_copy['OriginalDigital_label'].isin(movement_dict[mv]),'revised_movement_label'] = mv
    return(df_copy)

def annoFeA(FeA_data,annoMV_data):
    start = 0
    end = 0
    for i in FeA_data.index:
        end = FeA_data.loc[i,'segBoundary']
        single_annoMV_data = annoMV_data.loc[start:end,'revised_movement_label'].value_counts()
        FeA_new_label = single_annoMV_data.index[0]
        start = end
        FeA_data.loc[i,'revised_movement_label'] = FeA_new_label
    return(FeA_data)



def supplyFeA():   
    start = 0
    for i in FeA_data.index:
        end = FeA_data.loc[i,'segBoundary']
        FeA_data.loc[i,'segBoundary_end'] = end
        FeA_data.loc[i,'segBoundary_start'] = start
        FeA_data


movement_order = ['running','trotting','walking','right_turning','left_turning','stepping',
                  'rising','hunching','rearing','climbing','jumping','sniffing','grooming','scratching','pausing',]

info_dict = {'revised_movement_label':[],'intra_coff_mean':[],'inter_coff_mean':[]}



movement_dict = {'running':[29,28,13],                     # > 250 mm/s
                 'trotting':[14,22],                       # > 200 mm/s
                 'walking':[23,33,26,19],                  # > 80 mm/s
                 'right_turning':[18,17,2,1],              # > 
                 'left_turning':[27,12],
                 'stepping':[9],                           # >50mm/s
                 'climbing':[31,32,25],	
                 'rearing':[16],
                 'hunching':[24],
                 'rising':[5,8,34],
                 'grooming':[37,40,15],
                 'sniffing':[10,11,30,35,36,6,38,4,3],
                 'pausing':[39,20,21],
                 'jumping':[7],
                 'scratching':[],
                 }


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

cmap1 = plt.get_cmap('tab20b_r')
cmap2 = plt.get_cmap('tab20c_r')
new_cmap = ListedColormap(cmap2.colors+cmap1.colors)


single_FeA_data = []
for video_index in list(calculated_dataset['video_index']):
    FeA_data = pd.read_csv(FeA_path_dict[video_index],
                           usecols=['revised_movement_label',
                                    'segBoundary_start',
                                    'segBoundary_end',
                                    'frame_length',
                                    'average_speed',
                                    'average_back_height',
                                    'spine_angle_NBT',
                                    'angle_3d',
                                    'nose_momentum',
                                    'sum_body_distance',
                                    'average_nose_height',
                                    'average_front_claw_height',
                                    'average_hind_claw_height',
                                    'average_nose_tail_distance',
                                    'body_stretch_rate'])
    single_FeA_data.append(FeA_data)
    
all_FeA_data = pd.concat(single_FeA_data)    
all_FeA_data.reset_index(drop=True,inplace=True)

dict_weights = {}
mv_FeA_list = []
for mv in movement_order:
    mv_FeA = all_FeA_data[all_FeA_data['revised_movement_label']==mv]
    if len(mv_FeA) > 500:
        mv_FeA =  mv_FeA.sample(500,replace=False)
    print(mv,len(mv_FeA))
    
    dict_weights.setdefault(mv,len(mv_FeA))
    
    mv_FeA_list.append(mv_FeA)

select_FeA = pd.concat(mv_FeA_list,axis=0)
select_FeA.reset_index(drop=True,inplace=True)


matrix = []
movement_label_list = []  
for mv in movement_order:
    mv_df = select_FeA[select_FeA['revised_movement_label']==mv]
    mv_df.reset_index(drop=True,inplace=True)
    for i in mv_df.index:
        arr = [mv_df.loc[i,'average_speed'],mv_df.loc[i,'average_back_height'],mv_df.loc[i,'average_nose_height'],mv_df.loc[i,'nose_momentum'],mv_df.loc[i,'spine_angle_NBT'],mv_df.loc[i,'angle_3d'],
               mv_df.loc[i,'body_stretch_rate'],mv_df.loc[i,'average_nose_tail_distance'],mv_df.loc[i,'average_hind_claw_height'],mv_df.loc[i,'average_front_claw_height']] #'body_stretch_rate'
        matrix.append(arr)
        movement_label_list.append(mv)
matrix = np.array(matrix)       


umap_model = umap.UMAP(n_neighbors=5, min_dist=0.1, n_components=2,random_state=2024)

# low-dimensional embedding

umap_result = umap_model.fit_transform(matrix)

umap_result_df = pd.DataFrame(data=umap_result,columns=['umap1','umap2'],index=range(len(movement_label_list)))
umap_result_df['movement_label'] = movement_label_list
umap_result_df['color'] = umap_result_df['movement_label'].map(movement_color_dict)

fig, ax = plt.subplots(figsize=(10,10),dpi=300)
for i in umap_result_df.index:
    x = umap_result_df.loc[i,'umap1']
    y = umap_result_df.loc[i,'umap2']
    color = umap_result_df.loc[i,'color']
    ax.scatter(x,y,c=color,s=5)
plt.savefig(r'{}/low-dimensional_embedding_of_movement_features.png'.format(output_dir),dpi=300,bbox_inches='tight')
plt.xlabel('UMAP1',fontsize=10,fontfamily='arial')
plt.ylabel('UMAP2',fontsize=10,fontfamily='arial')
fig, ax = plt.subplots(figsize=(10,10),dpi=300)
distance_matrix = squareform(pdist(umap_result))
plt.imshow(distance_matrix)
plt.savefig(r'{}/similarity_heatmap_of_movement_features.png'.format(output_dir),dpi=300,bbox_inches='tight')



###### training a random forest classifier 

feat_cols = ['average_speed','average_back_height','average_nose_height','nose_momentum','spine_angle_NBT','angle_3d','body_stretch_rate','average_nose_tail_distance','average_hind_claw_height','average_front_claw_height']
X = select_FeA[feat_cols].values
y = select_FeA[['revised_movement_label']].values

X = StandardScaler().fit_transform(X)

skf = StratifiedKFold(n_splits=10,random_state=2024, shuffle=True)

accumulate_matrix = np.zeros(shape=(15,15))
lst_accu_stratified = []
lst_F1_stratified = []
importances_feature_list = []
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    clf = RandomForestClassifier(n_estimators=100, random_state=52,bootstrap=True,criterion='gini',oob_score=True,class_weight='balanced',max_depth=100,min_samples_split=2)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    accuracy = clf.score(X_test, y_test)
    F1_score = f1_score(y_test, y_pred,average='macro')
    
    # calculate accuracy, recall, F1 score  and support 
    report = classification_report(y_test, y_pred, target_names=movement_order)
    
    cm_matrix = confusion_matrix(y_test, y_pred)
    accumulate_matrix += cm_matrix
    lst_accu_stratified.append(clf.score(X_test, y_test))
    lst_F1_stratified.append(f1_score(y_test, y_pred,average='macro'))
    
    importances_feature = clf.feature_importances_
    importances_df = pd.DataFrame(data=importances_feature,columns=['importances'],index=feat_cols)
    importances_df['feature'] = importances_df.index
    importances_feature_list.append(importances_df)

all_importances_df = pd.concat(importances_feature_list,axis=0)
all_importances_df.reset_index(drop=True,inplace=True)
all_importances_df.sort_values(by=['importances'],ascending=False,inplace=True)

accumulate_importances_df = pd.DataFrame()
accumulate_importances = 0
num = 0
for feature in all_importances_df['feature'].unique():
    accumulate_importances += all_importances_df[all_importances_df['feature']==feature]['importances'].mean()
    accumulate_importances_df.loc[num,'accumulative_importances'] = accumulate_importances
    accumulate_importances_df.loc[num,'feature'] = feature
    num+=1


############################ feature importance ###############################
fig, ax = plt.subplots(figsize=(10,8),dpi=600)
sns.barplot(data=all_importances_df,x='feature',y='importances',color='#1565C0')
#sns.lineplot(data=accumulate_importances_df,x='feature',y='accumulative_importances')
sns.stripplot(data=all_importances_df,x='feature',y='importances',jitter=0.25,size=10,edgecolor='black',linewidth=2,facecolor='white',alpha=0.8)

ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
#ax.set_ylabel('')
#ax.set_xlabel('')
ax.set_yticks(np.arange(0,0.25,0.08))
ax.tick_params(length=4,width=2)
plt.xticks(rotation=60)
#ax.axes.xaxis.set_ticklabels([])
#ax.axes.yaxis.set_ticklabels([])
plt.savefig(r'{}/RandomForest_Features_Importances.png'.format(output_dir),dpi=300,bbox_inches='tight')


############################## confusion matrix###############################
print('average_accuracy:{}'.format(np.mean(lst_accu_stratified)))
print('average_F1_score:{}'.format(np.mean(lst_F1_stratified)))

class_names=movement_order # name  of classes
fig, ax = plt.subplots(figsize=(10,10),dpi=600)
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
#plt.text(x=0.2,y=-.1,s='Random forest prediction, Accuracy:{:.2f}'.format(accuracy_score(y_test, y_pred)),fontsize=26)
# create heatmap
sns.heatmap(pd.DataFrame(accumulate_matrix),annot=True, cmap=plt.cm.get_cmap('Blues', 10),cbar=False,annot_kws={"fontsize":18,"fontfamily":'arial',"fontweight":'bold'},linecolor='black',linewidths=2,vmin=0,vmax=500,
            xticklabels=True,yticklabels=True, fmt='.0f',)
ax.set_xticklabels(movement_order,fontsize=20,rotation=90)
ax.set_yticklabels(movement_order,fontsize=20,rotation=0)
ax.xaxis.set_label_position("top")
plt.tight_layout()
#plt.axis('off')
plt.title('Confusion matrix', y=1.1,fontsize=40)
plt.ylabel('Actual label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)

print('Accuracy:',accuracy_score(y_test, y_pred))
plt.savefig(r'{}/RandomForest_Confusion_Matrix.png'.format(output_dir),dpi=300,bbox_inches='tight')
plt.show()


    
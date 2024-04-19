# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 23:30:20 2023

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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score, f1_score
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import StratifiedKFold

InputData_path_dir = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label"
InputData_path_dir2 = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\01_BehaviorAtlas_collated_data"
output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure1&Figure3_behavioral_characteristics(related to sFigure3-5)\group_dissimilarities\singleFactor'

skip_file_list = [1,3,28,29,110,122]
def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-'))&(file_name.endswith(content)):
            USN = int(file_name.split('-')[1])
            #date = i.split('-')[3][0:8]
            #file_name = 'rec-{0}-G1-{1}'.format(USN,date)
            file_path_dict.setdefault(USN,file_dir+'\\'+file_name)
    return(file_path_dict)

Movement_Label_path = get_path(InputData_path_dir,'revised_Movement_Labels.csv')
Feature_space_path = get_path(InputData_path_dir,'Feature_Space.csv')
Location_path = get_path(InputData_path_dir2,'coordinates_back_XY.csv')

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
    #for index in range(30*seg_size,df.shape[0]+5,30*seg_size):
    mv_dict = {}
    for mv in movement_order:
        mv_dict.setdefault(mv,0)
    count_df = pd.DataFrame(mv_dict,index=[0])

    label_count = df.value_counts('revised_movement_label')
    for count_mv in label_count.index:
        count_df[count_mv]  = label_count[count_mv]
    return(count_df)

group_id = {'Morning-Light-on':0,'Afternoon-Light-on':1,'Night-Light-on':2,'Night-Light-off':3,'Stress-Light-on':4}
group_order = ['Morning-Light-on','Afternoon-Light-on','Night-Light-on','Night-Light-off','Stress-Light-on']    


using_factor = 'speed'    #example: speed, position, index, movement_fraction


def cal_center_rate(df):
    df_copy = df.copy()
    df_copy['back_x'] = df_copy['back_x'] - 250
    df_copy['back_y'] = df_copy['back_y'] - 250
    df_copy['center_rate'] =  np.sqrt(np.square(df_copy['back_x']-0) + np.square(df_copy['back_y']-0))    
    return(df_copy)

all_matrix = []
#for file_name in list(Movement_Label_path.keys())[2:73]:
for index in animal_info.index:
    video_index = animal_info.loc[index,'video_index']
    gender =  animal_info.loc[index,'gender']
    ExperimentCondition =  animal_info.loc[index,'ExperimentCondition']
    LightingCondition = animal_info.loc[index,'LightingCondition']
    MoV_file = pd.read_csv(Movement_Label_path[video_index])
    location_file = pd.read_csv(Location_path[video_index])
    center_rate = cal_center_rate(location_file)
    #MoV_file = pd.concat([MoV_file,location_file],axis=1)
    MoV_matrix = extract_segment(MoV_file)
    MoV_matrix['sniffing_grooming_fra'] = MoV_matrix['sniffing'].values[0]/MoV_matrix['grooming'].values[0]
    MoV_matrix['average_speed'] = MoV_file['smooth_speed'].mean()
    MoV_matrix['average_center_rate'] = center_rate['center_rate'].mean()
    
    if using_factor == 'speed':     
        MoV_matrix = MoV_matrix[['average_speed']]                           ### substitute with other para
    elif using_factor == 'position':     
        MoV_matrix = MoV_matrix[['average_center_rate']]
    elif using_factor == 'index':     
        MoV_matrix = MoV_matrix[['sniffing_grooming_fra']]
    elif using_factor == 'movement_fraction':     
        MoV_matrix = extract_segment(MoV_file)
    else:
        print('please check if the using_factor is right')
    
    MoV_matrix['group'] = ExperimentCondition+'-' + LightingCondition
    MoV_matrix['group_id'] = MoV_matrix['group'].map(group_id)
    all_matrix.append(MoV_matrix)


all_df = pd.concat(all_matrix)
all_df = all_df.sort_values(by=['group_id'])
all_df.reset_index(drop=True,inplace=True)

feat_cols = all_df.columns[:-2]
X = all_df[feat_cols].values
y = all_df['group_id']
X = StandardScaler().fit_transform(X)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 6)
skf = StratifiedKFold(n_splits=10,random_state=2020, shuffle=True)

accumulate_matrix = np.zeros(shape=(5,5))
lst_accu_stratified = []
lst_F1_stratified = []
importances_feature_list = []
for train_index, test_index in skf.split(X, y):
    #print('TRAIN:', train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = RandomForestClassifier(n_estimators=100, random_state=52,bootstrap=True,criterion='gini',class_weight='balanced',max_depth=100,min_samples_split=2)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    accuracy = clf.score(X_test, y_test)
    F1_score = f1_score(y_test, y_pred,average='macro')
   
    cm_matrix = confusion_matrix(y_test, y_pred)
    accumulate_matrix += cm_matrix
    lst_accu_stratified.append(clf.score(X_test, y_test))
    lst_F1_stratified.append(f1_score(y_test, y_pred,average='macro'))
    

print('average_accuracy:{}'.format(np.mean(lst_accu_stratified)))
print('average_F1_score:{}'.format(np.mean(lst_F1_stratified)))

class_names=[0,1,2,3,4] # name  of classes
fig, ax = plt.subplots(figsize=(10,10),dpi=300)
tick_marks = np.arange(len(class_names))

#plt.text(x=0.2,y=-.1,s='Random forest prediction, Accuracy:{:.2f}'.format(accuracy_score(y_test, y_pred)),fontsize=26)
# create heatmap
sns.heatmap(pd.DataFrame(accumulate_matrix),annot=True, cmap="Greys",cbar=False,annot_kws={"fontsize":50,"fontfamily":'arial',"fontweight":'bold'},vmin=0,vmax=32,linewidths=2,linecolor='black')
ax.xaxis.set_label_position("top")
plt.xticks(tick_marks+0.5, group_order,fontsize=20,rotation=60)
plt.yticks(tick_marks+0.5, group_order,fontsize=20,rotation=60)
plt.tight_layout()
plt.title('Confusion matrix ({})'.format(using_factor), y=1.1,fontsize=25)
plt.ylabel('Actual label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20,ha='center', va='bottom')
plt.text(0,7.2,'average_accuracy:{:.2f}'.format(np.mean(lst_accu_stratified)),fontsize=20)
plt.text(0,7.5,'average_F1_score:{}'.format(np.mean(lst_F1_stratified)),fontsize=20)

plt.savefig('{}/RandomForest_prediction_using_{}.png'.format(output_dir,using_factor),dpi=300,transparent=True)
plt.show()
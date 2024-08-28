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
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,f1_score
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import StratifiedKFold

InputData_path_dir = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label"                   
InputData_path_dir_predict_MO = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure7_Movement_sequence_prediction\predicted_movement_sequences\new_sequence-MOTP_MODP'                    
InputData_path_dir_predict_AS = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure7_Movement_sequence_prediction\predicted_movement_sequences\new_sequence-ASTP_ASDP'

output_dir = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure7_Movement_sequence_prediction\sFigure22_Comparison_of_simulated_sequences" 
skip_file_list = [1,3,28,29,110,122]

def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-'))&(file_name.endswith(content)):
            USN = int(file_name.split('-')[1])
            file_path_dict.setdefault(USN,file_dir+'\\'+file_name)
    return(file_path_dict)

def get_path2(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-'))&(file_name.endswith(content)):
            USN = file_name.split('-')[1]
            file_path_dict.setdefault(USN,file_dir+'\\'+file_name)
    return(file_path_dict)

Movement_Label_path = get_path(InputData_path_dir,'revised_Movement_Labels.csv')

Movement_Label_path_predict_MO = get_path2(InputData_path_dir_predict_MO,'Movement_Labels.csv')
Movement_Label_path_predict_AS = get_path2(InputData_path_dir_predict_AS,'Movement_Labels.csv')

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




dataset1 = Morning_lightOn_info
dataset2 = Stress_info

dataset_name1 = 'Morning_lightOn_info'
dataset_name2 =  'Stress'


dataset3 = Movement_Label_path_predict_MO
dataset4 = Movement_Label_path_predict_AS

dataset_name3 = 'predict_Morning'
dataset_name4 = 'predict_AS'


training_dataset = []

dateset1_actual_number =0
all_matrix = []

for index in dataset1.index:
    video_index = dataset1.loc[index,'video_index']
    if video_index in skip_file_list:
        pass
    else:
        dateset1_actual_number += 1
        gender =  dataset1.loc[index,'gender']
        ExperimentCondition =  dataset1.loc[index,'ExperimentCondition']
        LightingCondition = dataset1.loc[index,'LightingCondition']
        MoV_file = pd.read_csv(Movement_Label_path[video_index])
        MoV_matrix = extract_segment(MoV_file)
        MoV_matrix['group_id'] = 0
        MoV_matrix['mouse_info'] = gender+'-' + ExperimentCondition+'-' + LightingCondition
        all_matrix.append(MoV_matrix)
        training_dataset.append(MoV_matrix)

dateset3_actual_number =0
for video_index in dataset3.keys():
    MoV_file = pd.read_csv(Movement_Label_path_predict_MO[video_index])

    dateset3_actual_number += 1
    gender =  'unknown'
    ExperimentCondition =  'unknown'
    LightingCondition = 'unknown'
    MoV_matrix = extract_segment(MoV_file)
    MoV_matrix['group_id'] = 1
    MoV_matrix['mouse_info'] ='new_generate_day'
    all_matrix.append(MoV_matrix)


dateset2_actual_number = 0
for index in dataset2.index:
    video_index = dataset2.loc[index,'video_index']
    if video_index in skip_file_list:
        pass
    else:
        dateset2_actual_number += 1
        gender =  dataset2.loc[index,'gender']
        ExperimentCondition =  dataset2.loc[index,'ExperimentCondition']
        LightingCondition = dataset2.loc[index,'LightingCondition']
        MoV_file = pd.read_csv(Movement_Label_path[video_index])
        MoV_matrix = extract_segment(MoV_file)
        MoV_matrix['group_id'] = 2
        MoV_matrix['mouse_info'] = gender+'-' + ExperimentCondition + '-' + LightingCondition
        all_matrix.append(MoV_matrix)
        training_dataset.append(MoV_matrix)


dateset4_actual_number =0

for video_index in dataset4.keys():
    MoV_file = pd.read_csv(Movement_Label_path_predict_AS[video_index])

    dateset4_actual_number += 1
    gender =  'unknown'
    ExperimentCondition =  'unknown'
    LightingCondition = 'unknown'
    MoV_matrix = extract_segment(MoV_file)
    MoV_matrix['group_id'] = 3
    MoV_matrix['mouse_info'] ='new_generate_AS'
    all_matrix.append(MoV_matrix)



print('dateset1_actual_number:',dateset1_actual_number)
print('dateset2_actual_number:',dateset2_actual_number)

all_df = pd.concat(all_matrix)
all_df.reset_index(drop=True,inplace=True)


traning_df = pd.concat(training_dataset)
traning_df.reset_index(drop=True,inplace=True)

feat_cols = all_df.columns[:-2]
# Separating out the features
X = all_df[feat_cols].values
# Separating out the targetl
y = all_df['group_id']
# Standardizing the features
X = StandardScaler().fit_transform(X)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 6)
skf = StratifiedKFold(n_splits=10,random_state=2020, shuffle=True)
#print(skf)

accumulate_matrix = np.zeros(shape=(4,4))
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
    print(cm_matrix)
    accumulate_matrix += cm_matrix
    lst_accu_stratified.append(accuracy)
    lst_F1_stratified.append(F1_score)


print('average_accuracy:{}'.format(np.mean(lst_accu_stratified)))
print('average_F1_score:{}'.format(np.mean(lst_F1_stratified)))
print(accumulate_matrix)


class_names=[0,1] # name  of classes
fig, ax = plt.subplots(figsize=(10,10),dpi=600)
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
plt.text(x=0.2,y=-.3,s='Random forest prediction, Accuracy:{:.2f}'.format(accuracy_score(y_test, y_pred)),fontsize=20)
# create heatmap
sns.heatmap(pd.DataFrame(accumulate_matrix),annot=True, cmap="copper",cbar=False,annot_kws={"fontsize":30})
ax.set_xticklabels([dataset_name1,dataset_name3,dataset_name2,dataset_name4],fontsize=15,rotation=30)
ax.set_yticklabels([dataset_name1,dataset_name3,dataset_name2,dataset_name4],fontsize=15,rotation=30)
ax.xaxis.set_label_position("top")
plt.tight_layout()
#plt.axis('off')
plt.title('Confusion matrix', y=1.1,fontsize=40)
plt.ylabel('Actual label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)

plt.savefig('{}/{}&{}_RandomForest_prediction.png'.format(output_dir,dataset_name1,dataset_name2),dpi=600)

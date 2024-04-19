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
from scipy.cluster.hierarchy import linkage, dendrogram,optimal_leaf_ordering,leaves_list
from scipy.stats import pearsonr
import random
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score, f1_score,roc_curve, auc
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit, GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
import umap
from matplotlib.colors import ListedColormap
from scipy.cluster import hierarchy





orginFeA_path_csv_dir = r'F:\spontaneous_behavior\04返修阶段\01_BehaviorAtlas_collated_data'
anno_Mov_csv_dir = r'F:\spontaneous_behavior\04返修阶段\03_movement_feature'

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
animal_info_csv = r'F:\spontaneous_behavior\04返修阶段\Table_S1_animal_information_clean.csv'              
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

#select_for_plot = [29,28,13,14,22,23,33,26,19,18,17,27,12,10,31,32,25,16,24,34,37,40,15,3,20,21,7]
#select_for_plot = range(40)
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
                 'scratching':[],
                 }


movement_color_dict = {'running':'#FF3030',
                       'trotting':'#E15E8A',                       
                       'left_turning':'#F6BBC6', 
                       'right_turning':'#F8C8BA',
                       'walking':'#EB6148',
                       'stepping':'#C6823F',  
                       'sniffing':'#2E8BBE',
                       'rising':'#84CDD9',    #'#FFEA00'  ####FFEE58
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
    FeA_data = pd.read_csv(orginFeA_path_dict[video_index])
    FeA_data = add_movement_label_LightON(FeA_data)
    single_FeA_data.append(FeA_data)
all_FeA_data = pd.concat(single_FeA_data)    
all_FeA_data.reset_index(drop=True,inplace=True)

#all_FeA_data['color'] = all_FeA_data['OriginalDigital_label'].map(new_cmap)

#row_c = dict(zip([str(x) for x in range(0,40)], new_cmap.colors))

# =============================================================================
# #select_FeA = all_FeA_data
# dict_weights = {}
# mv_FeA_list = []
# for mv in movement_order:
#     print(mv)
#     for num_label in movement_dict[mv]:
#         mv_FeA = all_FeA_data[all_FeA_data['OriginalDigital_label']==num_label]
#         if len(mv_FeA) > 500:
#             mv_FeA =  mv_FeA.sample(500,replace=False)
#         print(num_label,len(mv_FeA))
#     
#         dict_weights.setdefault(mv,len(mv_FeA))
#     
#         mv_FeA_list.append(mv_FeA)
# 
# select_FeA = pd.concat(mv_FeA_list,axis=0)
# select_FeA.reset_index(drop=True,inplace=True)
# 
# =============================================================================

# =============================================================================
# Feature_Space_matrix = np.array(select_FeA[['umap1','umap2','zs_velocity']].values)
# distance_matrix = squareform(pdist(Feature_Space_matrix,metric='seuclidean'))
# distance_matrix = 1 - distance_matrix
# 
# fig, axs = plt.subplots(ncols=1,nrows=1,dpi=300)
# plt.imshow(distance_matrix,cmap='RdBu_r')
# 
# def get_index(lst=None, item=''):
#     return [index for (index,value) in enumerate(lst) if value == item]
# 
# intra_coffCell = []
# inter_coffCell = []
# 
# xcoor_dict = {}
# 
# 
# #distance_matrix = (distance_matrix - distance_matrix.min())/(distance_matrix.max()-distance_matrix.min())
# for i in range(1,41):
#     idx = get_index(select_FeA['OriginalDigital_label'].values.tolist(),i)
#     intra_idx = idx
#     inter_idx = [i for i in select_FeA.index if i not in idx]
#     
#     if len(intra_idx) %2 == 0:
#         pass
#     else:
#         intra_idx.pop(-1)
#     
#     intra_sample1 = random.sample(intra_idx, int(len(intra_idx)/2))
#     intra_sample2 = [i for i in intra_idx if i not in intra_sample1]
#     inter_sample3 = random.sample(inter_idx, len(intra_sample2))
#      
#     tem_nIntra = len(intra_sample2)
#     tem_nInter = len(inter_sample3)
#     
#     #sub_distMatIntra = distance_matrix[intra_idx, :]
#     #sub_distMatInter = distance_matrix[inter_idx, :]
#     
#     intra_coff = np.zeros(tem_nIntra)
#     inter_coff = np.zeros(tem_nIntra)
#     
#     
#     for j in range(tem_nIntra):
#         idx1 = intra_sample1[j]
#         xx = distance_matrix[idx1, :]
#         tem_all_coffIntra = np.zeros(tem_nIntra)
#         for k in range(tem_nIntra):
#             idx2 = intra_sample2[k]
#             yy = distance_matrix[idx2, :]
#             tem_coff = pearsonr(xx, yy)
#             tem_all_coffIntra[k] = tem_coff[0]
#         intra_coff[j] = np.mean(tem_all_coffIntra)
# 
#         tem_all_coffInter = np.zeros(tem_nInter)
#         for k in range(tem_nInter):
#             idx3 = inter_sample3[k]
#             zz = distance_matrix[idx3, :]
#             tem_coff = pearsonr(xx, zz)
#             tem_all_coffInter[k] = tem_coff[0]
#         inter_coff[j] = np.mean(tem_all_coffInter)
#     
#     intra_name = '{}_intra_similarity'.format(i)
#     inter_name = '{}_inter_similarity'.format(i)
# 
#     xcoor_dict.setdefault(intra_name,intra_coff)
#     xcoor_dict.setdefault(inter_name,inter_coff)
# 
# df_xcoor = pd.DataFrame.from_dict(xcoor_dict,orient='index')
# #df_xcoor.to_csv()
# 
# cmap1 = plt.get_cmap('Paired')
# 
# new_camp1 = []
# for i in range(12):
#     cmap1_sub = sns.light_palette(cmap1.colors[i],6)[2:]
#     new_camp1.extend(cmap1_sub)
# new_camp1 = ListedColormap(new_camp1)
# 
# cmap3 = sns.light_palette('grey',6)[2]
# 
# c_list = []
# for i in range(40):
#     color_c1 = new_camp1.colors[i]
#     color_g = cmap3    
#     c_list.append(color_c1)
#     c_list.append(color_g)
# 
# new_cmap = ListedColormap(c_list)
# 
# fig, ax = plt.subplots(figsize=(10,4),dpi=600)
# #sns.violinplot(df_xcoor.T,palette=new_cmap.colors,width=0.8,linewidth=1,inner='quart',saturation=1,
# #               inner_kws={'linewidth':1,})
# sns.boxplot(df_xcoor.T, showfliers=False,palette=new_cmap.colors,linecolor='black',linewidth=1)
# plt.ylim(-0.5,1.1)
# =============================================================================



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

# 将高维数据映射到低维空间
umap_result = umap_model.fit_transform(matrix)

umap_result_df = pd.DataFrame(data=umap_result,columns=['umap1','umap2'],index=range(len(movement_label_list)))
umap_result_df['movement_label'] = movement_label_list

umap_result_df['color'] = umap_result_df['movement_label'].map(movement_color_dict)

fig, ax = plt.subplots(figsize=(10,6),dpi=600)
for i in umap_result_df.index:
    x = umap_result_df.loc[i,'umap1']
    y = umap_result_df.loc[i,'umap2']
    color = umap_result_df.loc[i,'color']
    ax.scatter(x,y,c=color,s=5)
#sns.scatterplot(data=umap_result_df,x='umap1',y='umap2',palette=umap_result_df['color'])


distance_matrix = squareform(pdist(umap_result))
plt.imshow(distance_matrix)

#sns.clustermap(distance_matrix,col_cluster=False,row_cluster=False)   

#select_FeA['movement_id'] = pd.factorize(select_FeA['revised_movement_label'])[0]


feat_cols = ['average_speed','average_back_height','average_nose_height','nose_momentum','spine_angle_NBT','angle_3d','body_stretch_rate','average_nose_tail_distance','average_hind_claw_height','average_front_claw_height']

X = select_FeA[feat_cols].values
#y = select_FeA[['movement_id']].values

y = select_FeA[['revised_movement_label']].values

#X = StandardScaler().fit_transform(X)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 6)
skf = StratifiedKFold(n_splits=10,random_state=2024, shuffle=True)

accumulate_matrix = np.zeros(shape=(15,15))
lst_accu_stratified = []
lst_F1_stratified = []
importances_feature_list = []
for train_index, test_index in skf.split(X, y):
    #print('TRAIN:', train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    clf = RandomForestClassifier(n_estimators=100, random_state=52,bootstrap=True,criterion='gini',oob_score=True,class_weight='balanced',max_depth=100,min_samples_split=2)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    accuracy = clf.score(X_test, y_test)
    F1_score = f1_score(y_test, y_pred,average='macro')
    
    # 计算精度、召回率、F1分数和支持度
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

fig, ax = plt.subplots(figsize=(10,8),dpi=600)
sns.barplot(data=all_importances_df,x='feature',y='importances',color='#1565C0')
#sns.lineplot(data=accumulate_importances_df,x='feature',y='accumulative_importances')
sns.stripplot(data=all_importances_df,x='feature',y='importances',jitter=0.25,size=10,edgecolor='black',linewidth=2,facecolor='white',alpha=0.8)

ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)

ax.set_ylabel('')
ax.set_xlabel('')

ax.set_yticks(np.arange(0,0.25,0.08))

ax.tick_params(length=4,width=2)
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])

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
            xticklabels=False,yticklabels=False, fmt='.0f',)
#ax.set_xticklabels([dataset_name1,dataset_name2],fontsize=20)
#ax.set_yticklabels([dataset_name1,dataset_name2],fontsize=20)
ax.xaxis.set_label_position("top")
plt.tight_layout()
#plt.axis('off')
#plt.title('Confusion matrix', y=1.1,fontsize=40)
#plt.ylabel('Actual label',fontsize=20)
#plt.xlabel('Predicted label',fontsize=20)

print('Accuracy:',accuracy_score(y_test, y_pred))
plt.show()



# =============================================================================
# accumulate_matrix = np.zeros(shape=(15,15))
# lst_accu_stratified = []
# lst_F1_stratified = []
# 
# 
# skf = StratifiedKFold(n_splits=5,random_state=2024, shuffle=True)
# for train_index, test_index in skf.split(X, y):
#     #print('TRAIN:', train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
# 
#     # 创建SVM模型
#     model = make_pipeline(StandardScaler(), SVC(kernel='linear',probability=True,class_weight=dict_weights))# , C=1
#     
#     # 训练模型
#     model.fit(X_train, y_train)
#     
#     # 预测
#     y_pred2 = model.predict(X_test)
#     
#     accuracy = model.score(X_test, y_test)
#     print(f"SVM Accuracy: {accuracy}")
#     F1_score = f1_score(y_test, y_pred2,average='macro')
#     print(f"SVM F1_score: {F1_score}")
#     
#     # 计算精度、召回率、F1分数和支持度
#     report = classification_report(y_test, y_pred2, target_names=movement_order)
#     print(report)
#     
#     # 绘制混淆矩阵
#     conf_matrix = confusion_matrix(y_test, y_pred2)
#     print(conf_matrix)
#     
#     accumulate_matrix += conf_matrix
#     lst_accu_stratified.append(clf.score(X_test, y_test))
#     lst_F1_stratified.append(f1_score(y_test, y_pred2,average='macro'))
#     
# fig, ax = plt.subplots(figsize=(10,10),dpi=600)
# #plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues,)
# sns.heatmap(pd.DataFrame(accumulate_matrix),annot=True, cmap=plt.cm.get_cmap('Blues', 10),cbar=False,annot_kws={"fontsize":18,"fontfamily":'arial',"fontweight":'bold'},linecolor='black',linewidths=2,vmin=0,vmax=500,
#             xticklabels=False,yticklabels=False, fmt='.0f',)
# classes = movement_order
# tick_marks = np.arange(len(classes))
# plt.xticks([])
# plt.yticks([])
# #plt.colorbar()
# ax.spines['bottom'].set_linewidth(4)
# ax.spines['left'].set_linewidth(4)
# ax.spines['top'].set_linewidth(4)
# ax.spines['right'].set_linewidth(4)
# 
# plt.savefig(r'F:\spontaneous_behavior\04返修阶段\Figure_and_code\sFigure1_Movement_skeleton&parameter\xcorr\movement_SVM_prediction.png',transparent=True, dpi=300)
# 
# 
# #df_xcoor.to_csv(r'F:\spontaneous_behavior\04返修阶段\Figure_and_code\Figure1_ExpeimentDesign&DataProcessing\movement_correlation\combine_movement_xcoor.csv')
# 
# 
# 
# ######################################################################################################################################################
# 
# 
# 
# distance_matrix_df = pd.DataFrame(data=distance_matrix,columns=movement_label_list)
# 
# 
# info_dict = {'revised_movement_label':[],'intra_coff_mean':[],'inter_coff_mean':[]}
# 
# for column in distance_matrix_df.columns.unique():
#     intra_coff = []
#     inter_coff = []
#     
#     
#     sub_distMatIntra = distance_matrix_df[[column]]
#     sub_distMatIntra_average = sub_distMatIntra.mean(axis=1)
#     sub_distMatIntra.reset_index(drop=True,inplace=True)
#     
#     sub_distMatInter = distance_matrix_df.iloc[:, distance_matrix_df.columns != column]
#     sub_distMatInter.reset_index(drop=True,inplace=True)
#     sub_distMatInter.sample(n=100,axis=1)
#     
#     skip_list = []
#     for i in range(sub_distMatIntra.shape[1]):
#         xx = np.array(sub_distMatIntra.iloc[:,i]).astype(float)
#         skip_list.append(i)
#         for j in range(sub_distMatIntra.shape[1]):
#             if j not in skip_list:                
#                 yy = np.array(sub_distMatIntra.iloc[:,j]).astype(float)
#                 tem_coff = np.corrcoef(xx, yy)
#                 intra_coff.append(tem_coff[0][1])
#                 info_dict['revised_movement_label'].append(column)
#                 info_dict['intra_coff_mean'].append(tem_coff[0][1])
#                 info_dict['inter_coff_mean'].append(np.nan)
#         for k in range(sub_distMatInter.shape[1]):
#             zz = np.array(sub_distMatInter.iloc[:,k]).astype(float)
#             tem_coff = np.corrcoef(xx, zz)
#             inter_coff.append(tem_coff[0][1])
#             info_dict['revised_movement_label'].append(column)
#             info_dict['inter_coff_mean'].append(tem_coff[0][1])
#             info_dict['intra_coff_mean'].append(np.nan)
#     
#     #info_dict['revised_movement_label'].append(column)
#     #info_dict['intra_coff_mean'].append(np.mean(intra_coff))
#     #info_dict['inter_coff_mean'].append(np.mean(inter_coff))
# df = pd.DataFrame(info_dict)
# 
# new_df = pd.DataFrame()
# num = 0
# for l in df.index:
#     for m in  df.columns[1:]: 
#         new_df.loc[num,'revised_movement_label'] =  df.loc[l,'revised_movement_label']
#         new_df.loc[num,'group'] =  m
#         new_df.loc[num,'value'] =  df.loc[l,m]
#         num += 1
# 
# fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(20,5),dpi=300)
# sns.violinplot(x = 'revised_movement_label', y='value',hue='group',data=new_df,order=movement_order,cut=-0.5,scale_hue=True,bw='silverman',width=0.9,)
# plt.xticks(rotation=70)
# 
# 
#         #                     zz = np.array(sub_distMatInter.loc[k,['umap1','umap2','zs_velocity']]).astype(float)
#         #                     tem_coff = np.corrcoef(xx, zz)
#         #                     inter_coff.append(tem_coff[0][1])
#                 
# 
# 
# 
# # =============================================================================
# # for video_index in calculated_dataset['video_index']:
# #     FeA_data = pd.read_csv(FeA_path_dict[video_index])
# #     FeA_data = FeA_data[FeA_data['OriginalDigital_label'].isin(select_for_plot)]
# #     annoMV_data = pd.read_csv(annoMV_path_dict[video_index])
# #     FeA_data = annoFeA(FeA_data,annoMV_data)
# #     #FeA_data = add_movement_label_LightON(FeA_data)
# #       
# #     for mv in movement_order:
# #     #for mv in ['jumpping']:
# #         intra_coff = []
# #         inter_coff = []
# #         
# #         sub_distMatIntra = FeA_data[FeA_data['revised_movement_label']==mv]
# #         sub_distMatIntra.reset_index(drop=True,inplace=True)
# #         
# #         sub_distMatInter = FeA_data[FeA_data['revised_movement_label']!=mv]
# #         sub_distMatInter.reset_index(drop=True,inplace=True)
# #         
# #         if len(sub_distMatIntra) >5 and len(sub_distMatInter) > 5:
# #             for i in sub_distMatIntra.sample(5,random_state=1).index:
# #                 xx = np.array(sub_distMatIntra.loc[i,['umap1','umap2','zs_velocity']]).astype(float)
# #                 for j in sub_distMatIntra.sample(5,random_state=2).index:
# #                     yy = np.array(sub_distMatIntra.loc[j,['umap1','umap2','zs_velocity']]).astype(float)
# #                     tem_coff = np.corrcoef(xx, yy)
# #                     intra_coff.append(tem_coff[0][1])
# #                     
# #                 for k in sub_distMatInter.sample(5,random_state=3).index:
# #                     zz = np.array(sub_distMatInter.loc[k,['umap1','umap2','zs_velocity']]).astype(float)
# #                     tem_coff = np.corrcoef(xx, zz)
# #                     inter_coff.append(tem_coff[0][1])
# #         
# #             info_dict['revised_movement_label'].append(mv)
# #             info_dict['intra_coff_mean'].append(np.mean(intra_coff))
# #             info_dict['inter_coff_mean'].append(np.mean(inter_coff))
# #         else:
# #             info_dict['revised_movement_label'].append(mv)
# #             info_dict['intra_coff_mean'].append(np.nan)
# #             info_dict['inter_coff_mean'].append(np.nan)
# # 
# # 
# # df = pd.DataFrame(info_dict)
# #     
# #     
# # =============================================================================
# =============================================================================


    
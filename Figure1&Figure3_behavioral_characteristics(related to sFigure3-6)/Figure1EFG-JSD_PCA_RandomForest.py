# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 22:52:01 2023

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
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split 
import matplotlib.path as mpath
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score, f1_score
import scipy.stats


InputData_path_dir = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure1&Figure3_behavioral_characteristics(related to sFigure3-6)\group_dissimilarities'
if not os.path.exists(output_dir):             
    os.mkdir(output_dir)


def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-')) & (file_name.endswith(content)):
            video_index = int(file_name.split('-')[1])
            file_path_dict.setdefault(video_index,file_dir+'\\'+file_name)
    return(file_path_dict)

Movement_Label_path = get_path(InputData_path_dir,'Movement_Labels.csv')       



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


def extract_segment(df):                                                        ## Only use movement fractions as features
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
dataset_name1 = 'Morning_lightOn'
dataset_color1 = get_color(dataset_name1)

dataset2 = Stress_info
dataset_name2 =  'Stress'
dataset_color2 = get_color(dataset_name2)




all_array = []
all_matrix = []

for index in dataset1.index:
    video_index = dataset1.loc[index,'video_index']
    if video_index in skip_file_list:
        pass
    else:
        gender =  dataset1.loc[index,'gender']
        ExperimentCondition =  dataset1.loc[index,'ExperimentCondition']
        LightingCondition = dataset1.loc[index,'LightingCondition']
        MoV_file = pd.read_csv(Movement_Label_path[video_index])
        #location_file = pd.read_csv(Location_path1[video_index],usecols=['location'])
        #MoV_file = pd.concat([MoV_file,location_file],axis=1)
        MoV_matrix = extract_segment(MoV_file)
        all_array.append(MoV_matrix.values.flatten())
        MoV_matrix['group_id'] = 0
        MoV_matrix['mouse_info'] = gender+'-' + ExperimentCondition+'-' + LightingCondition
        all_matrix.append(MoV_matrix)

for index in dataset2.index:
    video_index = dataset2.loc[index,'video_index']
    if video_index in skip_file_list:
        pass
    else:
        gender =  dataset2.loc[index,'gender']
        ExperimentCondition =  dataset2.loc[index,'ExperimentCondition']
        LightingCondition = dataset2.loc[index,'LightingCondition']
        MoV_file = pd.read_csv(Movement_Label_path[video_index])
        #location_file = pd.read_csv(Location_path2[video_index],usecols=['location'])
        #MoV_file = pd.concat([MoV_file,location_file],axis=1)
        MoV_matrix = extract_segment(MoV_file)
        all_array.append(MoV_matrix.values.flatten())
        MoV_matrix['group_id'] = 1
        MoV_matrix['mouse_info'] = gender+'-' + ExperimentCondition + '-' + LightingCondition
        all_matrix.append(MoV_matrix)

all_df = pd.concat(all_matrix)
all_df.reset_index(drop=True,inplace=True)

feat_cols = all_df.columns[:-2]
X = all_df[feat_cols].values
y = all_df['group_id']
X = StandardScaler().fit_transform(X)



#%% JSD valus calculation

def JS_divergence(p,q):
    M=(p+q)/2
    return (0.5*scipy.stats.entropy(p, M, base=2)+0.5*scipy.stats.entropy(q, M, base=2))

df = pd.DataFrame(index=range(13),columns=range(13))
for i in range(len(all_array)):
    Q = all_array[i] / all_array[i].sum()
    for j in range(len(all_array)):
        P = all_array[j] / all_array[j].sum()
        df.loc[i,j] = JS_divergence(Q, P)
#df = df.replace([np.inf,-np.inf],2)
df = df.astype('float32')
fig,ax = plt.subplots(nrows=1, ncols=1,figsize=(10,10),dpi=300)
     
im = ax.imshow(df.values,cmap='copper',norm='linear',vmin=0,vmax=0.15)   #copper
plt.xticks([])
plt.yticks([])

plt.savefig('{}/{}&{}_JSD.png'.format(output_dir,dataset_name1,dataset_name2),dpi=300)

plt.show()


#%% low-dimensional embedding (PCA)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)

PC1 = principalComponents[:,0]
PC2 = principalComponents[:,1]
#PC3 = principalComponents[:,2]

fig,ax= plt.subplots(nrows=1,ncols=1,figsize=(10,10),dpi=600)


##################### construct female and male marker in the figure #####################
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

###############################  plot PCA space   ###################################   

principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, all_df[['mouse_info']]], axis = 1)
targets = all_df['mouse_info'].unique()

colors = [dataset_color1,dataset_color1,dataset_color2,dataset_color2]
shapes = [female_marker,male_marker,male_marker,female_marker,'*','*']
for target, color,shape in zip(targets,colors,shapes):
     indicesToKeep = finalDf['mouse_info'] == target
     # 选择某个label下的数据进行绘制
     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                , finalDf.loc[indicesToKeep, 'principal component 2']
                , c = color
                , ec = 'black'
                , s = 2000
                , lw = 4
                ,marker=shape
                ,alpha=0.8)

ax.spines['bottom'].set_linewidth(5)
ax.spines['left'].set_linewidth(5)
ax.spines['top'].set_linewidth(5)
ax.spines['right'].set_linewidth(5)

plt.xlim(-8,8)
plt.ylim(-6,6)
plt.xticks([])
plt.yticks([])

##################################### SVM training  #####################################
svm = SVC(kernel='linear')
svm.fit(principalComponents, all_df['group_id'])

w = svm.coef_[0]                       
a = -w[0]/w[1]                        
x1 = np.linspace(-5,5)                  
y1 = a * x1 -(svm.intercept_[0])/w[1]    

b = svm.support_vectors_[0]        
y_down = a * x1 + (b[1] - a*b[0])   
b = svm.support_vectors_[-1]       
y_up = a * x1 + (b[1] - a*b[0])     

plt.plot(x1, y1, 'k-', label='SVM Decision Boundary')
plt.plot(x1,y_down,'k--')   
plt.plot(x1,y_up,'k--')     

# Show the plot
#plt.axis('off')
plt.savefig('{}/{}&{}_PCA.png'.format(output_dir,dataset_name1,dataset_name2),dpi=600)
plt.show()

#%% Rondom Forest prediction

skf = StratifiedKFold(n_splits=10,random_state=2020, shuffle=True)

accumulate_matrix = np.zeros(shape=(2,2))
lst_accu_stratified = []
lst_F1_stratified = []
importances_feature_list = []
for train_index, test_index in skf.split(X, y):
    #print('TRAIN:', train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = RandomForestClassifier(n_estimators=10, random_state=52,bootstrap=True,criterion='gini',class_weight='balanced',max_depth=100,min_samples_split=2)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    accuracy = clf.score(X_test, y_test)
    F1_score = f1_score(y_test, y_pred,average='macro')
   
    cm_matrix = confusion_matrix(y_test, y_pred)
    accumulate_matrix += cm_matrix
    lst_accu_stratified.append(accuracy)
    lst_F1_stratified.append(F1_score)
    
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
    accumulate_importances_df.loc[num,'average_importances'] = all_importances_df[all_importances_df['feature']==feature]['importances'].mean()
    accumulate_importances_df.loc[num,'accumulative_importances'] = accumulate_importances
    accumulate_importances_df.loc[num,'feature'] = feature
    num+=1

accumulate_importances_df.sort_values(by=['average_importances'],ascending=False,inplace=True)


fig, ax = plt.subplots(figsize=(10,8),dpi=600)
sns.barplot(data=all_importances_df,x='feature',y='importances',color='#1565C0',order=accumulate_importances_df['feature'])
#sns.lineplot(data=accumulate_importances_df,x='feature',y='accumulative_importances')
sns.stripplot(data=all_importances_df,x='feature',y='importances',jitter=0.25,size=10,edgecolor='black',linewidth=2,facecolor='white',alpha=0.8,order=accumulate_importances_df['feature'])

ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.set_yticks(np.arange(0,0.31,0.1))
plt.xticks(rotation=60)

plt.savefig('{}/{}&{}_RandomForest_feature_importances.png'.format(output_dir,dataset_name1,dataset_name2),dpi=300,transparent=True)

print('average_accuracy:{}'.format(np.mean(lst_accu_stratified)))
print('average_F1_score:{}'.format(np.mean(lst_F1_stratified)))

class_names=[0,1] # name  of classes
fig, ax = plt.subplots(figsize=(10,10),dpi=300)
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
#plt.text(x=0.2,y=-.1,s='Random forest prediction, Accuracy:{:.2f}'.format(accuracy_score(y_test, y_pred)),fontsize=26)
# create heatmap
sns.heatmap(pd.DataFrame(accumulate_matrix),annot=True, cmap="Greys",cbar=False,annot_kws={"fontsize":50,"fontfamily":'arial',"fontweight":'bold'})
ax.set_xticklabels([dataset_name1,dataset_name2],fontsize=20)
ax.set_yticklabels([dataset_name1,dataset_name2],fontsize=20)
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.axis('off')
#plt.title('Confusion matrix', y=1.1,fontsize=40)
#plt.ylabel('Actual label',fontsize=20)
#plt.xlabel('Predicted label',fontsize=20)

print('Accuracy:',accuracy_score(y_test, y_pred))
plt.savefig('{}/{}&{}_RandomForest_prediction.png'.format(output_dir,dataset_name1,dataset_name2),dpi=300,transparent=True)
plt.show()



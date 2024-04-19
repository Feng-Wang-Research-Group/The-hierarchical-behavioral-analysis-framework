# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:01:53 2024

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
from sklearn.model_selection import StratifiedKFold
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import math
from scipy.stats import bootstrap
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report,roc_curve, auc,confusion_matrix,f1_score


InputData_path_dir = r"F:\spontaneous_behavior\GitHub\spontaneous_behavior_analysis\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\spontaneous_behavior_analysis\Figure1&Figure3_behavioral_characteristics(related to sFigure3-5)\movement_statistic'
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
Feature_space_path = get_path(InputData_path_dir,'Feature_Space.csv')


skip_file_list = [1,3,28,29,110,122] 

animal_info_csv = r'F:\spontaneous_behavior\GitHub\spontaneous_behavior_analysis\Table_S1_animal_information.csv'
animal_info = pd.read_csv(animal_info_csv)
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)]


Morning_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Morning') & (animal_info['LightingCondition']=='Light-on')]
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-off')]

Stress_info = animal_info[(animal_info['ExperimentCondition']=='Stress')]

movement_order = ['running','trotting','left_turning','right_turning','walking','stepping',
                  'sniffing','rising','hunching','rearing','climbing','jumping','grooming','scratching','pausing',]

big_cluster_dict4 = {'locomotion':['running','trotting','walking','left_turning','right_turning','stepping'],
                     'exploration':['climbing','rearing','hunching','rising','sniffing','jumping'],
                     'maintenance':['grooming','scratching'],
                     'nap':['pausing'],
                     } 

exploration_move=['climbing','rearing','hunching','rising','sniffing','jumping']

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


def extract_segment(df):                                                        ## Only use movement fractions as features    
    df_mov_fra = pd.DataFrame()
    label_count = df.value_counts('revised_movement_label')

    num = 0
    for mv in movement_order:
        if mv in label_count.index:
            count = label_count[mv]
        else:
            count = 0
        df_mov_fra.loc[num,'movement_label'] = mv
        df_mov_fra.loc[num,'movement_fraction'] = count
        num += 1
    df_mov_fra['movement_fraction'] = (df_mov_fra['movement_fraction'] /df_mov_fra['movement_fraction'].sum())*100
    return(df_mov_fra)

def cal_distribution(arr):
    data = (arr,)
    #res = bootstrap(data, np.std, confidence_level=0.9,random_state=rng)
    res = bootstrap(data, np.mean, axis=-1, confidence_level=0.95, n_resamples=10000, random_state=1)
    ci_l, ci_u = res.confidence_interval
    resample_values = res.bootstrap_distribution
    
    return(np.mean(resample_values))

def cal_interval(FeA):
    FeA_copy = FeA.copy()
    FeA_copy.reset_index(drop=True,inplace=True)
    interval_list = []
    for i in range(1,FeA_copy.shape[0]):
        t1 = FeA_copy.loc[i-1,'segBoundary_end']
        t2 = FeA_copy.loc[i,'segBoundary_start']
        interval = t2-t1
        interval_list.append(interval)
    return(interval_list)

def cal_otherPara(FeA):
    info_dict = {'movement_label':[],'movement_frequency':[],'movement_duration':[],'movement_intervels':[]}
    df = pd.DataFrame()
    num = 0
    for mv in movement_order:
        if mv in list(FeA['revised_movement_label'].values):
            FeA_mv = FeA[FeA['revised_movement_label']==mv]
            info_dict['movement_label'].append(mv)
            info_dict['movement_frequency'].append(len(FeA_mv))
            
            if len(list(FeA_mv['frame_length'].values)) < 10:
                average_frame_length = np.mean(FeA_mv['frame_length'].values)
            else:
                average_frame_length = cal_distribution(list(FeA_mv['frame_length'].values))
            
            info_dict['movement_duration'].append(average_frame_length)
            
            if len(list(FeA_mv['frame_length'].values)) < 10:
                average_duration_length = np.mean(cal_interval(FeA_mv))
            else:
                average_duration_length = cal_distribution(cal_interval(FeA_mv))
            
            info_dict['movement_intervels'].append(average_duration_length)
        else:
            info_dict['movement_label'].append(mv)
            info_dict['movement_frequency'].append(np.nan)
            info_dict['movement_duration'].append(np.nan)
            info_dict['movement_intervels'].append(np.nan)
            
    df_output = pd.DataFrame(info_dict)
    
    return(df_output)


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


dataset_color = {'Morning_Light-on':'#F5B25E',
                 'Afternoon_Light-on':'#936736',
                 'Night_Light-on':'#398FCB',
                 'Night_Light-off':'#003960',
                 'Stress_Light-on':'#f55e6f',}

dataset_order = ['Morning_Light-on',
                 'Afternoon_Light-on',
                 'Night_Light-on',
                 'Night_Light-off',
                 'Stress_Light-on',]

dataset = animal_info

def selectGroup_plotvariable(combine_df,plotvariable):
    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(12,3),dpi=1200,sharex=True)
    
    num = 0
    for sex in combine_df['gender'].unique():
        data1 = combine_df[combine_df['gender']==sex]
        if sex == 'female':
            marker = '^'
        else:        
            marker = 'o'
        
        sns.stripplot(
            data = data1, x='movement_label', y=plotvariable,hue='group',
            
            order=movement_order,
            dodge=True, alpha=.8, legend=False,jitter=0.2,
            linewidth=0.3,edgecolor='black',
            size=4,
            palette=dataset_color.values(),
            marker=marker,
            zorder=0,ax=ax
            )
    
    
    sns.boxplot(showmeans=True,
                meanline=True,
                meanprops={'color':'black', 'ls': '-', 'lw': 2,'label': '_mean_'},
                medianprops={'visible': False},
                whiskerprops={'visible': False},
                hue_order=dataset_order,
                order=movement_order,
                palette=dataset_color.values(),
                zorder=10,
                x="movement_label",
                y= plotvariable,
                hue='group',
                data=combine_df,
                showfliers=False,
                showbox=False,
                showcaps=False,
                legend=True,
                ax=ax)
    
    
    ax.set_xlabel('Movements')
    ax.set_title(plotvariable)
    plt.xticks(rotation=60)
    plt.legend()
    plt.savefig(r'{}\comparision_of_{}.png'.format(output_dir,plotvariable),dpi=300,transparent=True)

    

time_window1 = 0
time_window2 = 60

all_matrix = []
for index in dataset.index:
    video_index = dataset.loc[index,'video_index']
    gender =  dataset.loc[index,'gender']
    ExperimentCondition =  dataset.loc[index,'ExperimentCondition']
    LightingCondition = dataset.loc[index,'LightingCondition']
    MoV_file = pd.read_csv(Movement_Label_path[video_index])
    MoV_file = MoV_file.iloc[time_window1*30*60:time_window2*30*60,:]
    #MoV_file = MoV_file[(MoV_file['back_x']>(250-175))&(MoV_file['back_x']<(250+175))&(MoV_file['back_y']>(250-175))&(MoV_file['back_y']<(250+175))]
    MoV_file.reset_index(drop=True,inplace=True)
    FeA_file = pd.read_csv(Feature_space_path[video_index])
    FeA_file = FeA_file[(FeA_file['segBoundary_start']>=time_window1*30*60)&(FeA_file['segBoundary_end']<=time_window2*30*60)]
    Mov_para = cal_otherPara(FeA_file)
    MoV_frac = extract_segment(MoV_file)
    combine_df = MoV_frac.join(Mov_para.set_index('movement_label'), on='movement_label')
    combine_df['gender'] = gender
    combine_df['group'] = ExperimentCondition+'_' + LightingCondition
    all_matrix.append(combine_df)

all_df = pd.concat(all_matrix,axis=0)
all_df = all_df.fillna(0)
all_df['group_id'] = all_df.groupby('group').ngroup()
#all_df['color'] = all_df['movement_label'].map(movement_color_dict)
all_df.reset_index(drop=True,inplace=True)

all_df.to_csv(r'{}/movement_statistic.csv'.format(output_dir),index=None)


selectGroup_plotvariable(all_df,'movement_fraction')
selectGroup_plotvariable(all_df,'movement_frequency')
selectGroup_plotvariable(all_df,'movement_duration')
selectGroup_plotvariable(all_df,'movement_intervels')

#######  sniffing/grooming index
sniffing_df = all_df[all_df['movement_label']=='sniffing']
grooming_df = all_df[all_df['movement_label']=='grooming']
min_grooming = np.min(grooming_df[grooming_df['movement_fraction']>0]['movement_fraction'])
min_grooming = print(min_grooming/10)

for i in grooming_df.index:
    if grooming_df.loc[i,'movement_fraction'] == 0:
        grooming_df.loc[i,'movement_fraction'] = 0.001

df_index = pd.DataFrame()
df_index['group'] = sniffing_df['group']
df_index['group_id'] =sniffing_df['group_id']
df_index['gender'] = sniffing_df['gender']
df_index['sniffing_fra'] = sniffing_df['movement_fraction'].values
df_index['grooming_fra'] = grooming_df['movement_fraction'].values
df_index['sniffing_fre'] = sniffing_df['movement_frequency'].values
df_index['grooming_fre'] = grooming_df['movement_frequency'].values
df_index['SG_fra_index'] = sniffing_df['movement_fraction'].values/grooming_df['movement_fraction'].values
df_index['SG_fre_index'] = sniffing_df['movement_frequency'].values/grooming_df['movement_frequency'].values
df_index['color'] = df_index['group'].map(dataset_color)
df_index.reset_index(drop=True,inplace=True)
df_index.to_csv(r'{}\sniffing&groming_index.csv'.format(output_dir))


### construct marker
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

gender_marker = {'male':male_marker,'female':female_marker}
df_index['marker'] = df_index['gender'].map(gender_marker)

fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(10,10),dpi=300)
for index in df_index.index:
    x = df_index.loc[index,'SG_fra_index']
    y = df_index.loc[index,'SG_fre_index']
    c = df_index.loc[index,'color']
    marker = df_index.loc[index,'marker']
    ax.scatter(x=x,y=y,color=c,marker=marker,s=900,lw=2,alpha=0.9)

ax.set_xlabel(r'Sniffing/grooming fraction index')
ax.set_ylabel(r'Sniffing/grooming frequency index')
ax.set_title(r'Sniffing/grooming index')
# =============================================================================
# ax.spines['bottom'].set_linewidth(4)
# ax.spines['left'].set_linewidth(4)
# ax.spines['top'].set_linewidth(0)
# ax.spines['right'].set_linewidth(0)
# plt.xscale("log")
# plt.yscale("log")
# ax.xaxis.set_major_formatter(plt.NullFormatter())
# ax.yaxis.set_major_formatter(plt.NullFormatter())
# =============================================================================
ax.tick_params(length=7,width=3)

plt.xlim(0,20)
plt.ylim(0,20)

plt.savefig(r'{}\sniffing&groming_index_scatters.png'.format(output_dir),transparent=False,dpi=300)

### 
fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(10,10),dpi=300)
ax.scatter(x=df_index['grooming_fra'],y=df_index['sniffing_fra'],color=df_index['color'],s=50,lw=1,ec='white')
ax.set_xlabel(r'Sniffing')
ax.set_ylabel(r'grooming')
ax.set_title('Movement fraction')
plt.savefig(r'{}\sniffing&groming_MovementFraction_scatters.png'.format(output_dir),transparent=False,dpi=300)

fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(10,10),dpi=300)
ax.scatter(x=df_index['grooming_fre'],y=df_index['sniffing_fre'],color=df_index['color'],s=50,lw=1,ec='white')
ax.set_xlabel(r'Sniffing')
ax.set_ylabel(r'grooming')
ax.set_title('Movement frequency')
plt.savefig(r'{}\sniffing&groming_MovementFrequency_scatters.png'.format(output_dir),transparent=False,dpi=300)


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value
def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels,rotation=60)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')
    ax.set_yscale("log")
    
########################### violin plot ##############################

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5), sharey=True)
SG_fra_index_list = []
for group in dataset_order:
    SG_fra_index_list.append(df_index.loc[df_index['group']==group,'SG_fra_index'].values.tolist())
ax.set_title('Sniffing/grooming (Movement Fraction)')
parts = ax.violinplot(
        SG_fra_index_list, showmeans=False, showmedians=False,
        showextrema=False,widths=0.7)
for pc,color in zip(parts['bodies'],dataset_color.values()):
    pc.set_facecolor(color)
    pc.set_edgecolor('black')
    pc.set_alpha(1)

quartile1 = []
medians = [] 
quartile3 = [] 

for arr in SG_fra_index_list:

    a,b,c = np.percentile(arr, [25, 50, 75])
    quartile1.append(a)
    medians.append(b)
    quartile3.append(c)

whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(SG_fra_index_list, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
ax.scatter(inds, medians, marker='o', color='white', s=60, zorder=3)
ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=8)
ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=3)

# set style for the axes
labels = dataset_order
set_axis_style(ax, labels)

plt.savefig(r'{}\sniffing&groming_movementfraction_index.png'.format(output_dir),transparent=False,dpi=300)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5), sharey=True)
SG_fra_index_list = []
for group in dataset_order:
    SG_fra_index_list.append(df_index.loc[df_index['group']==group,'SG_fre_index'].values.tolist())
ax.set_title('Sniffing/grooming (Movement Frequency)')
parts = ax.violinplot(
        SG_fra_index_list, showmeans=False, showmedians=False,
        showextrema=False,widths=0.7)
for pc,color in zip(parts['bodies'],dataset_color.values()):
    pc.set_facecolor(color)
    pc.set_edgecolor('black')
    pc.set_alpha(1)

quartile1 = []
medians = [] 
quartile3 = [] 

for arr in SG_fra_index_list:

    a,b,c = np.percentile(arr, [25, 50, 75])
    quartile1.append(a)
    medians.append(b)
    quartile3.append(c)

whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(SG_fra_index_list, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
ax.scatter(inds, medians, marker='o', color='white', s=60, zorder=3)
ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=8)
ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=3)

# set style for the axes
labels = dataset_order
set_axis_style(ax, labels)

plt.savefig(r'{}\sniffing&groming_movementfrequency_index.png'.format(output_dir),transparent=False,dpi=300)


################################# random forest prediction ##################

X = df_index[['SG_fra_index','SG_fre_index']]
y = df_index['group_id']

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
    
    importances_feature = clf.feature_importances_
    importances_df = pd.DataFrame(data=importances_feature,columns=['importances'],index=['SG_fra_index','SG_fre_index'])
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
ax.set_title('Features importance')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
#ax.set_yticks(np.arange(0,0.31,0.1))
plt.xticks(rotation=60)

fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(10,10),dpi=300)
#plt.imshow(accumulate_matrix, interpolation='nearest', cmap=plt.cm.Blues,)
sns.heatmap(pd.DataFrame(accumulate_matrix),annot=True, cmap="Blues",cbar=False,annot_kws={"fontsize":40,"fontfamily":'arial',"fontweight":'bold'})
classes = dataset_order
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks+0.5, classes,rotation=60)
plt.yticks(tick_marks+0.5, classes)

ax.axhline(y=0, color='black',linewidth=10)
ax.axhline(y=accumulate_matrix.shape[1], color='black',linewidth=10)
ax.axvline(x=0, color='black',linewidth=10)
ax.axvline(x=accumulate_matrix.shape[0], color='black',linewidth=10)
# =============================================================================
# ax.set_xlabel('Predicted_label')
# ax.set_ylabel('True_label')
# ax.set_title('Random Forest classifier')
# 
# =============================================================================
plt.xticks(rotation=60)
plt.axis('off')
plt.savefig(r'{}\RF_classification.png'.format(output_dir),transparent=False,dpi=300)

plt.show()



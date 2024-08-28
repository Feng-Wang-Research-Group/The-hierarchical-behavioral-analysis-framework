# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 09:45:03 2023

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
import scipy.stats as stats



InputData_path_dir = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure4_time-varying(related to sFigure7)'

def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-')) & (file_name.endswith(content)):
            video_index = int(file_name.split('-')[1])
            file_path_dict.setdefault(video_index,file_dir+'\\'+file_name)
    return(file_path_dict)

Movement_Label_path = get_path(InputData_path_dir,'revised_Movement_Labels.csv')
Feature_space_path = get_path(InputData_path_dir,'Feature_Space.csv')


skip_file_list = [1,3,28,29,110,122] 
animal_info_csv =  r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Table_S1_animal_information.csv'             
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


def extract_segment(df):                                                        ## Only use movement fractions as features
    mv_dict = {}
    for mv in movement_order:
        mv_dict.setdefault(mv,0)
    count_df = pd.DataFrame(mv_dict,index=[0])

    label_count = df.value_counts('revised_movement_label')
    for count_mv in label_count.index:
        count_df[count_mv]  = label_count[count_mv]
    return(count_df)


movement_color_dict = {'running':'#FF3030',
                       'trotting':'#F06292',
                       'walking':'#FF5722',
                       'right_turning':'#F29B78',
                       'left_turning':'#FFBFB4',                       
                       'stepping':'#A1887F',  
                       'rising':'#FFEA00',
                       'hunching':'#ECAD4F',
                       'rearing':'#C0CA33',
                       'climbing':'#2E7939',                           
                       'jumping':'#80DEEA',
                       'sniffing':'#2C93CB',                       
                       'grooming':'#A13E97',
                       'scratching':'#00ACC1',
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
dataset1_male = dataset1[dataset1['gender']=='male']
dataset1_female = dataset1[dataset1['gender']=='female']
dataset1_color = get_color(dataset1_name)


dataset2 = stress_animal_info
dataset2_name = 'stress_animal_info'
dataset2_male = dataset2[dataset2['gender']=='male']
dataset2_female = dataset2[dataset2['gender']=='female']
dataset2_color = get_color(dataset2_name)   



 
def movement_label_count(df,time_segement):
    start = 0
    end = 0

    count_df_list = []
    num = 0
    for i in range(time_segement*60*30,df.shape[0]+5,time_segement*60*30):
        num += 1*time_segement
        mv_dict = {}
        for mv in movement_order:
            mv_dict.setdefault(mv,0)
            mv_dict['time'] = num
        count_df = pd.DataFrame(mv_dict,index=[num])
        
        end = i
        temp_df = df.loc[start:end,:]
        label_count = temp_df.value_counts('revised_movement_label')
        
        for count_mv in label_count.index:
            count_df[count_mv]  = label_count[count_mv]
        count_df_list.append(count_df)
       
        start = end
    combine_df = pd.concat(count_df_list,axis=0)
    
    
    # calculate_columns_percentage(df)
    for i in combine_df.columns:
        new_col = i + '_percent'
        combine_df[new_col] = round((combine_df[i] / combine_df[i].sum())*100,2)
    combine_df.fillna(0)
    return(combine_df)
    return(combine_df)

def calculate_distanceByMin(df,section_legth):
    start = 0
    end = 0
    num = 0
    accumulative_distance = 0
    info_dict = {'time':[],'total_distance':[],'accumulative_distance':[]}
    for i in range(section_legth*30*60,df.shape[0]+5,section_legth*30*60):
        num += section_legth
        end = i
        total_distance = df.loc[start:end,'smooth_speed'].sum() *(1/30)        ### 1s 30Hz
        accumulative_distance += total_distance
        info_dict['time'].append(num)
        info_dict['total_distance'].append(total_distance)
        info_dict['accumulative_distance'].append(accumulative_distance)
        start = end
    df_out = pd.DataFrame(info_dict)
    return(df_out)


def calculate_chaosByMin(df,section_legth):
    
    start = 0
    end = 0
    num = 0
    info_dict = {'time':[],'trans_num':[]}
    for i in range(section_legth*30*60,30*60*60+5,section_legth*30*60):
        end = i
        num += section_legth
        df_select = df[(df['segBoundary_start']>= start) & (df['segBoundary_end']<= end)]
        trans_num = len(df_select)

        info_dict['time'].append(num)
        info_dict['trans_num'].append(trans_num)
        start = end
    df_out = pd.DataFrame(info_dict)
    return(df_out)


def average_df(temp_df):
    #temp_df = pd.concat(df_list,axis=0)
    temp_df = temp_df.fillna(0)
    new_dict = pd.DataFrame()
    
    for i in range(1,61):
        time = i
        df_min = temp_df[temp_df['time']==i]
        for mv in movement_order:
            mv_percentage_name = mv + '_percent'
            mv_average_name = mv + '_mean'
            mv_sem_name = mv + '_sem'
            
            mv_arr = df_min[mv_percentage_name].values
            mv_average = np.mean(mv_arr)
            mv_sem = np.std(mv_arr,ddof=1) / np.sqrt(len(mv_arr))
            
            new_dict.loc[i,'time'] = time
            new_dict.loc[i,mv_average_name] = mv_average
            new_dict.loc[i,mv_sem_name] = mv_sem
    df_output = pd.DataFrame(new_dict)
    return(df_output)

def time_Speed_Distance(df_list):
    temp_df = pd.concat(df_list,axis=0)
    new_dict = {'time':[],'total_distance_mean':[],'total_distance_sem':[],'accumulative_distance_mean':[],'accumulative_distance_sem':[],}
    for i in range(1,61):
        minute_df = temp_df[temp_df['time']==i]
        total_distance_arr = minute_df['total_distance'].values
        total_distance_sem = np.std(total_distance_arr,ddof=1) / np.sqrt(len(total_distance_arr))
        total_distance_mean = np.mean(total_distance_arr)
        
        accumulative_distance_arr = minute_df['accumulative_distance'].values
        accumulative_distance_sem = np.std(accumulative_distance_arr,ddof=1) / np.sqrt(len(accumulative_distance_arr))
        accumulative_distance_mean = np.mean(accumulative_distance_arr)
        
        new_dict['time'].append(i)
        new_dict['total_distance_mean'].append(total_distance_mean)
        new_dict['total_distance_sem'].append(total_distance_sem)
        new_dict['accumulative_distance_mean'].append(accumulative_distance_mean)
        new_dict['accumulative_distance_sem'].append(accumulative_distance_sem)
        
    df_new = pd.DataFrame(new_dict)
    return(df_new)

def time_Chaos(df_list):
    temp_df = pd.concat(df_list,axis=0)
    new_dict = {'time':[],'chaos_mean':[],'chaos_sem':[]}
    for i in range(1,61):
        minute_df = temp_df[temp_df['time']==i]
        chaos_arr = minute_df['trans_num'].values
        chaos_sem = np.std(chaos_arr,ddof=1) / np.sqrt(len(chaos_arr))
        chaos_mean = np.mean(chaos_arr)
        
        new_dict['time'].append(i)
        new_dict['chaos_mean'].append(chaos_mean)
        new_dict['chaos_sem'].append(chaos_sem)
        
    df_new = pd.DataFrame(new_dict)
    return(df_new)


def time_Center_distance_rate(df_list):
    temp_df = pd.concat(df_list,axis=0)
    new_dict = {'time':[],'Center_distance_mean':[],'Center_distance_sem':[]}
    for i in range(1,61):
        minute_df = temp_df[temp_df['time']==i]
        Center_distance_arr = minute_df['average_center_ratio'].values *100
        Center_distance_sem = np.std(Center_distance_arr,ddof=1) / np.sqrt(len(Center_distance_arr))
        Center_distance_mean = np.mean(Center_distance_arr)
        
        new_dict['time'].append(i)
        new_dict['Center_distance_mean'].append(Center_distance_mean)
        new_dict['Center_distance_sem'].append(Center_distance_sem)
        
    df_new = pd.DataFrame(new_dict)
    return(df_new)



def plot_distance(df1,df2):
    fig, axes = plt.subplots(nrows = 1,ncols=1,sharex=True,figsize=(10,5),dpi=300)
    #axes.plot(df1.total_distance_mean,c=dataset1_color,alpha=0.2,lw=8)
    axes.plot(df1.total_distance_mean,c=dataset1_color,alpha=1,lw=4)
    #axes.plot(df1.total_distance_mean+df1.total_distance_sem,c=dataset1_color,alpha=0.3,lw=1)
    axes.fill_between(x=range(df1.shape[0]),y1=df1.total_distance_mean-df1.total_distance_sem,y2=df1.total_distance_mean+df1.total_distance_sem,color=dataset1_color,alpha=0.3,lw=1)
    #axes.plot(df1.total_distance_mean-df1.total_distance_sem,c=dataset1_color,alpha=0.3,lw=1)
    
    #axes.plot(df2.total_distance_mean,c=dataset2_color,alpha=0.2,lw=8)
    axes.plot(df2.total_distance_mean,c=dataset2_color,lw=4)
    #axes.plot(df2.total_distance_mean+df2.total_distance_sem,c=dataset2_color,alpha=0.3,lw=1)
    axes.fill_between(x=range(df2.shape[0]),y1=df2.total_distance_mean-df2.total_distance_sem,y2=df2.total_distance_mean+df2.total_distance_sem,color=dataset2_color,alpha=0.3,lw=1)
    
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['left'].set_linewidth(2)
    axes.spines['bottom'].set_linewidth(2)
    axes.set_ylim(1000,6500)
    axes.set_yticks(range(1000,6501,1000))
    axes.set_xticks(range(0,61,10))
    axes.set_ylabel('Travel distance (mm)',fontsize=10)
    axes.set_title('Travel distance',fontsize=15)
    #axes.plot(df2.total_distance_mean-df2.total_distance_sem,c=dataset2_color,alpha=0.3,lw=1)
    #axes.text(-10,np.mean(df2.total_distance_mean),'travel_distance',fontsize = 15,fontfamily='arial')
    #axes.axis('off')
    plt.savefig('{}/{}_{}_Temporal_travel_distance.png'.format(output_dir,dataset1_name,dataset2_name),transparent=True,dpi=300)


def plot_center_rate(df1,df2):
    fig, axes = plt.subplots(nrows = 1,ncols=1,sharex=True,figsize=(10,5),dpi=300)
    axes.plot(df1.Center_distance_mean,c=dataset1_color,alpha=1,lw=4)
    #axes.plot(df1.total_distance_mean+df1.total_distance_sem,c=dataset1_color,alpha=0.3,lw=1)
    axes.fill_between(x=range(df1.shape[0]),y1=df1.Center_distance_mean-df1.Center_distance_sem,y2=df1.Center_distance_mean+df1.Center_distance_sem,color=dataset1_color,alpha=0.3,lw=1)
    #axes.plot(df1.total_distance_mean-df1.total_distance_sem,c=dataset1_color,alpha=0.3,lw=1)
    
    axes.plot(df2.Center_distance_mean,c=dataset2_color,alpha=1,lw=4)
    #axes.plot(df2.total_distance_mean+df2.total_distance_sem,c=dataset2_color,alpha=0.3,lw=1)
    axes.fill_between(x=range(df2.shape[0]),y1=df2.Center_distance_mean-df2.Center_distance_sem,y2=df2.Center_distance_mean+df2.Center_distance_sem,color=dataset2_color,alpha=0.3,lw=1)
    
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['left'].set_linewidth(2)
    axes.spines['bottom'].set_linewidth(2)
    axes.set_ylim(60,78)
    axes.set_yticks(range(55,80,5))
    axes.set_xticks(range(0,61,10))
    axes.tick_params(length=2,width=1)
    axes.set_ylabel('Distance to center (%)',fontsize=10)
    axes.set_title('Distance to center',fontsize=15)
    #axes.xaxis.set_major_formatter(plt.NullFormatter())
    #axes.yaxis.set_major_formatter(plt.NullFormatter())
    #axes.axis('off')
    plt.savefig('{}/{}_{}_Temporal_distance2center_rate.png'.format(output_dir,dataset1_name,dataset2_name),transparent=True,dpi=300)
    
    
def plot_chaos(df1,df2):
    fig, axes = plt.subplots(nrows = 1,ncols=1,sharex=True,figsize=(10,5),dpi=300)
    axes.plot(df1.chaos_mean,c=dataset1_color,alpha=0.2,lw=8)
    axes.plot(df1.chaos_mean,c=dataset1_color)
    #axes.plot(df1.total_distance_mean+df1.total_distance_sem,c=dataset1_color,alpha=0.3,lw=1)
    axes.fill_between(x=range(df1.shape[0]),y1=df1.chaos_mean-df1.chaos_sem,y2=df1.chaos_mean+df1.chaos_sem,color=dataset1_color,alpha=0.3,lw=1)
    #axes.plot(df1.total_distance_mean-df1.total_distance_sem,c=dataset1_color,alpha=0.3,lw=1)
    
    axes.plot(df2.chaos_mean,c=dataset2_color,alpha=0.2,lw=8)
    axes.plot(df2.chaos_mean,c=dataset2_color)
    #axes.plot(df2.total_distance_mean+df2.total_distance_sem,c=dataset2_color,alpha=0.3,lw=1)
    axes.fill_between(x=range(df2.shape[0]),y1=df2.chaos_mean-df2.chaos_sem,y2=df2.chaos_mean+df2.chaos_sem,color=dataset2_color,alpha=0.3,lw=1)
    #axes.plot(df2.total_distance_mean-df2.total_distance_sem,c=dataset2_color,alpha=0.3,lw=1)
    
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['left'].set_linewidth(2)
    axes.spines['bottom'].set_linewidth(2)
    axes.set_ylim(15,65)
    axes.set_yticks(range(15,56,10))
    axes.set_xticks(range(0,61,10))
    axes.tick_params(length=2,width=1)
    axes.set_ylabel('Movement transition frequency (num)',fontsize=10)
    axes.set_title('Movement transition frequency',fontsize=15)
    #axes.xaxis.set_major_formatter(plt.NullFormatter())
    #axes.yaxis.set_major_formatter(plt.NullFormatter())
    
    #axes.axis('off')
    plt.savefig('{}/{}_{}_Temporal_movement_chaos_degree.png'.format(output_dir,dataset1_name,dataset2_name),transparent=True,dpi=300)

def plot(df1,df2,df3,df4,df5,df6,day_time_center_rate,night_time_center_rate):
    fig, axes = plt.subplots(nrows = len(movement_order),ncols=1,sharex=True,figsize=(15,13),dpi=300)
    
    for i in range(len(movement_order)):
        mv = movement_order[i]      
        mv_average_name = mv + '_mean'
        mv_sem_name = mv + '_sem'
        
        df1_average = average_df(df1)
        x1 = range(df1_average.shape[0])
        y1 = df1_average[mv_average_name]
        t1 = np.array(df1['time'])
        y1_all = np.array(df1[mv])
        
        #coefficients1 = np.polyfit(t1, y1_all, 1)
        #r1,p1 = stats.pearsonr(t1,y1_all)\
        
        coefficients1 = np.polyfit(x1, y1, 1)
        r1,p1 = stats.pearsonr(x1,y1)
        
        df2_average = average_df(df2)
        x2 = range(df2_average.shape[0])
        y2 = df2_average[mv_average_name]
        t2 = np.array(df2['time'])
        y2_all = np.array(df2[mv])
        #coefficients2 = np.polyfit(t2, y2_all, 1)
        #r2,p2 = stats.pearsonr(t2,y2_all)
        
        coefficients2 = np.polyfit(x2, y2, 1)
        r2,p2 = stats.pearsonr(x2,y2)
        label2 = mv
        
        #sns.barplot(data=df1,x='time',y=mv,color=dataset1_color,width=0.9,ax=axes[i])
        axes[i].bar(x = x1,height = y1,bottom = 10,color=dataset1_color,width=0.9)

        if p1 >=0.05:        
            axes[i].text(61,11.5,'r={:.2f},'.format(round(r1,2)),fontsize=12,fontfamily='arial',c='black',fontweight='bold')
            if r1 <0:
                axes[i].text(64.8,11.5,'n.s.'.format(round(r1,2)),fontsize=11,fontfamily='arial',c='black',fontweight='bold')
            else:
                axes[i].text(64.5,11.5,'n.s.'.format(round(r1,2)),fontsize=11,fontfamily='arial',c='black',fontweight='bold')
        else:
            if r1 == 0:
                axes[i].text(61,11.5,'r={:+.2f}'.format(round(r1,2)),fontsize=12,fontfamily='arial',c='black',fontweight='bold')
            elif (r1 > 0) & (p1<0.001):
                axes[i].text(61,11.5,'r={:.2f}, ***'.format(round(r1,2)),fontsize=12,fontfamily='arial',c='#E91E63',fontweight='bold')
            elif (r1 > 0) & (p1>=0.001) & (p1<0.01):
                axes[i].text(61,11.5,'r={:.2f}, **'.format(round(r1,2)),fontsize=12,fontfamily='arial',c='#E91E63',fontweight='bold')
            elif (r1 > 0) & (p1>=0.01)& (p1<0.05):
                axes[i].text(61,11.5,'r={:.2f}, *'.format(round(r1,2)),fontsize=12,fontfamily='arial',c='#E91E63',fontweight='bold')
            elif (r1 <0) & (p1<0.001):
                axes[i].text(61,11.5,'r={:+.2f}, ***'.format(round(r1,2)),fontsize=12,fontfamily='arial',c='#01579B',fontweight='bold')
            elif (r1 <0) & (p1>=0.001) & (p1<0.01):
                axes[i].text(61,11.5,'r={:+.2f}, **'.format(round(r1,2)),fontsize=12,fontfamily='arial',c='#01579B',fontweight='bold')
            elif (r1 <0) & (p1>=0.01)& (p1<0.05):
                axes[i].text(61,11.5,'r={:+.2f}, *'.format(round(r1,2)),fontsize=12,fontfamily='arial',c='#01579B',fontweight='bold')

        
        axes[i].bar(x = x2,height = -y2,bottom = 10,color=dataset2_color,width=0.9)
        if p2 >=0.05:
            axes[i].text(61,6.2,'r={:.2f},'.format(round(r2,2)),fontsize=12,fontfamily='arial',c='black',fontweight='bold')
            if r2 <0:
                axes[i].text(65,6.2,'n.s.'.format(round(r1,2)),fontsize=11,fontfamily='arial',c='black',fontweight='bold')
            else:
                axes[i].text(64.5,6.2,'n.s.'.format(round(r1,2)),fontsize=11,fontfamily='arial',c='black',fontweight='bold')
        else:
            if r2 == 0:
                axes[i].text(61,6.2,'r={:.2f}'.format(round(r2,2)),fontsize=12,fontfamily='arial',c='black',fontweight='bold')
            elif (r2 > 0) & (p2<0.001):
                axes[i].text(61,6.2,'r={:.2f}, ***'.format(round(r2,2)),fontsize=12,fontfamily='arial',c='#E91E63',fontweight='bold')
            elif (r2 > 0) & (p2>=0.001) & (p2<0.01):
                axes[i].text(61,6.2,'r={:.2f}, **'.format(round(r2,2)),fontsize=12,fontfamily='arial',c='#E91E63',fontweight='bold')
            elif (r2 > 0) & (p2>=0.01)& (p2<0.05):
                axes[i].text(61,6.2,'r={:.2f}, *'.format(round(r2,2)),fontsize=12,fontfamily='arial',c='#E91E63',fontweight='bold')
            elif (r2 < 0)& (p2<0.001):
                axes[i].text(61,6.2,'r={:+.2f}, ***'.format(round(r2,2)),fontsize=12,fontfamily='arial',c='#01579B',fontweight='bold')
            elif (r2 < 0)& (p2>=0.001) & (p2<0.01):
                axes[i].text(61,6.2,'r={:+.2f}, **'.format(round(r2,2)),fontsize=12,fontfamily='arial',c='#01579B',fontweight='bold')
            elif (r2 < 0)& (p2>=0.01)& (p2<0.05):
                axes[i].text(61,6.2,'r={:+.2f}, *'.format(round(r2,2)),fontsize=12,fontfamily='arial',c='#01579B',fontweight='bold')
        axes[i].plot((-1,66),(10,10),c='black',lw=2)
        #axes[i].plot((-1,-1),(4,16),c='black',lw=2)
        axes[i].set_ylim(4,16)
        #axes[i].text(-10,10,label1,fontsize = 15,fontfamily='arial')
        axes[i].set_yticks([])
        axes[i].axis('off')
    plt.savefig('{}/{}_{}_Temporal_varying_of_movement.png'.format(output_dir,dataset1_name,dataset2_name),transparent=True,dpi=300)



def average_center_ratio(coor_data):
    df_output = pd.DataFrame()
    num = 0
    for time_i in range(1,61):
        temp_df = coor_data[coor_data['time']==time_i]
        df_output.loc[num,'time'] = time_i
        df_output.loc[num,'average_center_ratio'] = temp_df['distance2center_rate'].mean()
        num += 1
    return(df_output)
    


dataset1_percentage_count_df_list = []
dataset1_speed_list = []
dataset1_chaos_list  = []
dataset1_center_rate_list = []
for i in dataset1.index:

    video_index = dataset1.loc[i,'video_index']
    ExperimentCondition = dataset1.loc[i,'ExperimentCondition']
    gender = dataset1.loc[i,'gender']
        
    Mov_data = pd.read_csv(Movement_Label_path[video_index])
    Mov_data.rename(columns={'locomotion_speed_smooth': 'smooth_speed'}, inplace=True)
    
    count_Min_percent_df = movement_label_count(Mov_data,1)           ### time intervals, min, 1 for 1min, 5 for 5min
    dataset1_percentage_count_df_list.append(count_Min_percent_df)    ### calculate movement fraction each n min
    
    speed_distance_df = calculate_distanceByMin(Mov_data,1)
    dataset1_speed_list.append(speed_distance_df)
    
    FeA_data = pd.read_csv(Feature_space_path[video_index])
    chaos = calculate_chaosByMin(FeA_data,1)
    dataset1_chaos_list.append(chaos)
    
    coor_data = Mov_data[['back_x','back_y']].copy()
    coor_data['back_x'] = coor_data['back_x'] - 250
    coor_data['back_y'] = coor_data['back_y'] - 250
    
    coor_data['distance2center'] = np.sqrt(np.square(coor_data['back_x'])+np.square(coor_data['back_y']))
    coor_data['distance2center_rate'] = round(coor_data['distance2center']/np.sqrt(np.square(250)*2),2)
    coor_data['flame'] = coor_data.index
    coor_data['time'] = coor_data['flame']/(30*60)
    coor_data['time'] = coor_data['time'].apply(lambda x: int(x)) + 1
    dataset1_center_rate_list.append(average_center_ratio(coor_data))

dataset1_df = pd.concat(dataset1_percentage_count_df_list,axis=0)

for mv in movement_order:
    data_name = mv + '_percent'
    temp_list = []
    for time_i in range(1,61):
        df_i = dataset1_df.loc[dataset1_df['time']==time_i,data_name]
        df_i_frame = df_i.to_frame(time_i)
        df_i_frame.reset_index(drop=True,inplace=True)
        temp_list.append(df_i_frame.T)
    df_mv_out = pd.concat(temp_list,axis=0)

    
dataset1_averageMindf = pd.concat(dataset1_percentage_count_df_list)
dataset1_speed_distance = time_Speed_Distance(dataset1_speed_list)
dataset1_chaos = time_Chaos(dataset1_chaos_list)
dataset1_center_rate = time_Center_distance_rate(dataset1_center_rate_list)



dataset2_percentage_count_df_list = []
dataset2_speed_list = []
dataset2_chaos_list  = []
dataset2_center_rate_list = []
for i in dataset2.index:

    video_index = dataset2.loc[i,'video_index']
    #ExperimentCondition = dataset2.loc[i,'ExperimentCondition']
    gender = dataset2.loc[i,'gender']
        
    Mov_data = pd.read_csv(Movement_Label_path[video_index])
    FeA_data = pd.read_csv(Feature_space_path[video_index]) 
    count_Min_percent_df = movement_label_count(Mov_data,1)           ### min, 1 for 1min, 5 for 5min
    dataset2_percentage_count_df_list.append(count_Min_percent_df)
       
    speed_distance_df = calculate_distanceByMin(Mov_data,1)
    dataset2_speed_list.append(speed_distance_df)
    
    chaos = calculate_chaosByMin(FeA_data,1)
    dataset2_chaos_list.append(chaos)
    
    coor_data = Mov_data[['back_x','back_y']].copy()
    coor_data['back_x'] = coor_data['back_x'] - 250
    coor_data['back_y'] = coor_data['back_y'] - 250
    
    coor_data['distance2center'] = np.sqrt(np.square(coor_data['back_x'])+np.square(coor_data['back_y']))
    coor_data['distance2center_rate'] = round(coor_data['distance2center']/np.sqrt(np.square(250)*2),2)
    coor_data['flame'] = coor_data.index
    coor_data['time'] = coor_data['flame']/(30*60)
    coor_data['time'] = coor_data['time'].apply(lambda x: int(x)) + 1
    dataset2_center_rate_list.append(average_center_ratio(coor_data))

dataset2_df = pd.concat(dataset2_percentage_count_df_list,axis=0)

for mv in movement_order:
    data_name = mv + '_percent'
    temp_list = []
    for time_i in range(1,61):
        df_i = dataset2_df.loc[dataset2_df['time']==time_i,data_name]
        df_i_frame = df_i.to_frame(time_i)
        df_i_frame.reset_index(drop=True,inplace=True)
        temp_list.append(df_i_frame.T)
    df_mv_out = pd.concat(temp_list,axis=0)

    
dataset2_averageMindf = pd.concat(dataset2_percentage_count_df_list)
dataset2_speed_distance = time_Speed_Distance(dataset2_speed_list)
dataset2_chaos = time_Chaos(dataset2_chaos_list)
dataset2_center_rate = time_Center_distance_rate(dataset2_center_rate_list)


plot(dataset1_averageMindf,dataset2_averageMindf,
    dataset1_speed_distance,dataset2_speed_distance,
     dataset1_chaos,dataset2_chaos,
     dataset1_center_rate,dataset2_center_rate)

plot_distance(dataset1_speed_distance,dataset2_speed_distance)
plot_chaos(dataset1_chaos,dataset2_chaos)
plot_center_rate(dataset1_center_rate,dataset2_center_rate)


# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 14:29:23 2024

@author: Jialin Ye
@institution: SIAT
@Contact_email: jl.ye@siat.ac.cn

"""
### supplymentary figure 7 region


import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import joypy
from scipy.stats import gaussian_kde
import matplotlib.ticker as ticker
from scipy.signal import find_peaks, peak_prominences
import matplotlib.patches as patches



InputData_path_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\01_BehaviorAtlas_collated_data'
Mov_file_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label'
output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure5_spatial_perference\comparison_of_density_distribution'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)  

skip_file_list = [1,3,28,29,110,122] 

def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-'))&(file_name.endswith(content)):
            USN = int(file_name.split('-')[1])
            file_path_dict.setdefault(USN,file_dir+'\\'+file_name)
    return(file_path_dict)


coordinates_file_path = get_path(InputData_path_dir,'normalized_coordinates_back_XY.csv')
Mov_file_path = get_path(Mov_file_dir,'revised_Movement_Labels.csv')

animal_info_csv =r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Table_S1_animal_information.csv'               
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


movement_order = ['running','trotting','walking','right_turning','left_turning','stepping',
                  'rising','hunching','rearing','climbing','jumping','sniffing','grooming','scratching','pausing',]


dataset = Morning_lightOn_info
dataset_name = 'Morning_lightOn'

dataset2 = stress_animal_info
dataset_name2 = 'stress_animal'

def random_rotation(grid_counts,sample_num):
    direction = [0,1,2,3]
    empty_matrix = np.zeros((grid_counts.shape[0],grid_counts.shape[1]))
    for i in range(sample_num):
        rcount = np.rot90(grid_counts,random.sample(direction,1)[0])
        empty_matrix += rcount
    average_matrix = empty_matrix/sample_num
    return(average_matrix)


def grid_count(coor_data,boundary,hist_bin):    
    set_bin = hist_bin
    step = 500/set_bin

    x = coor_data['back_x']
    y = coor_data['back_y']
    
    #fig,ax = plt.subplots(figsize=(10,10),dpi=300)
    grid_counts,x_edges,y_edges = np.histogram2d (x=x, y=y, bins=set_bin,range=[[0, 500], [0, 500]],)
    cvmin = grid_counts.min()
    cmax = grid_counts.max()
    rgrid_counts = random_rotation(grid_counts,1000)


    center = rgrid_counts[int((250-boundary)/step):int((250+boundary)/step),int((250-boundary)/step):int((250+boundary)/step)]
    center_std = center.std()/center.mean() #/Dcenter.mean()
    #print(Dcenter)    
    perimeter1 = rgrid_counts[:int((250-boundary)/step),int((250-boundary)/step):int((250+boundary)/step)]
    perimeter1_std = perimeter1.std()
    perimeter2 = rgrid_counts[int((250+boundary)/step):,int((250-boundary)/step):int((250+boundary)/step)]
    perimeter2_std = perimeter2.std()
    perimeter3 = rgrid_counts[int((250-boundary)/step):int((250+boundary)/step),:int((250-boundary)/step)]
    perimeter3_std = perimeter3.std()
    perimeter4 = rgrid_counts[int((250-boundary)/step):int((250+boundary)/step),int((250+boundary)/step):]
    perimeter4_std = perimeter4.std()
    perimeter = np.concatenate([perimeter1+perimeter2+perimeter3.T+perimeter4.T]).ravel()
    perimeter = perimeter[perimeter>0]
    perimeter_std = perimeter.std()/perimeter.mean()

    
    corner1 = rgrid_counts[:int((250-boundary)/step),:int((250-boundary)/step)]
    corner1_std = corner1.std()
    corner2 = rgrid_counts[int((250+boundary)/step):,int((250+boundary)/step):]
    corner2_std = corner2.std()
    corner3 = rgrid_counts[:int((250-boundary)/step),int((250+boundary)/step):]
    corner3_std = corner3.std()
    corner4 = rgrid_counts[int((250+boundary)/step):,:int((250-boundary)/step)]
    corner4_std = corner4.std()
    corner = np.concatenate([corner1+corner2+corner3.T+corner4.T]).ravel()
    corner_std = corner.std()/corner.mean()
    
    info_dict = {'location':[],'density_range':[],'boundary':[]}
    
# =============================================================================
#     info_dict['location'].append('center')
#     info_dict['density_rangeity'].append(Tcenter_std)
#     info_dict['method'].append('traditional')
#     
#     info_dict['location'].append('center')
#     info_dict['density_rangeity'].append(Dcenter_std)
#     info_dict['method'].append('data_driven')
#     
#     info_dict['location'].append('perimeter')
#     info_dict['density_rangeity'].append(Tperimeter_std)
#     info_dict['method'].append('traditional')
#     
#     info_dict['location'].append('perimeter')
#     info_dict['density_rangeity'].append(Dperimeter_std)
#     info_dict['method'].append('data_driven')
#     
#     info_dict['location'].append('corner')
#     info_dict['density_rangeity'].append(Tcorner_std)
#     info_dict['method'].append('traditional')
#     
#     info_dict['location'].append('corner')
#     info_dict['density_rangeity'].append(Dcorner_std)
#     info_dict['method'].append('data_driven')
# =============================================================================
    
    
    for i2 in center.ravel():
        info_dict['location'].append('center')
        info_dict['density_range'].append(i2)
        info_dict['boundary'].append(boundary)

    for i4 in perimeter:
        info_dict['location'].append('perimeter')
        info_dict['density_range'].append(i4)
        info_dict['boundary'].append(boundary)
    
    for i6 in corner:
        info_dict['location'].append('corner')
        info_dict['density_range'].append(i6)
        info_dict['boundary'].append(boundary)
    
    df_density_range = pd.DataFrame(info_dict)  
    return(rgrid_counts,df_density_range)


def cal_grid_density(dataset,boundary,hist_bin,start=0,end=60):
    
    coor_data_list = []
    grid_counts_list = []
    df_density_range_list = []
    for index in dataset.index:
        video_index = dataset.loc[index,'video_index']
        gender = dataset.loc[index,'gender']
        ExperimentCondition = dataset.loc[index,'ExperimentCondition']
        Mov_data = pd.read_csv(Mov_file_path[video_index],usecols=['revised_movement_label'])
        coor_data = pd.read_csv(coordinates_file_path[video_index],index_col=0)
        coor_data = coor_data.iloc[start*30*60:end*30*60,:]
        
        grid_counts,df_density_range = grid_count(coor_data,boundary,hist_bin)
        df_density_range_list.append(df_density_range)
        coor_data_list.append(coor_data)
        grid_counts_list.append(grid_counts)
        
    coor_data_all = pd.concat(coor_data_list,axis=0)
    coor_data_all.reset_index(drop=True,inplace=True)
    
    density_range_data_all = pd.concat(df_density_range_list,axis=0)
    density_range_data_all.reset_index(drop=True,inplace=True)
    
    grid_counts,x_edges,y_edges = np.histogram2d (x=coor_data_all['back_x'], y=coor_data_all['back_y'], bins=hist_bin,range=[[0, 500], [0, 500]],)
    grid_counts = grid_counts/len(dataset)
    
    cvmin = grid_counts.min()
    cmax = grid_counts.max()
    
    merge_density_grid = np.zeros((grid_counts_list[0].shape[0],grid_counts_list[0].shape[1]))
    for matrix in grid_counts_list:
        merge_density_grid += matrix
    merge_density_grid = merge_density_grid/len(dataset)
    return(density_range_data_all,merge_density_grid)



# derivation
def cal_deriv(x, y):                  
    diff_x = []                       
    for i, j in zip(x[0::], x[1::]):  
        diff_x.append(j - i)
 
    diff_y = []                       
    for i, j in zip(y[0::], y[1::]):
        diff_y.append(j - i)  
        
    slopes = []                       
    for i in range(len(diff_y)):
        slopes.append(diff_y[i] / diff_x[i])
        
    deriv = []                       
    for i, j in zip(slopes[0::], slopes[1::]):        
        deriv.append((0.5 * (i + j)))
    deriv.insert(0, slopes[0])       
    deriv.append(slopes[-1])          
 
    #for i in deriv:                  
    #    print(i)
    return(deriv)                      

def find_startpoint2(S_list,deriv_density_std,start_point_height):
    
    for i in range(1,len(S_list)):
        x1 = i-1
        x2 = i
        if i > 10: 
            y1 = deriv_density_std[i-1]
            y2 = deriv_density_std[i]
            
            if (y1 <=  start_point_height) &  (y2 >=  start_point_height):
                return(x1)

def plot_increasing_density(merge_density_grid):
    
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,10),dpi=300,constrained_layout=True,sharex=True)
    im = ax.imshow(merge_density_grid,cmap=plt.cm.jet)
    ax.set_axis_off()
    plt.savefig('{}/{}_grid_heatmap_{}_{}.png'.format(output_dir,dataset_name,start,end),transparent=True,dpi=300)
    
    a = 50
    density_std = []
    density_std2 = []
    density_average = []
    density_average2 = []
    S_list = []
    for S in range(1,50):
        S1 = a-S
        S2 = a+S
        count1 = merge_density_grid[S1:S2,S1:S2]
        density_std.append(count1.std())
        density_average.append(count1.sum()/(S1*S2))
        S_list.append(S)
    
    density_average = density_average / np.max(density_average)
    deriv_density_std = cal_deriv(S_list, density_std)
        
    
    peak_index = np.argmax(deriv_density_std)
    #initial_guess = [np.max(deriv_density_std), S_list[peak_index], 1.0]
    #params, covariance = curve_fit(gaussian, S_list, deriv_density_std, p0=initial_guess)
    #y_gaussian = gaussian(S_list, *params) 
    #peak_width = np.abs(params[2])*3
    #startpoint = find_startpoint(S_list,params[1]-peak_width)    
    startpoint = find_startpoint2(S_list,deriv_density_std,0.10*(deriv_density_std[peak_index]))

    fig,ax = plt.subplots(ncols=1,nrows=2,figsize = (11,8),dpi=300)
    ax[0].plot(S_list,density_std,c='#F4511E',lw = 6)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['bottom'].set_linewidth(5)
    ax[0].spines['left'].set_linewidth(5)
    ax[0].set_yticks(np.arange(0,25,12))
    ax[0].set_title('density variance',fontsize=15)
    #ax[0].xaxis.set_major_formatter(plt.NullFormatter())
    #ax[0].yaxis.set_major_formatter(plt.NullFormatter())
    ax[0].tick_params(length=8,width=4)
    
    
    ax[1].plot(S_list,deriv_density_std,c='#0288D1',lw = 6,zorder=3)
    #ax[1].plot(S_list,y_gaussian,c='#F06292',lw = 6,zorder=3)
    ax[1].fill_between(S_list,y1=0,y2=deriv_density_std,color='#80DEEA',alpha=0.8,zorder=0)
    
    ax[1].plot([startpoint,startpoint],[0,deriv_density_std[startpoint]],color='#E91E63',lw=4,zorder=3)
    
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['bottom'].set_linewidth(5)
    ax[1].spines['left'].set_linewidth(5)
    ax[1].set_yticks(np.arange(0,5,2))
    ax[1].set_title('change rate of density variance',fontsize=15)
    #ax[1].xaxis.set_major_formatter(plt.NullFormatter())
    #ax[1].yaxis.set_major_formatter(plt.NullFormatter())
    ax[1].tick_params(length=8,width=4)
    #ax[1].axvline(params[1]-peak_width,color='green',lw=2)
    
    fig.tight_layout(pad=5)
    plt.savefig('{}/boundary_identification_{}_{}.png'.format(output_dir,start,end),transparent=True,dpi=300)
    
    

fix_boundary = 35
test_boundary = 25

boundary = test_boundary *5
hist_bin = 100
start = 0
end = 60

density_range_data_all_175,merge_density_grid_175 = cal_grid_density(dataset,boundary=175,hist_bin=100,start=0,end=60)

plot_increasing_density(merge_density_grid_175) 
density_range_data_all_125,merge_density_grid_125 = cal_grid_density(dataset,boundary=125,hist_bin=100,start=0,end=60)
density_range_data_all_150,merge_density_grid_150 = cal_grid_density(dataset,boundary=150,hist_bin=100,start=0,end=60)
density_range_data_all_200,merge_density_grid_200 = cal_grid_density(dataset,boundary=200,hist_bin=100,start=0,end=60)







def plot_supply(merge_density_grid,hist_bin=100):
    fix_boundary = 175
    for boundary in [125,150,200]:
        fig,ax = plt.subplots(ncols=1,nrows=1,figsize = (10,10),dpi=300)

        ax.imshow(merge_density_grid,cmap=plt.cm.jet)
        
        ax.axvline(float((250-fix_boundary)/(500/hist_bin)),c='#32B154',lw=10)
        ax.axvline(float((250+fix_boundary)/(500/hist_bin)),c='#32B154',lw=10)                            
        ax.axhline(float((250-fix_boundary)/(500/hist_bin)),c='#32B154',lw=10)
        ax.axhline(float((250+fix_boundary)/(500/hist_bin)),c='#32B154',lw=10)
        
        ax.axvline(float((250-boundary)/(500/hist_bin)),c='#717070',lw=6,linestyle='--')
        ax.axvline(float((250+boundary)/(500/hist_bin)),c='#717070',lw=6,linestyle='--')
        ax.axhline(float((250-boundary)/(500/hist_bin)),c='#717070',lw=6,linestyle='--')
        ax.axhline(float((250+boundary)/(500/hist_bin)),c='#717070',lw=6,linestyle='--')
        
        ax.set_axis_off()
        plt.savefig(r'{}\{}_{}_{}min_compareTo_{}mm.png'.format(output_dir,dataset_name,start,end,boundary),dpi=300,transparent=True)

plot_supply(merge_density_grid_175,hist_bin=100)    
    
    
def get_nor_kde_value(arr):  
    xx = np.linspace(0,1000,1000)
    kde = gaussian_kde(arr)
    yy = kde(xx)
    return(xx,yy)


def plot_AreaDensity_std(density_range_data_all_fix,density_range_data_all_test,boundary):
    location_order = ['center','perimeter','corner']
    fig,ax = plt.subplots(ncols=1,nrows=3,figsize=(10,12),dpi=300,sharex=False)
    #center_density_range_data_all = density_range_data_all[density_range_data_all['location']=='center']
    CenterFix_density_range = density_range_data_all_fix.loc[density_range_data_all_fix['location']=='center','density_range']
    CenterTest_density_range = density_range_data_all_test.loc[density_range_data_all_test['location']=='center','density_range']
    #sns.kdeplot(data=center_density_range_data_all,x='density_range',hue='method',ax=ax[0],common_norm=False,)
    CenterFixxx,CenterFixyy = get_nor_kde_value(CenterFix_density_range)
    CenterTestxx,CenterTestyy = get_nor_kde_value(CenterTest_density_range)
    ax[0].plot(CenterFixxx, CenterFixyy,alpha=1,lw=8,color='#32B154',zorder=1)
    ax[0].fill_between(CenterFixxx, y1=0,y2=CenterFixyy,alpha=0.2,color='#32B154',zorder=0)
    ax[0].plot(CenterTestxx, CenterTestyy,alpha=1,lw=8,color='#717070',zorder=2)
    ax[0].fill_between(CenterTestxx, y1=0,y2=CenterTestyy,alpha=0.2,color='#717070',zorder=0)
    #ax[0].set_title('center')
    ax[0].set_xlim(0,30)
    ax[0].set_ylim(0,0.4)
    ax[0].set_xticks(np.arange(0,31,5))
    ax[0].set_yticks(np.arange(0,0.41,0.2))
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['bottom'].set_linewidth(6)
    ax[0].spines['left'].set_linewidth(6)
    ax[0].tick_params(length=8,width=4)
    ax[0].set_title('center',fontsize=15)
    #ax[0].xaxis.set_major_formatter(plt.NullFormatter())
    #ax[0].yaxis.set_major_formatter(plt.NullFormatter())
    
    PerimeterFix_density_range = density_range_data_all_fix.loc[density_range_data_all_fix['location']=='perimeter','density_range']
    PerimeterTest_density_range = density_range_data_all_test.loc[density_range_data_all_test['location']=='perimeter','density_range']
    #sns.kdeplot(data=center_density_range_data_all,x='density_range',hue='method',ax=ax[0],common_norm=False,)
    PFixxx,PFixyy = get_nor_kde_value(PerimeterFix_density_range)
    PTestxx,PTestyy = get_nor_kde_value(PerimeterTest_density_range)
    ax[1].plot(PFixxx, PFixyy,alpha=1,lw=8,color='#32B154',zorder=1)
    ax[1].fill_between(PFixxx, y1=0,y2=PFixyy,alpha=0.2,color='#32B154',zorder=0)
    ax[1].plot(PTestxx, PTestyy,alpha=1,lw=8,color='#717070',zorder=2)
    ax[1].fill_between(PTestxx, y1=0,y2=PTestyy,alpha=0.2,color='#717070',zorder=0)
    #ax[1].set_title('perimeter')
    ax[1].set_xlim(0,180)
    ax[1].set_ylim(0,0.04)
    ax[1].set_xticks(np.arange(0,181,30))
    ax[1].set_yticks(np.arange(0,0.041,0.02))
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['bottom'].set_linewidth(6)
    ax[1].spines['left'].set_linewidth(6)
    ax[1].tick_params(length=8,width=4)
    ax[1].set_title('perimeter',fontsize=15)
    #ax[1].xaxis.set_major_formatter(plt.NullFormatter())
    #ax[1].yaxis.set_major_formatter(plt.NullFormatter())
    
    CornerFix_density_range = density_range_data_all_fix.loc[density_range_data_all_fix['location']=='corner','density_range']
    CornerTest_density_range = density_range_data_all_test.loc[density_range_data_all_test['location']=='corner','density_range']
    #sns.kdeplot(data=center_density_range_data_all,x='density_range',hue='method',ax=ax[0],common_norm=False,)
    CornerFixxx,CornerFixyy = get_nor_kde_value(CornerFix_density_range)
    CornerTestxx,CornerTestyy = get_nor_kde_value(CornerTest_density_range)
    
    ax[2].plot(CornerFixxx, CornerFixyy,alpha=1,lw=8,color='#32B154',zorder=1)
    ax[2].fill_between(CornerFixxx, y1=0,y2=CornerFixyy,alpha=0.2,color='#32B154',zorder=0)
    ax[2].plot(CornerTestxx, CornerTestyy,alpha=1,lw=8,color='#717070',zorder=2)
    ax[2].fill_between(CornerTestxx, y1=0,y2=CornerTestyy,alpha=0.2,color='#717070',zorder=0)
    
    #ax[2].set_title('corner')
    ax[2].set_xlim(0,900)
    ax[2].set_ylim(0,0.012)
    ax[2].set_xticks(np.arange(0,901,150))
    ax[2].set_yticks(np.arange(0,0.013,0.006))
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['right'].set_visible(False)
    ax[2].spines['bottom'].set_linewidth(6)
    ax[2].spines['left'].set_linewidth(6)
    ax[2].tick_params(length=8,width=4)
    ax[2].set_title('corner',fontsize=15)
    #ax[2].xaxis.set_major_formatter(plt.NullFormatter())
    #ax[2].yaxis.set_major_formatter(plt.NullFormatter())
    fig.tight_layout(pad=8)
    
    plt.savefig(r'{}\{}_{}_{}min_175&{}mm_density_distribution_comparision.png'.format(output_dir,dataset_name,start,end,boundary),dpi=300,transparent=True)
    
plot_AreaDensity_std(density_range_data_all_175,density_range_data_all_125,125)
plot_AreaDensity_std(density_range_data_all_175,density_range_data_all_150,150)
plot_AreaDensity_std(density_range_data_all_175,density_range_data_all_200,200)




























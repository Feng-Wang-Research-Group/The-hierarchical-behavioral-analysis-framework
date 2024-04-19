# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:33:09 2024

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
import random
import joypy
from scipy.stats import gaussian_kde
import matplotlib.ticker as ticker
from scipy.signal import find_peaks, peak_prominences
from scipy.optimize import curve_fit



InputData_path_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\01_BehaviorAtlas_collated_data'
Mov_file_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label'
output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure5_spatial_perference\grid_denstity'


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

    
    Tcenter = rgrid_counts[int(125/step):int(375/step),int(125/step):int(375/step)]                                                 ###### calculate density in each cell within each region
    Tcenter_std = Tcenter.std()/Tcenter.mean()
    
    Tperimeter1 = rgrid_counts[:int(125/step),int(125/step):int(375/step)]
    Tperimeter1_std = Tperimeter1.std()
    Tperimeter2 = rgrid_counts[int(375/step):,int(125/step):int(375/step)]
    Tperimeter2_std = Tperimeter2.std()
    Tperimeter3 = rgrid_counts[int(125/step):int(375/step),:int(125/step)]
    Tperimeter3_std = Tperimeter3.std()
    Tperimeter4 = rgrid_counts[int(125/step):int(375/step),int(375/step):]
    Tperimeter4_std = Tperimeter4.std()
    Tperimeter = np.concatenate([Tperimeter1+Tperimeter2+Tperimeter3.T+Tperimeter4.T]).ravel()
    Tperimeter_std = Tperimeter.std()/Tperimeter.mean()

    
    Tcorner1 = rgrid_counts[:int(125/step),:int(125/step)]
    Tcorner1_std = Tcorner1.std()
    Tcorner2 = rgrid_counts[int(375/step):,int(375/step):]
    Tcorner2_std = Tcorner2.std()
    Tcorner3 = rgrid_counts[:int(125/step),int(375/step):]
    Tcorner3_std = Tcorner3.std()
    Tcorner4 = rgrid_counts[int(375/step):,:int(125/step)]
    Tcorner4_std = Tcorner4.std()
    Tcorner = np.concatenate([Tcorner1+Tcorner2+Tcorner3.T+Tcorner4.T]).ravel()
    Tcorner_std = Tcorner.std()/Tcorner.mean()

    Dcenter = rgrid_counts[int((250-boundary)/step):int((250+boundary)/step),int((250-boundary)/step):int((250+boundary)/step)]
    Dcenter_std = Dcenter.std()/Dcenter.mean() #/Dcenter.mean()
    #print(Dcenter)
    
    Dperimeter1 = rgrid_counts[:int((250-boundary)/step),int((250-boundary)/step):int((250+boundary)/step)]
    Dperimeter1_std = Dperimeter1.std()
    Dperimeter2 = rgrid_counts[int((250+boundary)/step):,int((250-boundary)/step):int((250+boundary)/step)]
    Dperimeter2_std = Dperimeter2.std()
    Dperimeter3 = rgrid_counts[int((250-boundary)/step):int((250+boundary)/step),:int((250-boundary)/step)]
    Dperimeter3_std = Dperimeter3.std()
    Dperimeter4 = rgrid_counts[int((250-boundary)/step):int((250+boundary)/step),int((250+boundary)/step):]
    Dperimeter4_std = Dperimeter4.std()
    Dperimeter = np.concatenate([Dperimeter1+Dperimeter2+Dperimeter3.T+Dperimeter4.T]).ravel()
    Dperimeter = Dperimeter[Dperimeter>0]
    Dperimeter_std = Dperimeter.std()/Dperimeter.mean()

    
    Dcorner1 = rgrid_counts[:int((250-boundary)/step),:int((250-boundary)/step)]
    Dcorner1_std = Dcorner1.std()
    Dcorner2 = rgrid_counts[int((250+boundary)/step):,int((250+boundary)/step):]
    Dcorner2_std = Dcorner2.std()
    Dcorner3 = rgrid_counts[:int((250-boundary)/step),int((250+boundary)/step):]
    Dcorner3_std = Dcorner3.std()
    Dcorner4 = rgrid_counts[int((250+boundary)/step):,:int((250-boundary)/step)]
    Dcorner4_std = Dcorner4.std()
    Dcorner = np.concatenate([Dcorner1+Dcorner2+Dcorner3.T+Dcorner4.T]).ravel()
    Dcorner_std = Dcorner.std()/Dcorner.mean()
    
    info_dict = {'location':[],'density_range':[],'method':[]}
    
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
    
    for i1 in Tcenter.ravel():   
        info_dict['location'].append('center')
        info_dict['density_range'].append(i1)
        info_dict['method'].append('traditional')
    
    for i2 in Dcenter.ravel():
        info_dict['location'].append('center')
        info_dict['density_range'].append(i2)
        info_dict['method'].append('data_driven')
    
    for i3 in Tperimeter:
        info_dict['location'].append('perimeter')
        info_dict['density_range'].append(i3)
        info_dict['method'].append('traditional')
    for i4 in Dperimeter:
        info_dict['location'].append('perimeter')
        info_dict['density_range'].append(i4)
        info_dict['method'].append('data_driven')
    
    for i5 in Tcorner:
        info_dict['location'].append('corner')
        info_dict['density_range'].append(i5)
        info_dict['method'].append('traditional')
    for i6 in Dcorner:
        info_dict['location'].append('corner')
        info_dict['density_range'].append(i6)
        info_dict['method'].append('data_driven')
    
    
    df_density_range = pd.DataFrame(info_dict)  
    return(rgrid_counts,df_density_range)


def plot_demo(coor_data_list,hist_bin):
    sample_index = random.sample(Morning_lightOn_info['video_index'].values.tolist(), 1)[0]
    coor_data = pd.read_csv(coordinates_file_path[sample_index],index_col=0)
       
    boundary = 50
    
    coor_data = add_location(coor_data,boundary)
    x = coor_data['back_x']
    y = coor_data['back_y']
    
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,10),dpi=300)
    
    sns.scatterplot(data=coor_data,x='back_x',y='back_y', edgecolor="none", s=10, hue='temp_location',hue_order=['center','perimeter'],
                     legend=False,palette=['#2C7EC2','#616161'],alpha = 1, ax=ax,zorder=1) ##009688  #BDBDBD
    
    ax.plot([0,0],[0,500],color='black',lw=8,zorder=2)
    ax.plot([0,500],[500,500],color='black',lw=8,zorder=2)
    ax.plot([500,500],[500,0],color='black',lw=8,zorder=2)
    ax.plot([500,0],[0,0],color='black',lw=8,zorder=2)
    
    
    #for line_i in range(5,500,5):        
    #    ax.plot([line_i,line_i],[0,500],color='black',lw=0.5,zorder=10)
    #    ax.plot([0,500],[line_i,line_i],color='black',lw=0.5,zorder=10)
    
     
    ax.plot([(250-boundary),(250-boundary)],[(250-boundary),(250+boundary)],color='#2C7EC2',lw=8,zorder=2) # tranditional defined center  (x1,x2)(y1,y2)  #2C7EC2
    ax.plot([(250+boundary),(250+boundary)],[(250-boundary),(250+boundary)],color='#2C7EC2',lw=8,zorder=2)
    ax.plot([(250-boundary),(250+boundary)],[(250-boundary),(250-boundary)],color='#2C7EC2',lw=8,zorder=2)
    ax.plot([(250-boundary),(250+boundary)],[(250+boundary),(250+boundary)],color='#2C7EC2',lw=8,zorder=2)
    
    plt.axis('off')
    
    plt.savefig('{}/demo/demo_Point_distribution.png'.format(output_dir),transparent=True,dpi=300)
        
    Inboundary = coor_data[coor_data['temp_location']=='center']
    x_In = Inboundary['back_x']
    y_In = Inboundary['back_y']
    
    fig2,ax2 = plt.subplots(nrows=1,ncols=1,figsize=(10,10),dpi=300)
    sns.scatterplot(data=Inboundary,x='back_x',y='back_y', edgecolor="none", s=25, color='black',alpha = 1, ax=ax2,zorder=1) ##009688  #BDBDBD
    
    ax2.plot([(250-boundary),(250-boundary)],[(250-boundary),(250+boundary)],color='#2C7EC2',lw=8,zorder=2)
    ax2.plot([(250+boundary),(250+boundary)],[(250-boundary),(250+boundary)],color='#2C7EC2',lw=8,zorder=2)
    ax2.plot([(250-boundary),(250+boundary)],[(250-boundary),(250-boundary)],color='#2C7EC2',lw=8,zorder=2)
    ax2.plot([(250-boundary),(250+boundary)],[(250+boundary),(250+boundary)],color='#2C7EC2',lw=8,zorder=2)
    
    for line_i in range((250-boundary),(250+boundary+1),5):        
        ax2.plot([line_i,line_i],[(250-boundary),(250+boundary)],color='black',lw=3,zorder=10)
        ax2.plot([(250-boundary),(250+boundary)],[line_i,line_i],color='black',lw=3,zorder=10)
    
    plt.axis('off')
    plt.savefig('{}/demo/demo_selectData_withGrid.png'.format(output_dir),transparent=True,dpi=300)
    
    fig3,ax3 = plt.subplots(nrows=1,ncols=1,figsize=(10,10),dpi=300)
    grid_counts,x_edges,x_edges = np.histogram2d (x=x_In, y=y_In, bins=20,range=[[250-boundary, 250+boundary], [250-boundary, 250+boundary]],)
    cvmin = grid_counts.min()
    cmax = grid_counts.max()
    
    ax3.imshow(grid_counts,cmap=plt.cm.Spectral_r)
    
    for axis in ['top','bottom','left','right']:
        ax3.spines[axis].set_linewidth(4)
    ax3.xaxis.set_major_formatter(plt.NullFormatter())
    ax3.yaxis.set_major_formatter(plt.NullFormatter())
    ax3.tick_params(width=0,length=0)
    #plt.axis('off')
    plt.savefig('{}/demo/demo_selectData_heatmap.png'.format(output_dir),transparent=True,dpi=300)




def cal_grid_density(dataset,boundary,hist_bin,start,end):    
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
        #combine_data = pd.concat([Mov_data,coor_data],axis=1)
        df_density_range_list.append(df_density_range)
        coor_data_list.append(coor_data)
        grid_counts_list.append(grid_counts)
    
     
    plot_demo(coor_data_list,hist_bin)
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
    return(coor_data_all,density_range_data_all,grid_counts,merge_density_grid)

def add_location(loc_data,boundary):
    loc_data_copy = loc_data.copy()
    loc_data_copy['temp_location'] = 'perimeter'   
    loc_data_copy.loc[(loc_data_copy['back_x']>(250-boundary))&(loc_data_copy['back_x']<(250+boundary))&(loc_data_copy['back_y']>(250-boundary))&(loc_data_copy['back_y']<(250+boundary)),'temp_location'] = 'center'
    return(loc_data_copy)


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

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev)**2 / 2)

def find_startpoint(S_list,deriv_density_std,start_point_height):
    for i in range(1,len(S_list)):
        x1 = i-1
        x2 = i
        if i > 10: 
            y1 = deriv_density_std[i-1]
            y2 = deriv_density_std[i]
            
            if (y1 <=  start_point_height) &  (y2 >=  start_point_height):
                return(x1)

def plot_increasing_density(merge_density_grid,merge_density_grid2,boundary,hist_bin,start,end):
    merge_density_grid = merge_density_grid
    merge_density_grid2 = merge_density_grid2
    #print(merge_density_grid.sum(),merge_density_grid2.sum())
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
        count2 = merge_density_grid2[S1:S2,S1:S2]
        density_std.append(count1.std())
        density_std2.append(count2.std())
        density_average.append(count1.sum()/(S1*S2))
        density_average2.append(count2.sum()/(S1*S2))
        S_list.append(S)
    
    density_average = density_average / np.max(density_average)
    density_average2 =  density_average2 / np.max(density_average2)
    deriv_density_std = cal_deriv(S_list, density_std)
    deriv_density_std2 = cal_deriv(S_list, density_std2)
        
    
    peak_index = np.argmax(deriv_density_std)
    startpoint = find_startpoint(S_list,deriv_density_std,0.10*(deriv_density_std[peak_index]))

    peak_index2 = np.argmax(deriv_density_std2)
    startpoint2 = find_startpoint(S_list,deriv_density_std2,0.10*(deriv_density_std2[peak_index2]))
    
    #print(S_list[startpoint]*5)
    #print(S_list[startpoint2]*5)
    
    fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(8,10),dpi=300,constrained_layout=True,sharex=True)
    ax[0].plot(S_list,density_average,c='#F5B25E',lw = 5,alpha=0.8)
    ax[0].scatter(S_list[startpoint],density_average[startpoint],marker='^',c='#F9A825',ec='black',s=200,zorder=3)
    ax[0].plot(S_list,density_average2,c='#E75C6B',lw = 5,alpha=0.8)
    ax[0].scatter(S_list[startpoint2],density_average2[startpoint2],marker='^',c='red',ec='black',s=200,zorder=3)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['bottom'].set_linewidth(3)
    ax[0].spines['left'].set_linewidth(3)
    #ax[0].set_yscale('log')
    ax[0].set_ylim(-0.1,1.1)
    ax[0].set_yticks(np.arange(0,1.1,0.5))
    ax[0].set_title('Density (in grid), {}-{}min'.format(start,end))
    #ax[0].xaxis.set_major_formatter(plt.NullFormatter())
    #ax[0].yaxis.set_major_formatter(plt.NullFormatter())
    ax[0].tick_params(length=7,width=3)
    
    
    ax[1].plot(S_list,density_std,c='#F5B25E',lw = 5,alpha=0.8)
    ax[1].scatter(S_list[startpoint],density_std[startpoint],marker='^',c='#F9A825',ec='black',s=200,zorder=3)
    ax[1].plot(S_list,density_std2,c='#E75C6B',lw = 5,alpha=0.8)
    ax[1].scatter(S_list[startpoint2],density_std2[startpoint2],marker='^',c='red',ec='black',s=200,zorder=3)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['bottom'].set_linewidth(3)
    ax[1].spines['left'].set_linewidth(3)
    if (start == 0) & (end == 10):
        ax[1].set_ylim(-0.5,5.5)
        ax[1].set_yticks(np.arange(0,5.1,2.5))
    elif (start == 10) & (end == 60):
        ax[1].set_ylim(-2,22)
        ax[1].set_yticks(np.arange(0,21,10))
    #ax[1].xaxis.set_major_formatter(plt.NullFormatter())
    #ax[1].yaxis.set_major_formatter(plt.NullFormatter())
    ax[1].set_title('Growth rate of density, {}-{}min'.format(start,end))
    ax[1].tick_params(length=7,width=3)
    
    ax[2].plot(S_list,deriv_density_std,c='#F5B25E',lw = 5,alpha=0.8)
    ax[2].scatter(S_list[startpoint],deriv_density_std[startpoint],marker='o',c='#EA5514',ec='black',s=200,zorder=3)
    ax[2].plot(S_list,deriv_density_std2,c='#E75C6B',lw = 5,alpha=0.8)
    ax[2].scatter(S_list[startpoint2],deriv_density_std2[startpoint2],marker='o',c='red',ec='black',s=200,zorder=3)
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['right'].set_visible(False)
    ax[2].spines['bottom'].set_linewidth(3)
    ax[2].spines['left'].set_linewidth(3)
    if (start == 0) & (end == 10):
        ax[2].set_ylim(-0.1,0.9)
        ax[2].set_yticks(np.arange(0,0.81,0.4))
    elif (start == 10) & (end == 60):
        ax[2].set_ylim(-0.5,5.5)
        ax[2].set_yticks(np.arange(0,5.1,2.5))
    #ax[2].xaxis.set_major_formatter(plt.NullFormatter())
    #ax[2].yaxis.set_major_formatter(plt.NullFormatter())
    ax[2].set_title('Derivative of growth rate, {}-{}min'.format(start,end))
    ax[2].tick_params(length=7,width=3)
        
    fig.tight_layout(pad=5.0)
    plt.savefig('{}/boundary_{}_{}.png'.format(output_dir,start,end),transparent=True,dpi=300)
    


def plot_density(merge_density_grid,dataset_name,vmin,vmax,boundary,hist_bin,start,end):
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize = (5,5),dpi=300)
    im = ax.imshow(merge_density_grid,cmap=plt.cm.jet,vmin=vmin,vmax=vmax)
    
    #cb = plt.colorbar(im,shrink=0.9, aspect=10, pad=0.05)                          #show colorbar
    #tick_locator = ticker.MaxNLocator(nbins=3)                                
    #cb.locator = tick_locator
    #cb.set_ticks([vmin, (vmin+vmax)/2, vmax])
    #cb.update_ticks()

    ax.axvline(float((250-boundary)/(500/hist_bin)),c='#00C853',lw=4)
    ax.axvline(float((250+boundary)/(500/hist_bin)),c='#00C853',lw=4)
    ax.axhline(float((250-boundary)/(500/hist_bin)),c='#00C853',lw=4)
    ax.axhline(float((250+boundary)/(500/hist_bin)),c='#00C853',lw=4)
    
    ax.plot((float(125/(500/hist_bin)),float(375/(500/hist_bin))),(float(125/(500/hist_bin)),float(125/(500/hist_bin))),linestyle = '--',c='#78909C',lw=4)
    ax.plot((float(125/(500/hist_bin)),float(375/(500/hist_bin))),(float(375/(500/hist_bin)),float(375/(500/hist_bin))),linestyle = '--',c='#78909C',lw=4)
    
    ax.plot((float(125/(500/hist_bin)),float(125/(500/hist_bin))),(float(125/(500/hist_bin)),float(375/(500/hist_bin))),linestyle = '--',c='#78909C',lw=4)
    ax.plot((float(375/(500/hist_bin)),float(375/(500/hist_bin))),(float(125/(500/hist_bin)),float(375/(500/hist_bin))),linestyle = '--',c='#78909C',lw=4)
    
    plt.axis('off')
    plt.savefig(r'{}\{}_{}_{}_stimulate_density.png'.format(output_dir,dataset_name,start,end),transparent=True,dpi=300)



def main():

    boundary = 175
    hist_bin = 100
    start = 0
    end = 10
    
    dataset1_coor_data,dataset1_density_range_data,dataset1_grid_count,dataset1_grid_count_random = cal_grid_density(dataset,boundary,hist_bin,start,end)
    dataset2_coor_data,dataset2_density_range_data,dataset2_grid_count,dataset2_grid_count_random = cal_grid_density(dataset2,boundary,hist_bin,start,end)
    
    
    plot_increasing_density(dataset1_grid_count_random,dataset2_grid_count_random,boundary,hist_bin,start,end)
    
    
    if (start == 10)&(end==60):
        vmin = 0
        vmax = 120
    elif (start == 0)&(end==10) :
        vmin = 0
        vmax = 120
    
    plot_density(dataset1_grid_count_random,'Morning',vmin,vmax,boundary,hist_bin,start,end)
    plot_density(dataset2_grid_count_random,'Stress',vmin,vmax,boundary,hist_bin,start,end)
    
    
    print(dataset2_grid_count_random.sum()/30/60)



# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 12:54:18 2023

@author: kanishkbjain
@original code: https://github.com/bermanlabemory/motionmapperpy
@modification: Jialin Ye SIAT Contact_email: jl.ye@siat.ac.cn

"""

import motionmapperpy as mmpy
from motionmapperpy import demoutils
import re

# Python standard library packages to do file/folder manipulations,
# pickle is a package to store python variables
import glob, os, pickle, sys

# time grabs current clock time and copy to safely make copies of large 
# variables in memory.
import time, copy 

# datetime package is used to get and manipulate date and time data
from datetime import datetime

# this packages helps load and save .mat files older than v7
import hdf5storage 

# numpy works with arrays, pandas used to work with fancy numpy arrays
import numpy as np
import pandas as pd

# matplotlib is used to plot and animate to make movies
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Scikit-learn is a go-to library in Python for all things machine learning
from sklearn.decomposition import PCA

# tqdm helps create progress bars in for loops 
from tqdm import tqdm 

# Scipy is a go-to scientific computing library. We'll use it for median filtering. 
from scipy.ndimage import median_filter

# Tensorflow and keras if we choose to do deep learning.
import tensorflow as tf
import tensorflow.keras as k

# moviepy helps open the video files in Python
from moviepy.editor import VideoClip, VideoFileClip
from moviepy.video.io.bindings import mplfig_to_npimage

# Configuring matplotlib to show animations in a colab notebook as javascript 
# objects for easier viewing. 
#from matplotlib import rc
#rc('animation', html='jshtml')

#from google.colab import output
#output.enable_custom_widget_manager()
import ipywidgets
from IPython.display import display


normalized_ske_file_dir   = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\01_BehaviorAtlas_collated_data'

def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if file_name.endswith(content):
            USN = int(file_name.split('-')[1])
            file_path_dict.setdefault(USN,file_dir+'/'+file_name)
    return(file_path_dict)



skip_file_list = [1,3,28,29,110,122] 

def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-'))&(file_name.endswith(content)):
            USN = int(file_name.split('-')[1])
            file_path_dict.setdefault(USN,file_dir+'\\'+file_name)
    return(file_path_dict)

Skeleton_path = get_path(normalized_ske_file_dir,'normalized_skeleton_XYZ.csv')


projectPath = r'D:\yejialin\motionmapper_project_umap\Naive'
if not os.path.exists(projectPath):
    os.mkdir(projectPath)
mmpy.createProjectDirectory(projectPath)


skip_file_list = [1,3,28,29,110,122] 
animal_info_csv = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Table_S1_animal_information.csv'            
animal_info = pd.read_csv(animal_info_csv)                                                                         
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)] 

Morning_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Morning') & (animal_info['LightingCondition']=='Light-on')]
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-off')]

stress_info = animal_info[(animal_info['ExperimentCondition']=='Stress')]


dataset1 = Morning_lightOn_info
dataset1_name = 'Morning_lightOn'

#dataset2 = stress_info
#dataset2_name = 'stress'

#dataset3 = Night_lightOff_info
#dataset3_name = 'Night_lightOff'

datasetnames = []
mouse_posture_list = []
for index in list(dataset1.index):
    video_index = dataset1.loc[index,'video_index']
    ExperimentCondition = dataset1.loc[index,'ExperimentCondition']
    LightingCondition = dataset1.loc[index,'LightingCondition']
    datasetnames.append('{}_{}_{}'.format(ExperimentCondition,LightingCondition,video_index))
    ske_Data = pd.read_csv(Skeleton_path[video_index],index_col=0)
    mouse_posture_list.append(np.array(ske_Data.iloc[:,:]))

# =============================================================================
# for index in list(dataset2.index):
#     video_index = dataset2.loc[index,'video_index']
#     ExperimentCondition = dataset2.loc[index,'ExperimentCondition']
#     LightingCondition = dataset2.loc[index,'LightingCondition']
#     datasetnames.append('{}_{}_{}'.format(ExperimentCondition,LightingCondition,video_index))
#     ske_Data = pd.read_csv(Skeleton_path[video_index])
#     mouse_posture_list.append(np.array(ske_Data.iloc[:,:]))
# =============================================================================


# =============================================================================
# for index in list(dataset3.index):
#     video_index = dataset3.loc[index,'video_index']
#     ExperimentCondition = dataset3.loc[index,'ExperimentCondition']
#     LightingCondition = dataset3.loc[index,'LightingCondition']
#     datasetnames.append('{}_{}_{}'.format(ExperimentCondition,LightingCondition,video_index))
#     ske_Data = pd.read_csv(Skeleton_path[video_index])
#     mouse_posture_list.append(np.array(ske_Data.iloc[:,:]))
# =============================================================================


projs_list = mouse_posture_list

for i,projs in enumerate(projs_list):
    hdf5storage.savemat('%s/Projections/%s_pcaModes.mat'%(projectPath, datasetnames[i]), {'projections':projs})


parameters = mmpy.setRunParameters() 

parameters.projectPath = projectPath #% Full path to the project directory.


parameters.method = 'UMAP' #% We can choose between 'TSNE' or 'UMAP'

parameters.minF = 1        #% Minimum frequency for Morlet Wavelet Transform

parameters.maxF = 15       #% Maximum frequency for Morlet Wavelet Transform,
                           #% usually equals to the Nyquist frequency for your
                           #% measurements.

parameters.samplingFreq = 30    #% Sampling frequency (or FPS) of data.

parameters.numPeriods = 30       #% No. of dyadically spaced frequencies to
                                 #% calculate between minF and maxF.

parameters.pcaModes = 48 #% Number of low-d features.

parameters.numProcessors = 12     #% No. of processor to use when parallel
                                 #% processing for wavelet calculation (if not using GPU)  
                                 #% and for re-embedding. -1 to use all cores 
                                 #% available.

parameters.useGPU = 0           #% GPU to use for wavelet calculation, 
                                 #% set to -1 if GPU not present.

parameters.training_numPoints = 300    #% Number of points in mini-trainings.


# %%%%% NO NEED TO CHANGE THESE UNLESS MEMORY ERRORS OCCUR %%%%%%%%%%

parameters.trainingSetSize = 360  #% Total number of training set points to find. 
                                 #% Increase or decrease based on
                                 #% available RAM. For reference, 36k is a 
                                 #% good number with 64GB RAM.

parameters.embedding_batchSize = 720  #% Lower this if you get a memory error when 
                                        #% re-embedding points on a learned map.

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#% can be 'barnes_hut' or 'exact'. We'll use barnes_hut for this tutorial for speed.
parameters.tSNE_method = 'exact' 

# %2^H (H is the transition entropy)
parameters.perplexity = 32

# %number of neigbors to use when re-embedding
parameters.maxNeighbors = 200

# %local neighborhood definition in training set creation
parameters.kdNeighbors = 5

# %t-SNE training set perplexity
parameters.training_perplexity = 20


# %%%%%%%% UMAP Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Size of local neighborhood for UMAP.
n_neighbors = 10

# Negative sample rate while training.
train_negative_sample_rate = 5

# Negative sample rate while embedding new data.
embed_negative_sample_rate = 1

# Minimum distance between neighbors.
min_dist = 0.1



wlets, freqs = mmpy.findWavelets(projs_list[0], projs_list[0].shape[1], 
                                 parameters.omega0, parameters.numPeriods, 
                                 parameters.samplingFreq, parameters.maxF, 
                                 parameters.minF, parameters.numProcessors, 
                                 parameters.useGPU)

fig, axes = plt.subplots(projs_list[0].shape[1], 1, figsize=(15,10),dpi=300)

for i, ax in enumerate(axes.flatten()):
  ax.imshow(wlets[:300,parameters.numPeriods*i:parameters.numPeriods*(i+1)].T.get(), cmap='PuRd', origin='lower')
  ax.set_yticks(np.arange(parameters.numPeriods, step=4))
  ax.set_yticklabels(['%0.1f'%freqs[j] for j in np.arange(parameters.numPeriods, step=4)])
  if i == 3:
    ax.set_ylabel("Frequencies (hz)", fontsize=14)
  ax.set_title('Projection #%i'%(i+1))
ax.set_xlabel('Frames', fontsize=14)
 

t1 = time.time()

mmpy.subsampled_tsne_from_projections(parameters, parameters.projectPath)

print('Done in %i seconds.'%(time.time()-t1))


trainy = hdf5storage.loadmat('%s/%s/training_embedding.mat'%(parameters.projectPath, parameters.method))['trainingEmbedding']
m = np.abs(trainy).max()


fig, axes = plt.subplots(1, 2, figsize=(12,6))
axes[0].scatter(trainy[:,0], trainy[:,1], marker='.', c=np.arange(trainy.shape[0]), s=1)
axes[0].set_xlim([-m-20, m+20])
axes[0].set_ylim([-m-20, m+20])
axes[0].set_title(trainy.shape)


cbar_ax = fig.add_axes([0.92, 0.3, 0.01, 0.4])


def makeframe(sigma, c_limit):
    ax = axes[1]
    ax.clear()
    _, xx, density = mmpy.findPointDensity(trainy, sigma, 511, [-m-20, m+20])
    im = ax.imshow(density, cmap=mmpy.gencmap(), extent=(xx[0], xx[-1], xx[0], xx[-1]), 
              origin='lower', vmax=np.max(density)*c_limit)
    # ax.axis('off')
    ax.set_title('%0.02f, %0.02f'%(sigma, c_limit))
    cbar_ax.clear()
    fig.colorbar(im, cax=cbar_ax, fraction=c_limit)


w = ipywidgets.FloatSlider
sliders = {'sigma':w(value=1.0, min=0.1, max=3.0, step=0.05, continuous_update=False, 
                     orientation='horizontal', 
                     description='sigma', layout=ipywidgets.Layout(width='1000px')),
        'c_limit':w(value=0.95, min=0.1, max=1.0, step=0.05, continuous_update=False, 
                    orientation='horizontal', 
                    description='C Limit',layout=ipywidgets.Layout(width='1000px'))}

i = ipywidgets.interactive_output(makeframe, sliders)

b = ipywidgets.VBox([sliders['sigma'], sliders['c_limit']])

display(b, i)
plt.show()


tall = time.time()

import h5py
tfolder = parameters.projectPath+'/%s/'%parameters.method

# Loading training data
with h5py.File(tfolder + 'training_data.mat', 'r') as hfile:
    trainingSetData = hfile['trainingSetData'][:].T

# Loading training embedding
with h5py.File(tfolder+ 'training_embedding.mat', 'r') as hfile:
    trainingEmbedding= hfile['trainingEmbedding'][:].T

if parameters.method == 'TSNE':
    zValstr = 'zVals' 
else:
    zValstr = 'uVals'

projectionFiles = glob.glob(parameters.projectPath+'\\Projections\\*pcaModes.mat')


for i in range(len(projectionFiles)):
    print('Finding Embeddings')
    t1 = time.time()
    print('%i/%i : %s'%(i+1,len(projectionFiles), projectionFiles[i]))


    # Skip if embeddings already found.
    if os.path.exists(projectionFiles[i][:-4] +'_%s.mat'%(zValstr)):
        print('Already done. Skipping.\n')
        continue

    # load projections for a dataset
    projections = hdf5storage.loadmat(projectionFiles[i])['projections']

    # Find Embeddings
    zValues, outputStatistics = mmpy.findEmbeddings(projections,trainingSetData,trainingEmbedding,parameters)

    # Save embeddings
    hdf5storage.write(data = {'zValues':zValues}, path = '/', truncate_existing = True,
                    filename = projectionFiles[i][:-4]+'_%s.mat'%(zValstr), store_python_metadata = False,
                      matlab_compatible = True)
    
    # Save output statistics
    with open(projectionFiles[i][:-4] + '_%s_outputStatistics.pkl'%(zValstr), 'wb') as hfile:
        pickle.dump(outputStatistics, hfile)

    del zValues,projections,outputStatistics

print('All Embeddings Saved in %i seconds!'%(time.time()-tall))


def makeframe(sigma, c_limit):
    ax = axes[1]
    ax.clear()
    _, xx, density = mmpy.findPointDensity(trainy, sigma, 511, [-m-20, m+20])
    im = ax.imshow(density, cmap=mmpy.gencmap(), extent=(xx[0], xx[-1], xx[0], xx[-1]), 
              origin='lower', vmax=np.max(density)*c_limit)
    # ax.axis('off')
    ax.set_title('%0.02f, %0.02f'%(sigma, c_limit))
    cbar_ax.clear()
    fig.colorbar(im, cax=cbar_ax, fraction=c_limit)

for i in glob.glob(parameters.projectPath+'/Projections/*%s*_%s*.mat'%('Morning',zValstr)):
    ally = hdf5storage.loadmat(i)['zValues']

    m = np.abs(ally).max()
    
    fig, axes = plt.subplots(1, 2, figsize=(12,6))
    axes[0].scatter(ally[:,0], ally[:,1], marker='.', c=np.arange(ally.shape[0]), s=1)
    axes[0].set_xlim([-m-20, m+20])
    axes[0].set_ylim([-m-20, m+20])
    axes[0].set_title(ally.shape)
    plt.suptitle(i)
    cbar_ax = fig.add_axes([0.92, 0.3, 0.01, 0.4])
    
    
    def makeframe(sigma, c_limit):
        ax = axes[1]
        ax.clear()
        _, xx, density = mmpy.findPointDensity(ally, sigma, 511, [-m-20, m+20])
        im = ax.imshow(density, cmap=mmpy.gencmap(), extent=(xx[0], xx[-1], xx[0], xx[-1]), 
                  origin='lower', vmax=np.max(density)*c_limit)
        # ax.axis('off')
        ax.set_title('%0.02f, %0.02f'%(sigma, c_limit))
        cbar_ax.clear()
        fig.colorbar(im, cax=cbar_ax, fraction=c_limit)
        
    
    w = ipywidgets.FloatSlider
    sliders = {'sigma':w(value=1.0, min=0.1, max=3.0, step=0.05, continuous_update=False, 
                         orientation='horizontal', 
                         description='sigma', layout=ipywidgets.Layout(width='1000px')),
            'c_limit':w(value=0.95, min=0.1, max=1.0, step=0.05, continuous_update=False, 
                        orientation='horizontal', 
                        description='C Limit',layout=ipywidgets.Layout(width='1000px'))}
    
    i = ipywidgets.interactive_output(makeframe, sliders)
    
    b = ipywidgets.VBox([sliders['sigma'], sliders['c_limit']])
    
    display(b, i)
    
    plt.show()

# decrease the startsigma values by 0.5 at a time if you get errors.
startsigma = 3 if parameters.method == 'TSNE' else 1.0

[os.remove(i) for i in glob.glob(r'%s\\%s\\zWshed*.png'%(parameters.projectPath, parameters.method))]

mmpy.findWatershedRegions(parameters, minimum_regions=14, startsigma=startsigma, pThreshold=[0.33, 0.67],
                     saveplot=True, endident = r'*_pcaModes.mat')

from IPython.display import Image
Image(glob.glob(r'%s\\%s\\zWshed*.png'%(parameters.projectPath, parameters.method))[0])




